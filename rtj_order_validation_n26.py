"""Validate aligned full-fold RT-J outputs and compare its relational and no-edge readouts."""
import hashlib
import json

import numpy as np
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from exhaustive_family_quorums_n26 import inputs
from full_roster_booking_validation_n23 import OUT as BASE,ROOT,data
from rtj_checkpoint_setup_n26 import OUT,WEIGHTS
from rtj_order_smoke_n26 import CONTEXT_POLICY


def counts(pred,baseline,y,mask):
    ok=pred==y
    old=baseline==y
    plus=int((mask&ok&~old).sum())
    minus=int((mask&~ok&old).sum())
    return dict(rows=int(mask.sum()),original_correct=int((mask&old).sum()),correct=int((mask&ok).sum()),
                fixed=plus,harmed=minus,net=plus-minus)


def stability(pred, baseline, y, mask, bookings, folds):
    """Paired accuracy change and a booking-clustered Wald interval, conditional on fitted models."""
    difference=(pred[mask]==y[mask]).astype(float)-(baseline[mask]==y[mask]).astype(float)
    groups,inverse=np.unique(bookings[mask],return_inverse=True)
    average=difference.mean()
    residual=np.bincount(inverse,weights=difference-average)
    se=np.sqrt(np.dot(residual,residual)*len(groups)/(len(groups)-1))/len(difference)
    return dict(rows=len(difference),bookings=len(groups),net=int(difference.sum()),delta_pp=100*average,
                conditional_95_interval_pp=[100*(average-1.96*se),100*(average+1.96*se)],
                net_by_fold={str(k):int(difference[folds[mask]==k].sum()) for k in np.unique(folds[mask])})


def evaluate():
    raw,y,g,f=data()
    with np.load(BASE/'mitra96.npz') as z:
        for key,expected in [('passenger',raw.PassengerId.to_numpy(str)),('fold',f),('y',y)]:
            np.testing.assert_array_equal(z[key],expected)
        p=z['p']
    _,current,_=inputs()
    parent=current['parent']
    scores=np.full((len(y),2),np.nan)
    context=np.full_like(scores,np.nan)
    report=dict(completed_folds=[],files={},fold_results={},context_policy=CONTEXT_POLICY,
                inference=dict(context_cells=512,local_context=96,bfs_width=16,random_walks=512,walk_length=12,
                               quickstart_context_cells=8192,upstream='https://github.com/stanford-star/relational-transformer/blob/cf91f4566c685043fb3db560dd44b237d494a7f3/notebooks/quickstart.ipynb'),
                prior_run='Superseded unmasked diagnostics remain in artifacts/nov26_rtj_probe/oof_placeholder_diagnostic. The released static-graph sampler can place other query rows’ zero placeholders in context. Current results mask all foreign query task cells before model forward and verify that every visible context label belongs to the training range.')
    for k in np.unique(f):
        ids=np.flatnonzero(f==k)
        entries=[OUT/'oof'/f'{k}_{mode}.npz' for mode in ('edges','no_edges')]
        if not all(path.exists() for path in entries):continue
        report['completed_folds'].append(int(k))
        assert not np.isin(g[ids],g[f!=k]).any()
        for j,path in enumerate(entries):
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],raw.PassengerId.iloc[ids].to_numpy(str))
                np.testing.assert_array_equal(z['y'],y[ids])
                np.testing.assert_array_equal(z['index_order'],np.arange(len(ids)))
                assert int(z['train_bookings'])==len(set(g[f!=k])) and str(z['weights'])==WEIGHTS
                assert str(z['context_policy'])==CONTEXT_POLICY and int(z['ctx'])==report['inference']['context_cells']
                scores[ids,j]=z['logits']
                context[ids,j]=z['context_labels']
                assert np.isfinite(scores[ids,j]).all() and np.std(scores[ids,j])>0
                report['files'][path.name]=dict(rows=len(ids),seconds=float(z['elapsed']),
                    masked_query_context_cells=int(z['masked_query_context_cells'].sum()),context_policy=CONTEXT_POLICY,
                    min_median_max_context_labels=np.quantile(z['context_labels'],[0,.5,1]).tolist(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report['fold_results'][str(k)]=dict(accuracy_edges=float(np.mean((scores[ids,0]>0)==y[ids])),
                accuracy_no_edges=float(np.mean((scores[ids,1]>0)==y[ids])),
                auc_edges=float(roc_auc_score(y[ids],scores[ids,0])),
                auc_no_edges=float(roc_auc_score(y[ids],scores[ids,1])))
    known=np.isfinite(scores).all(1)
    assert known.any()
    mitra=p>=.5
    om=logit(p.clip(.001,.999))
    modes=['Mitra_calibrated','Mitra_plus_RTJ_edges','Mitra_plus_RTJ_no_edges','Mitra_plus_both',
           'Mitra_plus_context_only','Mitra_plus_routed_context']
    # Context sizes are distinct properties of each query; use a fixed transform
    # and a matched arm to distinguish informative scores from retrieval density.
    strength=np.log1p(context)/3
    core=np.column_stack([om,strength,om[:,None]*strength])
    enriched=np.column_stack([core,scores,scores*strength,scores*np.abs(om[:,None])/3])
    fitted_modes=modes+['RTJ_edges_calibrated','RTJ_no_edges_calibrated']
    features=[om[:,None],np.column_stack([om,scores[:,0]]),np.column_stack([om,scores[:,1]]),
              np.column_stack([om,scores]),core,enriched,scores[:,[0]],scores[:,[1]]]
    pred={name:np.zeros(len(y),bool) for name in fitted_modes}
    for k in report['completed_folds']:
        ids=np.flatnonzero(f==k)
        for a,b in GroupKFold(4,shuffle=True,random_state=5128).split(ids,groups=g[ids]):
            t,q=ids[a],ids[b]
            assert not np.isin(g[t],g[q]).any()
            for name,x in zip(fitted_modes,features,strict=True):
                fit=LogisticRegression(C=.4,max_iter=300).fit(x[t],y[t])
                pred[name][q]=fit.predict(x[q])
    pred['edges']=scores[:,0]>0
    pred['no_edges']=scores[:,1]>0
    both_signs_agree=pred['edges']==pred['no_edges']
    joint_confidence=np.nan_to_num(expit(np.abs(scores)).min(axis=1))
    for name,scope in [('available',known),('known_frozen',known&current['known']),
                       ('eleven_agree',known&current['common'])]:
        report[name]={k:dict(against_Mitra=counts(v,mitra,y,scope),
           against_calibrated_Mitra=counts(v,pred[modes[0]],y,scope),
           incremental_flips_only=counts(np.where(pred[modes[0]]==parent,v,parent),parent,y,scope&current['known']),
           against_TabNet_release=counts(v,parent,y,scope&current['known'])) for k,v in pred.items()}
        report[name]['edge_ablation']=counts(pred['edges'],pred['no_edges'],y,scope)
        report[name]['context_aware_vs_matched']=counts(pred[modes[-1]],pred[modes[-2]],y,scope)
        parent_control=pred[modes[-2]]
        report[name]['context_aware_new_challenge']=counts(np.where(parent_control==parent,pred[modes[-1]],parent),
                                                           parent,y,scope&current['known'])
        routed=np.where(parent_control==parent,pred[modes[-1]],parent)
        report[name]['stability']=dict(
            edges_against_no_edges=stability(pred['edges'],pred['no_edges'],y,scope,g,f),
            both_plus_Mitra_against_calibrated_Mitra=stability(pred['Mitra_plus_both'],pred[modes[0]],y,scope,g,f),
            contextual_new_challenges_against_release=stability(routed,parent,y,scope&current['known'],g,f))
        report[name]['dual_checkpoint_confidence_challenges']={}
        for threshold in (.50,.60,.65,.70,.75):
            active=scope&current['known']&both_signs_agree&(joint_confidence>=threshold)&(pred['edges']!=parent)
            report[name]['dual_checkpoint_confidence_challenges'][str(threshold)]=counts(pred['edges'],parent,y,active)
    report['scope']='Each retained fold has two matched RT-J files. Neither real query outcomes nor other queries’ placeholders reach the model. Before each forward pass, padding masks block all other query task rows and an assertion checks every visible context label lies inside the labeled training node range. Personal cabin, deck, side, party size and planet information are matched. Both branches retain the home-planet graph. Meta-model fits use separate bookings inside the same base fold.'
    cache=OUT/'oof/validated_probabilities.npz'
    temp_cache=cache.with_suffix('.tmp.npz')
    np.savez_compressed(temp_cache,passenger=raw.PassengerId.to_numpy(str)[known],fold=f[known],y=y[known],
                        logit=scores[known],probability=expit(scores[known]),context_label_count=context[known],
                        mode=['edges','no_edges'],weights=WEIGHTS)
    temp_cache.replace(cache)
    report['probability_cache']=str(cache.relative_to(ROOT))
    report['probability_cache_sha256']=hashlib.sha256(cache.read_bytes()).hexdigest()
    dest=ROOT/'reports/nov23_matched_learning_curve/rtj_order_validation.json'
    temp=dest.with_suffix('.tmp')
    temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(dest)
    print(report['completed_folds'], report['fold_results'],flush=True)


if __name__=='__main__':
    with threadpool_limits(1):evaluate()
