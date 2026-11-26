"""Validate aligned full-fold RT-J outputs and compare its relational and no-edge readouts."""
import hashlib
import json

import numpy as np
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from exhaustive_family_quorums_n26 import inputs
from full_roster_booking_validation_n23 import OUT as BASE,ROOT,data
from rtj_checkpoint_setup_n26 import OUT,WEIGHTS


def counts(pred,baseline,y,mask):
    ok=pred==y
    old=baseline==y
    plus=int((mask&ok&~old).sum())
    minus=int((mask&~ok&old).sum())
    return dict(rows=int(mask.sum()),original_correct=int((mask&old).sum()),correct=int((mask&ok).sum()),
                fixed=plus,harmed=minus,net=plus-minus)


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
    report=dict(completed_folds=[],files={},fold_results={})
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
                scores[ids,j]=z['logits']
                context[ids,j]=z['context_labels']
                assert np.isfinite(scores[ids,j]).all() and np.std(scores[ids,j])>0
                report['files'][path.name]=dict(rows=len(ids),seconds=float(z['elapsed']),
                    min_median_max_context_labels=np.quantile(z['context_labels'],[0,.5,1]).tolist(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report['fold_results'][str(k)]=dict(accuracy_edges=float(np.mean((scores[ids,0]>0)==y[ids])),
                accuracy_no_edges=float(np.mean((scores[ids,1]>0)==y[ids])))
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
    pred={name:np.zeros(len(y),bool) for name in modes}
    for k in report['completed_folds']:
        ids=np.flatnonzero(f==k)
        for a,b in GroupKFold(4,shuffle=True,random_state=5128).split(ids,groups=g[ids]):
            t,q=ids[a],ids[b]
            assert not np.isin(g[t],g[q]).any()
            for name,x in zip(modes,[om[:,None],np.column_stack([om,scores[:,0]]),
                                     np.column_stack([om,scores[:,1]]),np.column_stack([om,scores]),core,enriched],strict=True):
                fit=LogisticRegression(C=.4,max_iter=300).fit(x[t],y[t])
                pred[name][q]=fit.predict(x[q])
    pred['edges']=scores[:,0]>0
    pred['no_edges']=scores[:,1]>0
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
    report['scope']='Each retained fold has two complete matched RT-J files. Query labels are placeholders through sampling and inference. Personal cabin, deck, side, party size and planet information is matched; no_edges removes passenger-to-booking and passenger-to-cabin graph links. Both branches retain planet links. Meta-model fits use separate bookings inside the same base fold.'
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
