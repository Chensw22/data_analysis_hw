"""Cross-fit continuous arbitration between the queued decision and an opposing spline."""
import argparse
import hashlib
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from booking_spline_network_n26 import CONFIG, METHODS, OUT as PREVIOUS, basis, fit_neural, splits
from full_roster_booking_validation_n23 import ROOT, data
from tabnet_history_test_n26 import historical_indices

OUT = ROOT/'artifacts/nov26_historical_spline_comparator'
REPORT = ROOT/'reports/nov23_matched_learning_curve/historical_spline_comparator.json'


def train():
    import torch
    torch.set_num_threads(1)
    raw, y, g, _ = data()
    OUT.mkdir(exist_ok=True)
    for k in range(3):
        t, q, _ = historical_indices(raw,y,g,k)
        x = basis(raw,g,t)
        scores = {'additive':LogisticRegression(C=1,max_iter=1000).fit(x[t],y[t]).predict_proba(x[q])[:,1]}
        for m in METHODS[1:]:
            scores[m] = np.stack([fit_neural(x,y,t,q,seed,m) for seed in CONFIG['seeds']])
        np.savez_compressed(OUT/f'historical{k}.npz',t=t,q=q,train_y=y[t],
                            passenger=raw.PassengerId.iloc[q].to_numpy(str),**scores)
        print(k,len(t),len(q),flush=True)


def packets(raw,y,g,fold):
    old=np.full((len(y),5),np.nan)
    full=np.full_like(old,np.nan)
    new=np.full_like(old,np.nan)
    receipts={}
    def load(path,t,q,target):
        with np.load(path) as z:
            for name,value in [('t',t),('q',q),('passenger',raw.PassengerId.iloc[q]),('train_y',y[t])]:
                np.testing.assert_array_equal(z[name],value)
            assert not np.isin(g[t],g[q]).any() and np.isnan(target[q]).all()
            target[q]=np.column_stack([z['learned_spline'].T,z['silu'].T,z['additive']])
            assert np.isfinite(target[q]).all() and ((target[q]>=0)&(target[q]<=1)).all()
        receipts[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for k in range(3):
        t,q,_=historical_indices(raw,y,g,k)
        load(OUT/f'historical{k}.npz',t,q,old)
    for tag,t,q in splits(raw,g,fold):
        load(PREVIOUS/(tag+'.npz'),t,q,full if tag!='history_to_new' else new)
    return old,full,new,receipts


def design(p,parent):
    mean=np.column_stack([p[:,:2].mean(1),p[:,2:4].mean(1),p[:,4]])
    signed=logit(mean.clip(.002,.998))*np.where(parent,-1,1)[:,None]
    logits=logit(p.clip(.002,.998))
    seed_gaps=abs(logits[:,[0,2]]-logits[:,[1,3]])
    return np.column_stack([signed,seed_gaps,parent]),(mean[:,0]>=.5)!=parent


def fit_compare(train,parent,wrong,query,now):
    xt,use=design(train,parent)
    xq,dissent=design(query,now)
    models={}
    weights={}
    for name,size in [('spline_only',1),('compared',xt.shape[1])]:
        scale=xt[use,:size].std(0).clip(.15)
        center=xt[use,:size].mean(0)
        X=(xt[use,:size]-center)/scale
        Q=(xq[:,:size]-center)/scale
        assert np.unique(wrong[use]).size==2
        fitted=LogisticRegression(C=.2,max_iter=500).fit(X,wrong[use])
        models[name]=fitted.predict_proba(Q)[:,1]
        weights[name]=dict(intercept=float(fitted.intercept_[0]),mean=center.tolist(),standard_deviation=scale.tolist(),
                           coefficients=fitted.coef_[0].tolist())
    return models,dissent,dict(rows=len(train),disagreements=int(use.sum()),spline_correct=int(wrong[use].sum()),models=weights)


def describe(q,scope,action,parent,y,fold):
    used=action&scope
    wrong=(parent!=y[q])
    gain=(2*wrong.astype(int)-1)*used
    return dict(rows=int(scope.sum()),parent_errors=int((wrong&scope).sum()),fixed=int((gain==1).sum()),harmed=int((gain==-1).sum()),
                net=int(gain.sum()),overall_gain_pp=float(100*gain.sum()/max(scope.sum(),1)),
                by_fold=[int(gain[fold[q]==k].sum()) for k in range(8)])


def evaluate():
    from sklearn.metrics import log_loss,roc_auc_score
    from audit_legacy_queue_reach_n26 import contemporary
    from distinct_challenger_error_reach_n26 import population
    from tabnet_attention_transfer_n26 import collect
    from target_mix_expert_routing_n26 import batches,sources
    raw,y,g,fold=data()
    old,full,new,receipts=packets(raw,y,g,fold)
    panels=sources()[-1]
    historic=panels['historical']
    h=historic['ids']
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        old_versions,hf,cold=contemporary(h,z['v131'],g[h].astype(int))
    np.testing.assert_array_equal(hf,historic['fold'])
    oldparent=old_versions['six']
    approved=cold & historic['known']
    # Cold source customers have no order members on other historical folds even in
    # older reference components which predate the booking-purged caches.
    history_fit=h[approved]
    master,truth,groups,split,_=collect()
    for a,b in [(y,truth),(g,groups),(fold,split)]:
        np.testing.assert_array_equal(a,b)
    result={}
    fitted={}
    saved={}
    segments=[]
    for b in batches(True):
        if b.tag=='bookingmix':
            continue
        j=0 if b.tag=='historical_to_new' else len(segments)
        q,common,base,actions=master[b.tag][j]
        np.testing.assert_array_equal(q,b.ids)
        queued=base^(common & actions['actual_and_scores'])
        if b.tag=='historical_to_new':
            t=history_fit
            src=oldparent[approved]
            dst=new[q]
            probs,dissent,record=fit_compare(old[t],src,y[t]!=src,dst,queued)
            key='history_to_new'
        else:
            t=b.train_ids
            src=b.train_cell%2!=0
            assert np.unique(fold[np.r_[t,q]]).size==1
            # src are nested out-of-order six-way decisions inside the calibration
            # sample; using global eight-fold queued decisions here would make
            # these features depend on the responses in q.
            np.testing.assert_array_equal(b.train_error,y[t]!=src)
            probs,dissent,record=fit_compare(full[t],src,b.train_error,full[q],queued)
            key=f'full_{j}'
        assert not np.isin(g[t],g[q]).any()
        if b.tag=='historical_to_new':
            assert not np.isin(g[h],g[q]).any()
        fitted[key]=record
        group=segments if b.tag=='full' else []
        group.append((q,common,base,queued,dissent,probs['spline_only'],probs['compared']))
        if b.tag=='historical_to_new':
            saved['history_to_new']=group
    saved['eight_fold']=segments
    _,yy,_,ff,former_parent,shared,count,_,_,wrong,_,_=population()
    for a,b in [(y,yy),(fold,ff)]:
        np.testing.assert_array_equal(a,b)
    blind=shared&(count==0)&wrong
    oldwins=blind & ((full[:,:2].mean(1)>=.5)==y)
    assert blind.sum()==172 and oldwins.sum()==2
    additionally={}
    for key,parts in saved.items():
        q,common,base,queued,dissent,sole,compared=(np.concatenate([x[k] for x in parts]) for k in range(7))
        assert len(np.unique(q))==len(q)
        p=full[q] if key=='eight_fold' else new[q]
        np.testing.assert_array_equal(dissent,(p[:,:2].mean(1)>=.5)!=queued)
        change={'take_every_dissent':dissent,'spline_only':dissent&(sole>.5),
                'compared':dissent&(compared>.5),'added_by_controls':dissent&(compared>.5)&(sole<=.5),
                'rejected_by_controls':dissent&(compared<=.5)&(sole>.5)}
        scopes={'all':np.ones(len(q),bool),'ten_agree_queue_untouched':common&(queued==base),'any_queue_stage_differs':~common|(queued!=base)}
        if key=='eight_fold':
            scopes['formerly_blind_172_and_their_correct_peers']=shared[q]&(count[q]==0)
        result[key]={name:{tag:describe(q,scope,a,queued,y,fold) for tag,a in change.items()} for name,scope in scopes.items()}
        risk=(queued!=y[q])[dissent]
        result[key]['scores_on_disagreements']={tag:dict(n=len(risk),auc=float(roc_auc_score(risk,prob[dissent])),
                log_loss=float(log_loss(risk,prob[dissent]))) for tag,prob in [('spline_only',sole),('compared',compared)]}
        np.savez_compressed(OUT/(key+'_decisions.npz'),q=q,passenger=raw.PassengerId.iloc[q].to_numpy(str),parent=queued,
                            common=common,queued_changed=base!=queued,dissent=dissent,spline_only=sole,compared=compared)
        additionally[key]=dict(original_two_present=int(oldwins[q].sum()),
            calibrated_changes_on_original_two=int((oldwins[q]&change['compared']).sum()))
    covered=np.isfinite(old[:,0])|np.isfinite(new[:,0])
    repeated=0
    for i in np.flatnonzero(oldwins&covered):
        rep=old[i] if np.isfinite(old[i,0]) else new[i]
        repeated+=int((rep[:2].mean()>=.5)==y[i])
    output=dict(configuration='Logistic regression C=.2, predictor standardization fitted only on calibration rows; probability >.5 with no tuning. Use all actual spline-parent disagreements to learn acceptance. Controls are oriented mean spline/SiLU/additive log-odds, unsigned per-network absolute seed-logit gaps and parent direction. The matched baseline has only oriented spline log-odds.',
       history='Full source of 6954 passengers has three out-of-order neural predictions, with every validation booking purged from support. Fit using known, fully cold historical six-way references and their disagreements; the match against the final queue is approximate in calibration because a TabNet-routed historical OOF parent is unavailable. Modern outcomes use the independently reconstructed queued parent in each of the two protocols.',
       fresh='The comparator and all of its three input models exclude every query booking and its labels. Target queue is collect.historical_to_new, 1033 of 1044 unique fresh passengers known. The queued reference itself includes some frozen full-OOF experts; fresh labels are never used to select comparator settings.',
       matched='For 8 outer group folds, split the held fold in 3 meta parts. Every underlying spline/SiLU/additive sees only the same separate outer-training labels. The calibration target is the nested six-way reference inside the other 2 held parts; no query-order labels enter that training. Each query is evaluated against the correctly reconstructed held-subset queued TabNet. Protocols overlap.',
       historical_source_rows=int(len(history_fit)),historical_total_oof=len(h),inputs_sha256=receipts, fits=fitted,results=result,
       original_two=dict(have_second_independent_booking_heldout_prediction=int((oldwins&covered).sum()),raw_spline_again_correct=repeated,
                         original_blind_errors=172,raw_eight_fold_successes=2,calibrated=additionally))
    REPORT.write_text(json.dumps(output,indent=2)+'\n')
    for key,r in result.items():
        print(key, {m:(v['fixed'],v['harmed']) for m,v in r['all'].items()},r['scores_on_disagreements'],flush=True)
    print('two',output['original_two'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','evaluate'])
    args=parser.parse_args()
    with threadpool_limits(1):
        if args.action=='train':
            train()
        else:
            evaluate()
