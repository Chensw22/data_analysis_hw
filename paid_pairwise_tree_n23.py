"""Train ranking trees on outcome-discordant payers with and without matching teacher confidence."""
import numpy as np
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize
from exaone_validation_data import CATEGORICAL
from group_crossfit_residual_discovery_n23 import teachers
from paid_conditional_tilt_n23 import REPORT, rank_accuracy
from released_correction_audit_n23 import development


def pairs(target, group, risk, matched):
    rng=np.random.default_rng(256)
    positives=np.flatnonzero(target)
    negatives=np.flatnonzero(~target)
    result=[]
    for i in positives:
        other=negatives[group[negatives]!=group[i]]
        if matched:
            gap=abs(risk[i]-risk[other])
            other=other[np.argpartition(gap,39)[:40]]
        sample=rng.choice(other,8,replace=False)
        result.extend((i,j) for j in sample)
    a=np.asarray(result)
    assert (target[a[:,0]]&~target[a[:,1]]).all() and (group[a[:,0]]!=group[a[:,1]]).all()
    return a


def fold_scores(frame, y, group, paid, t, q, f):
    pt,pq=teachers(frame,y,group,t,q,376+f)
    sub=paid[t]
    source=t[sub]
    target=y[source]
    learner=dict(iterations=260,depth=5,learning_rate=.065,random_seed=153,
                 thread_count=2,verbose=False,allow_writing_files=False)
    # The identical ordinary classifier, unpaired, uses exactly the same payers.
    data=Pool(frame.iloc[source],target,cat_features=CATEGORICAL)
    query=Pool(frame.iloc[q],cat_features=CATEGORICAL)
    ordinary=CatBoostClassifier(**learner).fit(data).predict(query,prediction_type='RawFormulaVal')
    out=dict(teacher=logit(pq.mean(1).clip(.002,.998)),ordinary=ordinary)
    gaps={}
    for key in ['matched','random']:
        order=pairs(target,group[source],pt[sub],key=='matched')
        training=Pool(frame.iloc[source],target.astype(int),cat_features=CATEGORICAL,pairs=order)
        model=CatBoostRanker(loss_function='PairLogit',**learner).fit(training)
        out[key]=model.predict(query)
        gaps[key]=float(abs(pt[sub][order[:,0]]-pt[sub][order[:,1]]).mean())
    return out,gaps


def main():
    row,y,frame,g,_=load()
    ids,target,masks=historical()
    np.testing.assert_array_equal(row,ids)
    np.testing.assert_array_equal(y,target)
    _,releases,_=development()
    paid=strata(frame)==2
    masks={**masks,'paid_transported':masks['repeat']&paid&y,'never_paid':masks['never']&paid&y}
    directory=ART/'nov23_paid_pairwise_tree'
    directory.mkdir(exist_ok=True)
    hist,refs,plans={},[],{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,base,ref=z['fold'],z['base'],z['ref']
        np.testing.assert_array_equal(ref,releases[131] if seed=='v131' else base>=.5)
        eligible=paid&~ref&(base<.5)
        fields,gaps={},[]
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            outs,summary=fold_scores(frame,y,g,paid,t,q,int(f))
            for k,s in outs.items():fields.setdefault(k,np.zeros(len(y)))[q]=s
            gaps.append(summary)
        stats={}
        anchor=ref==y
        if seed!='v131':refs.append(anchor)
        for mode in ['ordinary','matched','random']:
            x=np.c_[logit(base.clip(.002,.998)),fields['teacher'],fields['ordinary']]
            if mode!='ordinary':
                x=np.c_[x,fields[mode]]
            p=base.copy()
            for f in np.unique(fold):
                ix=np.flatnonzero((fold==f)&eligible)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
                    t,q=ix[a],ix[b]
                    p[q]=calibrated(x[t],x[q],y,t)
            vote=ref.copy()
            vote[eligible]=p[eligible]>=.5
            h=vote==y
            baseline_hits=0
            for f in np.unique(fold):
                ix=np.flatnonzero((fold==f)&eligible)
                n=int(vote[ix].sum())
                baseline_hits+=int(y[ix[np.argsort(-base[ix],kind='stable')[:n]]].sum())
            stats[mode]=dict(accuracy=float(h.mean()),fixed=int((h&~anchor).sum()),harmed=int((~h&anchor).sum()),
                             equal_count_base_fixed=baseline_hits,matched_pairs=rank_accuracy(p,base,y,eligible))
            if seed!='v131':hist.setdefault(mode,[]).append(h)
        plans[seed]=dict(models=stats,training_risk_gaps=gaps)
        np.savez_compressed(directory/f'{seed}.npz',row=row,y=y,fold=fold,base=base,ref=ref,**fields)
        print(seed,stats['matched'],flush=True)
    write(REPORT/'pairs.json',dict(plans=plans,grouped={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in hist.items()}))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
