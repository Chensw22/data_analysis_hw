"""Average independent odds corrections for paid negatives and zero-spending positives."""
import numpy as np
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize
from exaone_validation_data import CATEGORICAL
from paid_conditional_tilt_n23 import REPORT, features, tilted_odds
from released_correction_audit_n23 import development


def independent_evidence(frame,y,group,state,t,q):
    results={k:[] for k in ('teacher','paid','paid_control','zero','zero_control')}
    for seed in [545,765]:
        for teacher_ids,adjust_ids in GroupKFold(2,shuffle=True,random_state=seed).split(t,groups=group[t]):
            a,b=t[teacher_ids],t[adjust_ids]
            for first,second in [(a,b),(a,q),(b,q)]:
                assert not np.intersect1d(group[first],group[second]).size
            teacher=CatBoostClassifier(iterations=380,depth=5,learning_rate=.06,random_seed=191,
                    thread_count=2,verbose=False,allow_writing_files=False)
            teacher.fit(frame.iloc[a], y[a],cat_features=CATEGORICAL)
            pr=np.full(len(y),np.nan)
            use=np.r_[b,q]
            pr[use]=teacher.predict_proba(frame.iloc[use])[:,1]
            results['teacher'].append(logit(pr[q].clip(.002,.998)))
            for key,labels,risk,region in [('paid',y,pr,state==2),('zero',~y,1-pr,state==0)]:
                train=b[region[b]&(risk[b]<.5)]
                query=q[region[q]]
                x=features(frame,train)
                for shuffled,suffix in [(False,''),(True,'_control')]:
                    column=np.zeros(len(q))
                    column[region[q]]=tilted_odds(x,risk,labels,train,query,sham=shuffled)
                    results[key+suffix].append(column)
    return {k:np.stack(v,axis=1) for k,v in results.items()}


def main():
    row,y,frame,g,_=load()
    ids,target,masks=historical()
    np.testing.assert_array_equal(row,ids)
    np.testing.assert_array_equal(y,target)
    _,release,_=development()
    state=strata(frame)
    masks.update(paid_transported=masks['repeat']&(state==2)&y,zero_stayed=masks['repeat']&(state==0)&~y)
    out=ART/'nov23_crossfit_tilt_committee'
    out.mkdir(exist_ok=True)
    hist,refs,plans={},[],{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,base,ref=z['fold'],z['base'],z['ref']
        np.testing.assert_array_equal(ref,release[131] if seed=='v131' else base>=.5)
        raw={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            for k,v in independent_evidence(frame,y,g,state,t,q).items():
                raw.setdefault(k,np.zeros((len(y),v.shape[1])))[q]=v
        outputs={}
        anchor=ref==y
        if seed!='v131':refs.append(anchor)
        for direction in ['paid','zero']:
            pos=direction=='zero'
            select=(state==(0 if pos else 2))&(ref==pos)&(ref==(base>=.5))
            risk=1-base if pos else base
            teacher=raw['teacher'].mean(1)*(-1 if pos else 1)
            basic=np.c_[logit(risk.clip(.002,.998)),teacher]
            families=dict(plain=basic,one=np.c_[basic,raw[direction][:,0]],
                committee=np.c_[basic,raw[direction].mean(1)],
                shuffled=np.c_[basic,raw[direction+'_control'].mean(1)])
            for mode,x in families.items():
                vote=outputs.setdefault(mode,ref.copy())
                for f in np.unique(fold):
                    ix=np.flatnonzero(select&(fold==f))
                    for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
                        t,q=ix[a],ix[b]
                        flip=calibrated(x[t],x[q],y!=ref,t)>=.5
                        vote[q]=ref[q]^flip
        metrics={}
        for mode,vote in outputs.items():
            hit=vote==y
            def paired(select):
                return dict(fixed=int((hit&~anchor&select).sum()),harmed=int((~hit&anchor&select).sum()))
            metrics[mode]=dict(accuracy=float(hit.mean()),paid=paired(state==2),zero=paired(state==0),
                               **paired(np.ones(len(y),bool)))
            if seed!='v131':hist.setdefault(mode,[]).append(hit)
        plans[seed]=metrics
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=base,**raw,**outputs)
        print(seed,metrics['committee'],flush=True)
    write(REPORT/'committee.json',dict(plans=plans,grouped={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in hist.items()}))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
