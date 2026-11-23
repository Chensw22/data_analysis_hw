"""Measure the effects of revealing labels for correctly predicted and mispredicted training groups of passengers."""
from itertools import permutations
import numpy as np
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,SEEDS,historical,write
from broad_prediction_assisted_training_n23 import REPORT
from broad_release_miss_coverage_n23 import summarize
from exaone_validation_data import CATEGORICAL
from group_crossfit_residual_discovery_n23 import teachers
from released_correction_audit_n23 import development

COMPONENTS=['confirmation','missed_positive','missed_negative']


def label_arms(frame,y,g,t,q,f):
    pt,pq=teachers(frame,y,g,t,q,432+int(f))
    truth=y[t]
    direction=pt>=.5
    segment=np.where(truth==direction,0,np.where(truth,1,2))
    outcomes=[]
    for mask in range(8):
        reveal=(mask & (1<<segment))!=0
        response=np.where(reveal,truth,pt)
        if mask==7:np.testing.assert_array_equal(response,truth)
        if mask==0:np.testing.assert_array_equal(response,pt)
        clf=CatBoostClassifier(iterations=320,depth=5,learning_rate=.07,random_seed=279,
                               thread_count=2,loss_function='CrossEntropy',allow_writing_files=False,verbose=False)
        clf.fit(frame.iloc[t],response,cat_features=CATEGORICAL)
        p=clf.predict_proba(frame.iloc[q])[:,1]
        outcomes.append(p)
    return np.stack(outcomes,axis=1),pq.mean(1),np.bincount(segment,minlength=3).tolist()


def attribute(hits,masks):
    # Net correct passengers attributed across every possible order of revealing the three label types.
    out={}
    for label,rows in masks.items():
        counts=hits[rows].sum(0)
        credits=np.zeros(3)
        for order in permutations(range(3)):
            mask=0
            for k in order:
                nxt=mask|(1<<k)
                credits[k]+=(counts[nxt]-counts[mask])/6
                mask=nxt
        np.testing.assert_allclose(credits.sum(),counts[7]-counts[0],atol=1e-9)
        out[label]=dict(rows=int(rows.sum()),soft_correct=int(counts[0]),true_correct=int(counts[7]),
                        contributions=dict(zip(COMPONENTS,credits.tolist())))
    return out


def main():
    row,y,frame,g,_=load()
    rows,target,masks=historical()
    np.testing.assert_array_equal(rows,row)
    np.testing.assert_array_equal(target,y)
    _,released,_=development()
    state=strata(frame)
    masks={**masks,'paid_transported':masks['repeat']&(state==2)&y,'zero_stayed':masks['repeat']&(state==0)&~y}
    out=ART/'nov23_training_error_factorial'
    out.mkdir(exist_ok=True)
    plans,hist,refs={},{},[]
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            for k,v in [('row',row),('y',y)]:np.testing.assert_array_equal(z[k],v)
            fold,base,ref=z['fold'],z['base'],z['ref']
        np.testing.assert_array_equal(ref,released[131] if seed=='v131' else base>=.5)
        prob=np.zeros((len(y),8))
        teacher=np.zeros(len(y))
        sizes=[]
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            prob[q],teacher[q],counts=label_arms(frame,y,g,t,q,f)
            sizes.append(counts)
        votes={str(i):prob[:,i]>=.5 for i in range(8)}
        def odds(p):
            return logit(p.clip(.002,.998))
        core=np.c_[odds(base),odds(teacher),odds(prob[:,0]),odds(prob[:,7])]
        # Complement pairs provide the average score response to each label type, including interactions.
        delta=[]
        for i in range(3):
            pairs=[(k,k|(1<<i)) for k in range(8) if not (k&(1<<i))]
            delta.append(np.mean([odds(prob[:,b])-odds(prob[:,a]) for a,b in pairs],axis=0))
        # The raw arm probabilities can themselves be averaged without any trained gating.
        votes['average']=prob.mean(1)>=.5
        for mode in ['endpoints','label_types']:
            x=core if mode=='endpoints' else np.column_stack([core,*delta])
            v=ref.copy()
            for f in np.unique(fold):
                ix=np.flatnonzero((fold==f)&(ref==(base>=.5)))
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
                    t,q=ix[a],ix[b]
                    v[q]=calibrated(x[t],x[q],y,t)>=.5
            votes[mode]=v
        anchor=ref==y
        if seed!='v131':refs.append(anchor)
        metrics={}
        for k,v in votes.items():
            h=v==y
            metrics[k]=dict(accuracy=float(h.mean()),fixed=int((h&~anchor).sum()),harmed=int((~h&anchor).sum()))
            if seed!='v131':hist.setdefault(k,[]).append(h)
        hits=(prob>=.5)==y[:,None]
        plants=attribute(hits,dict(population=np.ones(len(y),bool),**masks))
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,base=base,ref=ref,prob=prob,teacher=teacher)
        plans[seed]=dict(models=metrics,training_counts=sizes,effects=plants)
        print(seed,metrics['label_types'],plants['repeat'],flush=True)
    write(REPORT/'factorial.json',dict(plans=plans,grouped={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in hist.items()}))


if __name__=='__main__':
    with threadpool_limits(2):main()
