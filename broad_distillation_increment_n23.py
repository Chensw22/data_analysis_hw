"""Distinguish a student's novel correct decisions from copying a teacher or averaging parents."""
import numpy as np
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,SEEDS,historical,write
from broad_prediction_assisted_training_n23 import REPORT
from broad_release_miss_coverage_n23 import summarize
from released_correction_audit_n23 import development


def transition(student, native, teacher, y):
    result={}
    for name,mask in [('agree',native==teacher),('disagree',native!=teacher)]:
        result[name]=dict(rows=int(mask.sum()),fixed=int((mask&(native!=y)&(student==y)).sum()),
                          harmed=int((mask&(native==y)&(student!=y)).sum()))
    return result


def main():
    row,y,frame,g,_=load()
    rows,target,masks=historical()
    np.testing.assert_array_equal(rows,row)
    np.testing.assert_array_equal(target,y)
    _,release,_=development()
    state=strata(frame)
    masks={**masks,'paid_transported':masks['repeat']&(state==2)&y,'zero_stayed':masks['repeat']&(state==0)&~y}
    history,refs,plans,agreements={},[],{},[]
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_crossfamily_unlabelled_distillation/{seed}.npz') as z:
            for k,v in [('row',row),('y',y)]:np.testing.assert_array_equal(z[k],v)
            fold,base,ref=z['fold'],z['base'],z['ref']
            native,teacher,target,sham=[z[k] for k in ['native','teacher','target_mix','shuffled']]
        np.testing.assert_array_equal(ref,release[131] if seed=='v131' else base>=.5)
        mimic=native.copy()
        weights=[]
        diff=teacher-native
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            weight=float(np.clip(diff[q]@(target[q]-native[q])/max(diff[q]@diff[q],1e-10),0,1))
            mimic[q]+=weight*diff[q]
            weights.append(weight)
        def odds(p):return logit(p.clip(.002,.998))
        votes={'student':target>=.5,'mimic':mimic>=.5}
        designs={'parents':np.c_[odds(base),odds(native),odds(teacher)]}
        for k,p in [('student',target),('shuffled',sham)]:
            designs[k]=np.c_[designs['parents'],odds(p)]
        for k,x in designs.items():
            v=ref.copy()
            for f in np.unique(fold):
                ix=np.flatnonzero((fold==f)&(ref==(base>=.5)))
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
                    t,q=ix[a],ix[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    v[q]=calibrated(x[t],x[q],y,t)>=.5
            votes['combined_'+k]=v
        anchor=ref==y
        if seed!='v131':
            refs.append(anchor)
            agreements.append(((teacher>=.5)!=y)&((native>=.5)!=y))
        stats={}
        for k,v in votes.items():
            hit=v==y
            stats[k]=dict(accuracy=float(hit.mean()),fixed=int((hit&~anchor).sum()),harmed=int((~hit&anchor).sum()))
            if seed!='v131':history.setdefault(k,[]).append(hit)
        plans[seed]=dict(teacher_weights=weights,models=stats,
                         transitions=transition(votes['student'],native>=.5,teacher>=.5,y))
    both_wrong=np.logical_and.reduce(agreements)
    masks={**masks,'both_parents_always_wrong':both_wrong,
           'both_parents_and_base_always_wrong':both_wrong&masks['repeat']}
    write(REPORT/'increment.json',dict(plans=plans,counts={k:int(m.sum()) for k,m in masks.items()},
          grouped={k:summarize(np.stack(h),np.stack(refs),masks) for k,h in history.items()}))
    print(plans,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
