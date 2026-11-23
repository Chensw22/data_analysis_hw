"""Transfer only the posterior correction contributed by the identity of a service basket."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write

REPORT=ROOT/'reports/nov23_conditional_service_regimes'


def calibrate(anchor,delta,groups,folds,ref,y):
    active=ref==(anchor>=0)
    votes={k:ref.copy() for k in ['calibrated_anchor','calibrated_real','calibrated_sham']}
    for f in np.unique(folds):
        idx=np.flatnonzero(active&(folds==f))
        for a,b in GroupKFold(3,shuffle=True,random_state=642).split(idx,groups=groups[idx]):
            t,q=idx[a],idx[b]
            assert not np.intersect1d(groups[t],groups[q]).size
            for k,n in [('calibrated_anchor',None),('calibrated_real',0),('calibrated_sham',1)]:
                x=anchor[:,None] if n is None else np.c_[anchor,delta[:,n],delta[:,n]*np.tanh(anchor)]
                lr=LogisticRegression(C=.4,max_iter=200).fit(x[t],y[t])
                votes[k][q]=lr.predict(x[q])
    return votes


def main():
    row,y,_,g,_=load()
    wrong,right,hard=cohorts(row,y)
    history,plans={},{}
    for s in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_conditional_service_regimes/{s}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            odds=lambda p:logit(np.clip(p,.002,.998))
            anchor,ref=odds(z['base']),z['ref']
            active=ref==(anchor>=0)
            delta=odds(np.column_stack([z['joint'],z['swapped']]))-odds(z['prior'])[:,None]
            votes=calibrate(anchor,delta,g,z['fold'],ref,y)
            votes['reference']=ref
            for k,d in [('posterior',delta[:,0]),('sham',delta[:,1]),('identity',delta[:,0]-delta[:,1])]:
                votes[k]=np.where(active,anchor+d>=0,ref)
            disagreement=(z['joint']>=.5)!=(z['prior']>=.5)
            true_wins=int((disagreement & ((z['joint']>=.5)==y)).sum())
            sham_wins=int((disagreement & ((z['swapped']>=.5)==y)).sum())
        hits={k:v==y for k,v in votes.items()}
        plans[s]=dict(changes={k:paired(v,ref==y) for k,v in hits.items()},posterior_vs_prior=dict(
            disagreements=int(disagreement.sum()),joint_correct=true_wins,swapped_correct=sham_wins))
        if s!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        print(s,{k:v['fixed']-v['harmed'] for k,v in plans[s]['changes'].items()},flush=True)
    baseline=np.stack(history['reference'])
    stable={}
    for k,values in history.items():
        h=np.stack(values)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-baseline).mean(0),g))
    write(REPORT/'posterior_transfer.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
