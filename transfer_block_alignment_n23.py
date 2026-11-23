"""Test how much class-dependent alignment is already present in the incumbent's odds."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_unexpected_bill_n23 import REPORT


def main():
    row,y,_,g,_=load()
    wrong,right,hard=cohorts(row,y)
    history,plans={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_manifest_block_dependence/{seed}.npz') as z:
            np.testing.assert_array_equal(row,z['row'])
            np.testing.assert_array_equal(y,z['y'])
            base,ref,fold=z['base'],z['ref'],z['fold']
            odds=lambda v:logit(v.clip(.002,.998))
            delta={name:odds(z[name+'_real'])-odds(z[name+'_synthetic']) for name in ['authentic','sham']}
            real=z['authentic_real']>=.5
            synthetic=z['authentic_synthetic']>=.5
        active=ref==(base>=.5)
        anchor=odds(base)
        baseline=ref==y
        hits={'reference':baseline}
        learned={}
        for name,d in delta.items():
            weights=[]
            for direction,v in [('add',1),('subtract',-1)]:
                hits[name+'_'+direction]=np.where(active,anchor+v*d>=0,ref)==y
            pred=ref.copy()
            for f in np.unique(fold):
                ids=np.flatnonzero((fold==f)&active)
                for a,b in GroupKFold(3,shuffle=True,random_state=947).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    def objective(w):
                        eta=anchor[t]+w[0]*d[t]
                        return (np.logaddexp(0,eta)-y[t]*eta).sum()+3*w[0]**2, np.array([d[t]@(expit(eta)-y[t])+6*w[0]])
                    fit=minimize(objective,[0.],jac=True,method='L-BFGS-B',bounds=[(-2,2)])
                    assert fit.success
                    pred[q]=anchor[q]+fit.x[0]*d[q]>=0
                    weights.append(float(fit.x[0]))
            learned[name]=weights
            hits[name+'_learned']=pred==y
        metrics={k:paired(h,baseline) for k,h in hits.items()}
        plans[seed]=dict(metrics=metrics,authentic_over_synthetic=paired(real==y,synthetic==y),
                         coefficients=learned)
        if seed!='v131':
            for k,h in hits.items():
                history.setdefault(k,[]).append(h)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in metrics.items()},flush=True)
    prev=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-prev).mean(0),g))
    write(REPORT/'block_transfer.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
