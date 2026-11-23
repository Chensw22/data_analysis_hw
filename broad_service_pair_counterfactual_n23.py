"""Hold the fitted pair model fixed and replace only the second service distribution."""
import numpy as np
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_service_bag_trees_n23 import expand
from broad_service_pairs_n23 import REPORT,fit_pair,pair_table,partner_bank


def counterfactual(frame,g,y,t,q):
    owner,rows,prior=expand(frame)
    train,xt,wt=pair_table(owner,rows,prior,t,t)
    fit=fit_pair(xt,y[train],wt)
    def score(donor):
        people,x,w=pair_table(owner,rows,prior,q,donor)
        p=fit.predict_proba(x)[:,1]
        return np.bincount(people,weights=w*p,minlength=len(frame))[q]
    original=score(q)
    donors=partner_bank(frame,g,t,q,6)
    alternatives=np.column_stack([score(ids) for ids in donors.T])
    return original,alternatives.mean(1),alternatives.std(1)


def main():
    row,y,frame,g,_=load()
    wrong,right,hard=cohorts(row,y)
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_service_pairs/{seed}.npz') as z:
            np.testing.assert_array_equal(row,z['row'])
            np.testing.assert_array_equal(y,z['y'])
            fold,base,ref,previous=z['fold'],z['base'],z['ref'],z['same_passenger_both']
        active=ref==(base>=.5)
        out={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            real,other,spread=counterfactual(frame,g,y,t,q)
            np.testing.assert_allclose(real,previous[q],atol=1e-12,rtol=0)
            a=logit(base[q].clip(.002,.998))
            placebo=logit(other.clip(.002,.998))
            delta=logit(real.clip(.002,.998))-placebo
            for name,p in [('quarter_actual',.75*base[q]+.25*real),('quarter_other',.75*base[q]+.25*other),
                           ('quarter_difference',expit(a+.25*delta)),('full_difference',expit(a+delta))]:
                out.setdefault(name,ref.copy())[q]=np.where(active[q],p>=.5,ref[q])
            for aidx,b in GroupKFold(3,shuffle=True,random_state=617).split(q,groups=g[q]):
                assert not np.intersect1d(g[q[aidx]],g[q[b]]).size
                aidx=aidx[active[q[aidx]]]
                for name,x in [('calibrated_other',np.c_[a,placebo]),('calibrated_joint',np.c_[a,placebo,delta,spread])]:
                    x=(x-x[aidx].mean(0))/x[aidx].std(0).clip(.08)
                    model=LogisticRegression(C=.3,max_iter=400).fit(x[aidx],y[q[aidx]])
                    out.setdefault(name,ref.copy())[q[b]]=np.where(active[q[b]],model.predict(x[b]),ref[q[b]])
        hits={k:v==y for k,v in {**out,'reference':ref}.items()}
        plans[seed]={k:paired(v,ref==y) for k,v in hits.items()}
        if seed!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in plans[seed].items()},flush=True)
    baseline=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-baseline).mean(0),g))
    write(REPORT/'pair_replacement.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
