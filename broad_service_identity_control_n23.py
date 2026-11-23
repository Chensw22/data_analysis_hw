"""Compare identified services to anonymous bill ranks with identical observed amounts."""
import numpy as np
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_fixed_vote_totals_n23 import REPORT
from broad_geographic_transfer_n23 import BILLS,load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_service_bag_n23 import design,fit


def main():
    row,y,frame,g,_=load()
    wrong,right,hard=cohorts(row,y)
    anonymous=frame.copy()
    raw=frame[BILLS].to_numpy(float)
    # Missing values go to the end; all present values are retained exactly.
    anonymous[BILLS]=np.sort(raw,axis=1)
    np.testing.assert_array_equal(np.sort(raw,axis=1),anonymous[BILLS].to_numpy())
    x,gate,prior=design(anonymous)
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_service_bag/{seed}.npz') as data:
            np.testing.assert_array_equal(row,data['row'])
            np.testing.assert_array_equal(y,data['y'])
            fold,ref,base,real=data['fold'],data['ref'],data['base'],data['fixed']
        rank=np.full(len(y),np.nan)
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            rank[q]=fit(x,gate,prior,y,t,q,'fixed')
        assert np.isfinite(rank).all()
        active=ref==(base>=.5)
        delta=logit(real.clip(.002,.998))-logit(rank.clip(.002,.998))
        shift=expit(logit(base.clip(.002,.998))+.25*delta)
        hits={'reference':ref==y,'anonymous':(rank>=.5)==y,'named':(real>=.5)==y,
              'quarter_anonymous':np.where(active,.75*base+.25*rank>=.5,ref)==y,
              'quarter_named':np.where(active,.75*base+.25*real>=.5,ref)==y,
              'named_increment':np.where(active,shift>=.5,ref)==y}
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
    write(REPORT/'service_identity.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
