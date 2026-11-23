"""Evaluate hypothetical paid baskets as predictors for passengers with no positive bill."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS,load
from broad_local_profile_fit_n23 import cohorts,coordinates
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_service_bag_trees_n23 import fit
from broad_service_pairs_n23 import REPORT


def simulate(frame,g,y,t,q,paid):
    support=t[paid[t]]
    fit_paid=fit(frame.iloc[support],y[support],np.ones(len(support)))
    personal=frame.drop(columns=BILLS)
    x,_=coordinates(personal,support)
    # The complete query bookings have been excluded from the paid support.
    assert not np.intersect1d(g[q],g[support]).size
    rng=np.random.default_rng(729)
    support=rng.permutation(support)
    near=support[cKDTree(x[support]).query(x[q],k=24)[1]]
    random=rng.choice(support,size=near.shape,replace=True)
    out={'zero_extrapolation':fit_paid.predict_proba(frame.iloc[q])[:,1]}
    for k,donors in [('matched',near),('random',random)]:
        person=np.repeat(q,donors.shape[1])
        new=frame.iloc[person].copy()
        new[BILLS]=frame.iloc[donors.ravel()][BILLS].to_numpy()
        predictions=fit_paid.predict_proba(new)[:,1].reshape(donors.shape)
        out[k]=predictions.mean(1)
        out[k+'_dispersion']=predictions.std(1)
        out[k+'_outcomes']=y[donors].mean(1)
    return out


def main():
    row,y,frame,g,_=load()
    wrong,right,hard=cohorts(row,y)
    paid=frame[BILLS].gt(0).any(axis=1).to_numpy()
    plans,history={},{}
    directory=ART/'nov23_zero_spend_analogues'
    directory.mkdir(exist_ok=True)
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,base,ref=z['fold'],z['base'],z['ref']
        votes={'reference':ref}
        raw={}
        for f in np.unique(fold):
            complete=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[complete]))
            q=complete[(~paid[complete])&(ref[complete]==(base[complete]>=.5))]
            result=simulate(frame,g,y,t,q,paid)
            for name,p in result.items():
                raw.setdefault(name,np.full(len(y),np.nan))[q]=p
            votes.setdefault('always_negative',ref.copy())[q]=False
            off=logit(base[q].clip(.002,.998))
            for key in ('zero_extrapolation','matched','random','matched_outcomes'):
                p=result[key]
                votes.setdefault(key,ref.copy())[q]=p>=.5
                votes.setdefault('quarter_'+key,ref.copy())[q]=.75*base[q]+.25*p>=.5
            for a,b in GroupKFold(3,shuffle=True,random_state=106).split(q,groups=g[q]):
                assert not np.intersect1d(g[q[a]],g[q[b]]).size
                for key in ('base','matched','random'):
                    cols=[off]
                    if key!='base':
                        cols.extend([logit(result[key].clip(.002,.998)),result[key+'_dispersion'],
                                     result[key+'_outcomes'],logit(result['zero_extrapolation'].clip(.002,.998))])
                    x=np.column_stack(cols)
                    x=(x-x[a].mean(0))/x[a].std(0).clip(.08)
                    lr=LogisticRegression(C=.3,max_iter=320).fit(x[a],y[q[a]])
                    votes.setdefault('calibrated_'+key,ref.copy())[q[b]]=lr.predict(x[b])
        np.savez_compressed(directory/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=base,
                            **raw,**{'vote_'+k:v for k,v in votes.items()})
        hits={k:v==y for k,v in votes.items()}
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
    write(REPORT/'zero_analogues.json',dict(plans=plans,stable=stable,zero_passengers=int((~paid).sum()),
                                             hard_zero=int((hard&~paid).sum())))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
