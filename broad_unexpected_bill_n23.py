"""Locate the same surprising service with two independent booking samples and remove its evidence."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_local_profile_fit_n23 import cohorts, coordinates
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_service_bag_trees_n23 import fit

REPORT=ROOT/'reports/nov23_unexpected_bill'
OUT=ART/'nov23_unexpected_bill'


def evidence(frame, groups, y, train, query):
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=619).split(train,groups=groups[train]))
    first,second=train[a],train[b]
    assert not np.intersect1d(groups[first],groups[second]).size
    assert not np.intersect1d(groups[train],groups[query]).size
    amounts=frame[BILLS].to_numpy(float)
    medians=np.nanmedian(np.where(amounts[train]>0,amounts[train],np.nan),axis=0)
    state=(amounts>0).astype(int)+(amounts>medians).astype(int)
    known=np.isfinite(amounts)
    p1,p2,masked=[],[],[]
    weights=np.ones(len(train))
    full=fit(frame.iloc[train],y[train],weights).predict_proba(frame.iloc[query])[:,1]
    for j,name in enumerate(BILLS):
        visible=frame.drop(columns=name)
        l=first[known[first,j]]
        r=second[known[second,j]]
        x,_=coordinates(visible,l)
        peer=l[cKDTree(x[l]).query(x[query],k=32)[1]]
        target=state[query,j]
        p1.append((1+(state[peer,j]==target[:,None]).sum(1))/35)
        x,_=coordinates(visible,r)
        model=ExtraTreesClassifier(n_estimators=45,min_samples_leaf=12,max_features=.8,random_state=913,n_jobs=1)
        model.fit(x[r],state[r,j])
        np.testing.assert_array_equal(model.classes_,[0,1,2])
        p2.append(model.predict_proba(x[query])[np.arange(len(query)),target])
        scorer=fit(visible.iloc[train],y[train],weights)
        masked.append(scorer.predict_proba(visible.iloc[query])[:,1])
    return full,np.stack(masked,axis=1),np.stack(p1,axis=1),np.stack(p2,axis=1),state[query],known[query]


def variations(full, masked, p1, p2, state, known, ids):
    i=np.arange(len(full))
    leaders=[np.where(known,p,np.inf).argmin(1) for p in [p1,p2]]
    selected=leaders[0]
    consensus=(selected==leaders[1])&known.any(1)
    equivalent=known & (state==state[i,selected,None]) & (np.arange(len(BILLS))!=selected[:,None])
    comparable=consensus & equivalent.any(1)
    # Stable passenger IDs, independent of row order or labels, break placebo ties.
    rng=np.random.default_rng(544)
    keys=rng.random((int(ids.max())+1,len(BILLS)))[ids]
    placebo=np.where(equivalent,keys,np.inf).argmin(1)
    odds=lambda p:logit(np.clip(p,.002,.998))
    candidate=odds(masked[i,selected])-odds(full)
    sham=odds(masked[i,placebo])-odds(full)
    return {'any_consensus':np.where(consensus,candidate,0.),'matched':np.where(comparable,candidate,0.),
            'sham':np.where(comparable,sham,0.),'difference':np.where(comparable,candidate-sham,0.)},consensus,comparable,selected


def main():
    row,y,frame,g,_=load()
    wrong,right,hard=cohorts(row,y)
    plans,history={},{}
    OUT.mkdir(exist_ok=True)
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,base,ref=z['fold'],z['base'],z['ref']
        n=len(row)
        arrays={k:np.empty((n,5)) for k in ['p1','p2','masked']}
        state=np.empty((n,5),int)
        known=np.empty((n,5),bool)
        full=np.empty(n)
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            full[q],arrays['masked'][q],arrays['p1'][q],arrays['p2'][q],state[q],known[q]=evidence(frame,g,y,t,q)
        shifts,consensus,eligible,selected=variations(full,arrays['masked'],arrays['p1'],arrays['p2'],state,known,row)
        hit=ref==y
        active=ref==(base>=.5)
        anchor=logit(base.clip(.002,.998))
        hits={'reference':hit}
        for k,delta in shifts.items():
            for name,sign in [('drop',1),('emphasize',-1)]:
                hits[name+'_'+k]=np.where(active,anchor+sign*delta>=0,ref)==y
        np.savez_compressed(OUT/f'{seed}.npz',row=row,y=y,ref=ref,fold=fold,base=base,full=full,**arrays,
                            state=state,known=known,consensus=consensus,eligible=eligible,selected=selected)
        metrics={k:paired(h,hit) for k,h in hits.items()}
        plans[seed]=dict(metrics=metrics,both_select_same=int(consensus.sum()),comparable=int(eligible.sum()),
                         hard_in_comparable=int((hard&eligible).sum()),wrong_in_comparable=int((~hit&eligible).sum()))
        if seed!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        print(seed,plans[seed]['comparable'],{k:v['fixed']-v['harmed'] for k,v in metrics.items()},flush=True)
    prev=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       stable_harms=int((right&~h.any(0)).sum()),common_stable=int((wrong&h.all(0)).sum()),
                       gain95_pp=interval((h.astype(float)-prev).mean(0),g))
    write(REPORT/'consensus.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
