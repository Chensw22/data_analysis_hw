"""Recover a masked posterior from outcome-conditioned bill completions and an independent outcome model."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit, logit, softmax
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from broad_bill_completion_states_n23 import REPORT
from broad_geographic_transfer_n23 import BILLS
from broad_local_profile_fit_n23 import coordinates
from broad_opposing_passengers_n23 import evaluate
from broad_service_bag_trees_n23 import fit


def posterior(model, frame, groups, y, donor, query, fields, rng):
    donor = donor[frame.iloc[donor][fields].notna().all(axis=1).to_numpy()]
    assert not np.intersect1d(groups[donor], groups[query]).size
    x, _ = coordinates(frame.drop(columns=fields), donor)
    labels = y[donor]
    shuffled = rng.permutation(labels)
    pools = [np.ones(len(donor),bool), ~labels, labels, ~shuffled, shuffled]
    pieces = [cKDTree(x[donor[v]]).query(x[query], k=16) for v in pools]
    bandwidth = np.maximum(pieces[0][0][:,-1], .12)
    estimates = []
    for mask,(dist,idx) in zip(pools,pieces):
        peers = donor[mask][idx]
        original = frame.iloc[np.repeat(query, peers.shape[1])].copy()
        original[fields] = frame.iloc[peers.ravel()][fields].to_numpy()
        pred = model.predict_proba(original, thread_count=2)[:,1].reshape(peers.shape)
        estimates.append((softmax(-dist**2 / bandwidth[:,None]**2,axis=1)*pred).sum(1))
    pooled, neg, pos, sham_neg, sham_pos = estimates
    # If the generator and full outcome model agree, m = (1-m)*neg + m*pos by total expectation.
    def stationary(n,p):
        return n / (1-p+n)
    return pooled, stationary(neg,pos), stationary(sham_neg,sham_pos)


def run_fold(frame, groups, y, train, query):
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=493).split(train,groups=groups[train]))
    t,d=train[a],train[b]
    assert not np.intersect1d(groups[t],groups[d]).size
    assert not np.intersect1d(groups[train],groups[query]).size
    model=fit(frame.iloc[t],y[t],np.ones(len(t)))
    fact=model.predict_proba(frame.iloc[query],thread_count=2)[:,1]
    rng=np.random.default_rng(688)
    pop,post,sham=posterior(model,frame,groups,y,d,query,BILLS,rng)
    odds=lambda p:logit(np.clip(p,.002,.998))
    results=dict(factual=fact, pooled_without_bills=pop, stationary_without_bills=post,
                 shuffled_without_bills=sham, factual_with_shift=expit(odds(fact)+odds(post)-odds(pop)),
                 shuffled_shift=expit(odds(fact)+odds(sham)-odds(pop)))
    missing=frame.iloc[query][BILLS].isna().to_numpy()
    completed=np.column_stack([fact,fact,fact])
    patterns=np.unique(missing[missing.any(1)],axis=0)
    for pattern in patterns:
        inds=np.flatnonzero((missing==pattern).all(1))
        fields=[n for n,absent in zip(BILLS,pattern) if absent]
        completed[inds]=np.column_stack(posterior(model,frame,groups,y,d,query[inds],fields,rng))
    for name,column in zip(['pooled_completion','stationary_completion','shuffled_completion'],completed.T):
        results[name]=column
    return results,dict(model_rows=len(t), donor_rows=len(d), query_rows=len(query), patterns=len(patterns))


if __name__=='__main__':
    REPORT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(2):
        evaluate('conditional_bill_completion',run_fold,REPORT)
