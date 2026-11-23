"""Estimate a passenger's local joint bill tails, conditional on registration attributes."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_local_profile_fit_n23 import cohorts, coordinates
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write

OUT = ART/'nov23_local_spend_depth'
REPORT = ROOT/'reports/nov23_local_spend_depth'
ARMS = ('raw', 'axis', 'independent', 'joint')


def directions():
    # Positive and negative contrasts capture coordinated and offsetting service use.
    eye = np.eye(len(BILLS), dtype=np.float32)
    pairs = [eye[i] + sign*eye[j] for i,j in combinations(range(len(BILLS)),2) for sign in (-1,1)]
    return np.array([*pairs, np.ones(5), [1,-1,1,1,-1]], dtype=np.float32).T


def rank_features(donors, query):
    ties = (donors == query[:,None]).mean(axis=1)
    midrank = (donors < query[:,None]).mean(axis=1) + .5*ties
    magnitude = np.abs(2*midrank-1)
    return np.c_[midrank, ties, magnitude.max(1), magnitude.mean(1), (magnitude>.9).mean(1)]


def feature_views(frame, groups, train, rows, k=80):
    context,_ = coordinates(frame.drop(columns=BILLS), train)
    bills = np.log1p(frame[BILLS].to_numpy(dtype=np.float32))
    bills = np.where(np.isfinite(bills), bills, np.nanmedian(bills[train],axis=0))
    projections = directions()
    projected = bills @ projections
    rng = np.random.default_rng(794)
    permutations = [rng.permutation(k) for _ in BILLS]
    values = {name:[] for name in ARMS[1:]}
    for start in range(0,len(rows),96):
        q = rows[start:start+96]
        distance = cdist(context[q],context[train],metric='sqeuclidean')
        distance[groups[q,None]==groups[train][None,:]] = np.inf
        nearest = np.argpartition(distance,k-1,axis=1)[:,:k]
        ids = train[nearest]
        assert np.all(groups[ids]!=groups[q,None])
        donor = bills[ids]
        axis = rank_features(donor,bills[q])
        # Preserve every service's conditional empirical distribution exactly in the sham.
        sham = np.stack([donor[:,order,j] for j,order in enumerate(permutations)],axis=2)
        independent = rank_features(sham@projections,projected[q])
        joint = rank_features(projected[ids],projected[q])
        values['axis'].append(axis)
        values['independent'].append(np.c_[axis,independent])
        values['joint'].append(np.c_[axis,joint])
    raw = frame.copy()
    for col in raw.select_dtypes(exclude=np.number):
        raw[col] = pd.Categorical(raw[col],categories=raw[col].iloc[train].unique())
    raw = pd.get_dummies(raw).to_numpy(dtype=np.float32)[rows]
    return dict(raw=raw,**{k:np.c_[raw,np.concatenate(v)] for k,v in values.items()})


def main():
    row,y,frame,g,_ = load()
    _,right,hard = cohorts(row,y)
    solo = frame.BookingSize.to_numpy()==1
    OUT.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    panel, summary = {}, {'folds':{},'across':{}}
    for seed in [*SEEDS,'v131']:
        fold,ref,op,_ = read_opinions(seed,row,y)
        scores = {key:np.full(len(y),np.nan) for key in ARMS}
        for f in np.unique(fold):
            q = np.flatnonzero(fold==f)
            t = np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            views = feature_views(frame,g,t,np.r_[t,q])
            for key,x in views.items():
                model = HistGradientBoostingClassifier(max_iter=175,learning_rate=.065,max_leaf_nodes=15,
                    min_samples_leaf=26,l2_regularization=4,early_stopping=False,random_state=873)
                model.fit(x[:len(t)],y[t])
                scores[key][q] = model.predict_proba(x[len(t):])[:,1]
        assert all(np.isfinite(p).all() for p in scores.values())
        np.savez_compressed(OUT/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,**scores)
        hits = {'reference':ref==y}
        for key,p in scores.items():
            hits[key] = (p>=.5)==y
            vote = np.where(ref==(op[:,0]>=.5),.75*op[:,0]+.25*p>=.5,ref)
            hits[key+'_quarter'] = vote==y
        summary['folds'][str(seed)] = {}
        for key,h in hits.items():
            result = paired(h,ref==y)
            result.update(hard=int(h[hard].sum()),solo_hard=int(h[hard&solo].sum()),
                          right_missed=int((~h&right).sum()))
            summary['folds'][str(seed)][key] = result
        print(seed, {key:(round(100*h.mean(),3),int(h[hard].sum())) for key,h in hits.items()},flush=True)
        panel[str(seed)] = hits
    group = {key:np.stack([panel[str(s)][key] for s in SEEDS]) for key in hits}
    for key,v in group.items():
        ref = group['reference']
        comp = group['independent_quarter' if key.endswith('_quarter') else 'independent']
        summary['across'][key] = dict(
            hard_all=int((v.all(0)&hard).sum()),hard_any=int((v.any(0)&hard).sum()),
            solo_hard_all=int((v.all(0)&hard&solo).sum()),solo_hard_any=int((v.any(0)&hard&solo).sum()),
            right_wrong_all=int((~v.any(0)&right).sum()),
            ci_vs_reference=interval((v.astype(float)-ref).mean(0),g),
            ci_vs_independent=interval((v.astype(float)-comp).mean(0),g))
    summary['sizes'] = dict(rows=len(y),hard=int(hard.sum()),solo_hard=int((hard&solo).sum()),right=int(right.sum()))
    write(REPORT/'results.json',summary)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
