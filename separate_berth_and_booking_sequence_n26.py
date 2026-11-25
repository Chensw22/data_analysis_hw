"""Separate cabin number, anonymous booking order, and local departures from the track sequence."""
import json
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from catboost import CatBoostRanker,Pool
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from conflict_field_blocks_n26 import designs
from conflict_pair_ordering_n26 import CACHE,REPORT as RANK_REPORT,training_pairs
from cross_order_conflict_cover_n26 import relations
from exhaustive_family_quorums_n26 import inputs,summary
from full_roster_booking_validation_n23 import OUT,ROOT,data
from kaggle_workflow import sha256


def geometry(roster):
    booking = roster.PassengerId.str.split('_').str[0].astype(int).to_numpy()
    pieces = roster.Cabin.str.split('/',expand=True)
    track = (pieces[0]+pieces[2]).fillna('?').to_numpy(str)
    berth = pd.to_numeric(pieces[1]).to_numpy(float)
    center = np.full(len(roster),np.nan)
    gap = np.full(len(roster),np.nan)
    exact = np.zeros(len(roster),bool)
    locations = 0
    for key in np.unique(track[track!='?']):
        ix = np.flatnonzero(track==key)
        mapping = np.unique(np.column_stack([booking[ix],berth[ix].astype(int)]),axis=0)
        orders,numbers = mapping.T
        assert (np.diff(orders)>0).all() and (np.diff(numbers)>0).all()
        locations += len(mapping)
        pos = np.searchsorted(orders,booking[ix])
        keep = (pos>0)&(pos+1<len(orders))
        ask = ix[keep]
        j = pos[keep]
        lam = (booking[ask]-orders[j-1])/(orders[j+1]-orders[j-1])
        center[ask] = (1-lam)*numbers[j-1]+lam*numbers[j+1]
        gap[ask] = numbers[j+1]-numbers[j-1]-1
        exact[ask] = gap[ask]==1
    summary = dict(identified_passengers=int(np.isfinite(berth).sum()),distinct_booking_tracks=locations,
              passengers_between_two_other_orders=int(np.isfinite(center).sum()),
              exact_berth_forced_by_neighbors=int(exact.sum()),
              median_absolute_interpolation_error=float(np.nanmedian(abs(center-berth))))
    return booking,track,berth,center,gap,exact,summary


def run():
    raw,y,g,f = data()
    _,cur,_ = inputs()
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    joined = pd.concat([raw,test],ignore_index=True)
    chronology,track,berth,center,gap,exact,geo = geometry(joined)
    n = len(y)
    chrono = chronology[:n]
    track,berth,center,gap,exact = [z[:n] for z in (track,berth,center,gap,exact)]
    original = designs(raw,g)[3]
    observed = np.isfinite(berth)
    np.testing.assert_array_equal(original[:,-1],np.nan_to_num(berth,nan=-1))
    missing_value = np.nan_to_num(center,nan=-1)
    residual = np.nan_to_num(berth-center,nan=0)
    base = original[:,:-1]
    matrices = dict(track=base,actual=original,sequence=np.column_stack([base,chrono]),
                    interpolated=np.column_stack([base,missing_value]),actual_and_sequence=np.column_stack([original,chrono]),
                    local_geometry=np.column_stack([original,chrono,residual,np.nan_to_num(gap,nan=-1),exact]))
    cache = CACHE/'held.npz'
    assert sha256(cache)==json.loads(RANK_REPORT.read_text())['cached_predictions_sha256']
    with np.load(cache) as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.to_numpy(str))
        pairs = {s:z[s+'_pairs'] for s in ('spending','registration')}
    a,b,d = relations(raw,g,False)
    pred = np.full((n,len(matrices)),np.nan)
    diff = {s:np.full((len(p),len(matrices)),np.nan) for s,p in pairs.items()}
    for k,h in combinations_with_replacement(np.unique(f),2):
        t = np.flatnonzero((f!=k)&(f!=h))
        selection = {s:(np.sort(f[p],axis=1)==[k,h]).all(1) for s,p in pairs.items()}
        q = np.flatnonzero(f==k) if k==h else np.unique(np.concatenate([p[selection[s]].ravel() for s,p in pairs.items()]))
        if not len(q):
            continue
        assert not np.isin(g[t],g[q]).any()
        real,_ = training_pairs(a,b,d,y,g,(f!=k)&(f!=h))
        for j,(method,x) in enumerate(matrices.items()):
            model = CatBoostRanker(loss_function='PairLogit',iterations=115,depth=4,learning_rate=.07,
                l2_leaf_reg=8,thread_count=2,verbose=False,allow_writing_files=False,random_seed=3505)
            score = model.fit(Pool(x[t],label=y[t].astype(int),pairs=real)).predict(x[q])
            if k==h:
                pred[q,j] = score
            for s,p in pairs.items():
                m = selection[s]
                u,v = np.searchsorted(q,p[m]).T
                diff[s][m,j] = score[u]-score[v]
    assert np.isfinite(pred).all() and all(np.isfinite(v).all() for v in diff.values())
    report = dict(roster_geometry=geo,ordering={},deployment={})
    for s,p in pairs.items():
        u,v = p.T
        same = observed[u]&observed[v]&(track[u]==track[v])
        np.testing.assert_array_equal(berth[u[same]]>berth[v[same]],chrono[u[same]]>chrono[v[same]])
        report['ordering'][s] = dict(same_observed_track_pairs=int(same.sum()),correct_equivalents={name:float(
            ((diff[s][:,j]>0)+.5*(diff[s][:,j]==0)).sum()) for j,name in enumerate(matrices)})
        old = json.loads((RANK_REPORT.parent/'conflict_field_blocks.json').read_text())
        assert report['ordering'][s]['correct_equivalents']['actual']==old['ordering'][s]['correct_equivalents_per_mask'][3]
    with np.load(OUT/'mitra96.npz') as z:
        np.testing.assert_array_equal(z['fold'],f)
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.to_numpy(str))
        strong = logit(z['p'].clip(.003,.997))
    votes = np.zeros_like(pred,dtype=bool)
    for k in np.unique(f):
        rows = np.flatnonzero(f==k)
        for u,v in GroupKFold(4,shuffle=True,random_state=169).split(rows,groups=g[rows]):
            t,q = rows[u],rows[v]
            assert not np.isin(g[t],g[q]).any()
            for j in range(pred.shape[1]):
                x = np.column_stack([strong,pred[:,j]])
                votes[q,j] = LogisticRegression(C=.5,max_iter=350).fit(x[t],y[t]).predict(x[q])
    actual = votes[:,list(matrices).index('actual')]
    for s,mask in [('all',cur['known']),('common',cur['common']),('new',cur['known']&cur['new'])]:
        report['deployment'][s] = {}
        for j,method in enumerate(matrices):
            new = votes[:,j]
            extra = (new!=cur['parent'])&(actual==cur['parent'])
            report['deployment'][s][method] = dict(versus_actual=summary(new!=actual,actual!=y,mask,f),
                        extra_challenges=summary(extra,cur['wrong'],mask,f))
    report['scope'] = ('Anonymous booking order is an ordinal predictor. Interpolation excludes the entire target '
       'booking from both bounding points and uses no outcome labels. A forced integer berth requires both '
       'bounding berths with just one admissible integer between them on the same observed deck and side.')
    (ROOT/'reports/nov23_matched_learning_curve/separate_berth_and_booking_sequence.json').write_text(json.dumps(report,indent=2)+'\n')
    print(geo,report['ordering'],flush=True)
    print({k:v['extra_challenges']['net'] for k,v in report['deployment']['common'].items()},flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
