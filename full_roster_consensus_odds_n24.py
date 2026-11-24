"""Test conditional risk ranking among unanimous votes at matched initial risk."""
import json

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference


def cells(rows, code):
    return [rows[code[rows] == k] for k in np.unique(code[rows])]


def comparisons(rows, code, wrong, x, group):
    diffs = []
    for ids in cells(rows,code):
        pos, neg = ids[wrong[ids]], ids[~wrong[ids]]
        if len(pos) and len(neg):
            cross = group[pos,None] != group[neg]
            diffs.append((x[pos,None,:] - x[neg])[cross])
    return np.concatenate(diffs) if diffs else np.empty((0,x.shape[1]))


def allocate(ids, cells_in_fold, risk, signal, slope):
    output = risk.copy()
    for rows in cells_in_fold:
        z = logit(risk[rows].clip(1e-8,1-1e-8)) + slope*signal[rows]
        expected = risk[rows].sum()
        offset = brentq(lambda a: expit(z+a).sum()-expected, -60, 60)
        output[rows] = expit(z+offset)
        assert abs(output[rows].sum()-expected) < 1e-7
    return output[ids]


def run():
    raw,y,group,fold = data()
    names,h,p = banks(raw,y,fold)
    ref = latest_reference(y,group,fold,names,h,p)
    call = p[:,4] >= .5
    certain = ((p >= .5) == call[:,None]).all(1)
    np.testing.assert_array_equal(call[certain],ref[certain])
    wrong = y != call
    risk = np.where(call[:,None],1-p,p)
    own, peer = risk[:,4], risk[:,:5].mean(1)
    x = np.column_stack([own,peer,risk.mean(1),risk.max(1)])
    signal = logit(peer.clip(.001,.999))-logit(own.clip(.001,.999))
    code = 100*call.astype(int) + np.floor(own*100).astype(int)
    oof = np.full(len(y),np.nan)
    fold_stats, diffs = [], []
    for f in np.unique(fold):
        outer = np.flatnonzero((fold==f)&certain)
        partitions = cells(outer,code)
        d = comparisons(outer,code,wrong,x,group)
        diffs.append(d)
        slopes = []
        for a,b in GroupKFold(4,shuffle=True,random_state=462).split(outer,groups=group[outer]):
            t,q = outer[a],outer[b]
            assert not np.isin(group[t],group[q]).any()
            dx = comparisons(t,code,wrong,signal[:,None],group).ravel()
            def loss(w):
                return np.logaddexp(0,-w*dx).sum() + 10*w*w
            w = minimize_scalar(loss, bounds=(-5,5),method='bounded').x
            oof[q] = allocate(q, partitions, own, signal, w)
            slopes.append(float(w))
        fold_stats.append(dict(pairs=len(d),auc=((d>0)+.5*(d==0)).mean(0).tolist(), slopes=slopes))
    assert np.isfinite(oof[certain]).all()
    pooled = np.concatenate(diffs)
    bands = np.digitize(own,[.025,.05,.1,.2,.3])
    report = dict(study_rows=int(certain.sum()),actual_errors=int(wrong[certain].sum()),
                  expected_errors=float(own[certain].sum()),bands=[],folds=fold_stats,
                  order_matched_pairs=len(pooled),auc=((pooled>0)+.5*(pooled==0)).mean(0).tolist(),
                  ordering=['mitra','five_mean','all_mean','all_max'])
    for i in range(6):
        use = certain & (bands==i)
        report['bands'].append(dict(rows=int(use.sum()),errors=int(wrong[use].sum()),expected=float(own[use].sum())))
    report['directions'] = {str(j):dict(rows=int((certain&(call==j)).sum()),
        errors=int(wrong[certain&(call==j)].sum()),expected=float(own[certain&(call==j)].sum())) for j in [False,True]}
    for key,s in [('mitra',own),('redistributed',oof)]:
        flip = certain & (s>.5)
        eligible = np.flatnonzero(certain)
        top = eligible[np.argsort(-s[eligible],kind='stable')[:100]]
        report[key]=dict(fixed=int((flip&wrong).sum()),harmed=int((flip&~wrong).sum()),
                         highest=float(s[certain].max()),errors_in_top_100=int(wrong[top].sum()),
                         brier=float(np.mean((s[certain]-wrong[certain])**2)))
    path = ROOT/'reports/nov23_matched_learning_curve/consensus_odds.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
