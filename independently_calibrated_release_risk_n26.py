"""Test whether high estimated release-error risk transfers across independent booking batches."""
import json

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import fit
from frozen_queue_shared_errors_n26 import paired
from target_mix_expert_routing_n26 import ROOT, batches, ticker


def label_bounds(train, y, q):
    # Venn-Abers calibration: assign the new passenger each of the two possible labels,
    # separately fit the monotone calibration curve, then read its value at that passenger.
    unique, counts = np.unique(train,return_counts=True)
    pos = np.bincount(np.searchsorted(unique, train),weights=y,minlength=len(unique))
    values = np.unique(q)
    endpoints = np.empty((len(values),2))
    for i, val in enumerate(values):
        k = np.searchsorted(unique,val)
        if k < len(unique) and unique[k]==val:
            x, n, a = unique, counts.copy(), pos.copy()
            n[k] += 1
        else:
            x,n,a = np.insert(unique,k,val),np.insert(counts,k,1),np.insert(pos,k,0.)
        for label in (0,1):
            target = a.copy()
            target[k] += label
            endpoints[i,label] = IsotonicRegression().fit_transform(x,target/n,sample_weight=n)[k]
    assert (endpoints[:,0] <= endpoints[:,1]+1.e-10).all()
    return endpoints[np.searchsorted(values,q)]


def run():
    raw,y,g,f,known,versions = paired()
    parents = versions[:,-1]
    all_agree = (versions==parents[:,None]).all(1)
    parts = []
    for b in batches(True):
        if b.tag!='full':
            continue
        a,q = b.train_ids,b.ids
        assert not np.isin(g[a],g[q]).any()
        assert np.array_equal(b.query_error, parents[q]!=y[q])
        reps = []
        for tr, cal in GroupKFold(2,shuffle=True,random_state=2631).split(a,groups=g[a]):
            assert not np.isin(g[a[tr]],g[a[cal]]).any()
            matrix = np.vstack([b.train_fields[cal],b.query_fields])
            r = fit(b.train_fields[tr],b.train_error[tr],matrix,np.ones(len(tr)))
            n = len(cal)
            query = r[n:]
            bounds = label_bounds(r[:n],b.train_error[cal],query)
            sham = b.train_error[cal].copy()
            rng = np.random.default_rng(513)
            for direction in (0,1):
                subset = np.flatnonzero(b.train_cell[cal]%2==direction)
                sham[subset] = rng.permutation(sham[subset])
            shuffled = IsotonicRegression(out_of_bounds='clip').fit(r[:n],sham).predict(query)
            # Convert the two probabilities into a single regularized prediction.
            reg = bounds[:,1]/(1-bounds[:,0]+bounds[:,1])
            reps.append((query,bounds[:,0],reg,shuffled))
        values = np.array(reps)
        score = values[:,2].mean(0)
        rawrisk = values[:,0].mean(0)
        actions = dict(uncalibrated=rawrisk>.5, calibrated=score>.5,
                       both_above_half=(values[:,2]>.5).all(0),
                       both_lower_above_half=(values[:,1]>.5).all(0),
                       only_after_calibration=(score>.5)&(rawrisk<=.5),
                       shuffled=values[:,3].mean(0)>.5)
        # Ranking does not inspect query labels, and equal scores are handled together.
        cuts = np.quantile(score,[.5,.8,.9,.95])
        higher = [score>=cut for cut in cuts]
        parts.append((q,actions,higher,score))
    items=parts
    ids = np.concatenate([r[0] for r in items])
    np.testing.assert_array_equal(np.sort(ids),np.flatnonzero(known))
    wrong = parents[ids]!=y[ids]
    common = all_agree[ids]
    report = {}
    for n in items[0][1]:
        action = np.concatenate([r[1][n] for r in items])
        report[n] = dict(all=ticker(np.ones(len(ids),bool),wrong,action),
                         unanimous=ticker(common,wrong,action),
                         fold_net=[int(np.sum(np.where(wrong,1,-1)*action*(f[ids]==k))) for k in np.unique(f)])
    for j, frac in enumerate((.5,.2,.1,.05)):
        m=np.concatenate([r[2][j] for r in items])
        report[f'top_{frac}'] = dict(rows=int(m.sum()),errors=int(wrong[m].sum()),
             common_rows=int((m & common).sum()),common_errors=int((m & common & wrong).sum()),
             common_errors_per_fold=[int((m & common & wrong & (f[ids]==k)).sum()) for k in np.unique(f)])
    path = ROOT/'artifacts/nov26_independently_calibrated_release_risk'
    path.mkdir(exist_ok=True)
    np.savez_compressed(path/'held_risk.npz', ids=ids, passenger=raw.PassengerId.iloc[ids].to_numpy(str),
                        score=np.concatenate([r[3] for r in items]), top20=np.concatenate([r[2][1] for r in items]))
    print(report,flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/independently_calibrated_release_risk.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
