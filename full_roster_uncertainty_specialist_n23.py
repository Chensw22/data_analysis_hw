"""Train on uncertain cases identified without their labels; compare with size-matched controls."""
import json

import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold

from broad_booking_policy_replay_n23 import interval
from exaone_validation_data import CATEGORICAL, compact_features
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_unexpected_peers_n23 import reference_score


def estimate(frame, y, t, q, weight=None):
    model = CatBoostClassifier(iterations=300, depth=5, learning_rate=.065, thread_count=3,
                               verbose=False, random_seed=248, allow_writing_files=False)
    model.fit(frame.iloc[t], y[t].astype(int), cat_features=CATEGORICAL, sample_weight=weight)
    return model.predict_proba(frame.iloc[q])[:,1]


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    ref = latest_reference(y, group, fold, names, half, full)
    p = reference_score(full, ref, names.index('balanced_full'))
    frame = compact_features(raw)
    _, index, sizes = np.unique(group, return_inverse=True, return_counts=True)
    frame['BookingSize'] = sizes[index]
    frame['PassengerPosition'] = raw.PassengerId.str.split('_').str[1].astype(int)
    middle = (p > .3) & (p < .7)
    assert middle.sum() == 2368 and (ref[middle] != y[middle]).sum() == 968
    scores = {k: np.full(len(y),np.nan) for k in ('all', 'ambiguous', 'size_control', 'smooth')}
    counts = []
    for f in np.unique(fold):
        t, q = np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
        assert not np.isin(group[t],group[q]).any()
        proxy = np.full(len(t),np.nan)
        for a,b in GroupKFold(3,shuffle=True,random_state=631).split(t,groups=group[t]):
            assert not np.isin(group[t[a]],group[t[b]]).any()
            proxy[b] = estimate(frame,y,t[a],t[b])
        assert np.isfinite(proxy).all()
        keep = (proxy > .3) & (proxy < .7)
        subset = t[keep]
        rng = np.random.default_rng(516+int(f))
        control = np.sort(np.concatenate([rng.choice(t[y[t] == value], (y[subset] == value).sum(),replace=False)
                                          for value in (False,True)]))
        np.testing.assert_array_equal(np.bincount(y[subset].astype(int)),np.bincount(y[control].astype(int)))
        weights = (4*proxy*(1-proxy))**2
        for k,rows,w in [('all',t,None),('ambiguous',subset,None),('size_control',control,None),
                         ('smooth',t,weights/weights.mean())]:
            scores[k][q] = estimate(frame,y,rows,q,w)
        counts.append(len(subset))
        print('fold',f,'uncertain training rows',len(subset),flush=True)
    shared = ((full>=.5)!=y[:,None]).all(1)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    results = {}
    for key,prob in scores.items():
        assert np.isfinite(prob).all()
        for mode, use in [('standalone',prob),('middle_replace',np.where(middle,prob,p)),
                          ('middle_blend',np.where(middle,(prob+p)/2,p))]:
            delta = ((use>=.5)==y).astype(int) - (ref==y)
            results[key+'_'+mode] = dict(accuracy=float(((use>=.5)==y).mean()), fixed=int((delta==1).sum()),
                harmed=int((delta==-1).sum()), middle_fixed=int((delta[middle]==1).sum()),middle_harmed=int((delta[middle]==-1).sum()),
                common_fixed=int((delta[shared]==1).sum()),additional_net=int(delta.sum()-delta[dev].sum()),
                fold_net=[int(delta[fold==f].sum()) for f in np.unique(fold)],ci_pp=interval(delta,group))
    out = dict(ambiguous_train_n=counts,results=results)
    (ROOT/'reports/nov23_matched_learning_curve/uncertainty_specialist.json').write_text(json.dumps(out,indent=2)+'\n')
    for k,v in results.items():
        print(k,round(v['accuracy']*100,3),v['fixed'],v['harmed'],v['additional_net'])


if __name__ == '__main__':
    run()
