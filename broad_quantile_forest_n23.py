"""Honest forest leaves estimate conditional bill distribution ranks under both outcomes."""
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel, predictions
from broad_noise_transition_n23 import ART, ROOT, SEEDS
from conditional_bill_likelihood import BILLS
from exaone_validation_data import CATEGORICAL, compact_features
from fresh_family_rescue_n23 import NEW


def distribution(x, bills, y, g, t, q):
    signals = []
    for a, b in GroupKFold(2, shuffle=True, random_state=532).split(t, groups=g[t]):
        grow, support = t[a], t[b]
        # A separate set of bookings supplies the empirical distributions in each leaf.
        response = np.column_stack([np.log1p(np.maximum(bills[grow], 0))/4,
                                    bills[grow] == 0, bills[grow] < 0])
        forest = ExtraTreesRegressor(n_estimators=32, min_samples_leaf=45, random_state=564,
                                     max_features=.8, n_jobs=2).fit(x[grow], response)
        leaf, query_leaf = forest.apply(x[support]), forest.apply(x[q])
        parts = []
        for label in (None, False, True):
            sup = np.ones(len(support), bool) if label is None else y[support] == label
            measurements = []
            for k in range(bills.shape[1]):
                query = bills[q, k]
                # Put missing records at -1 as a discrete measurement state, and
                # preserve the zero-spending atom in the two one-sided ranks.
                totals = np.zeros((len(q), 2))
                global_ref = np.sort(bills[support[sup], k])
                prior = np.column_stack([np.searchsorted(global_ref, query, side=side) / len(global_ref)
                                         for side in ("left", "right")])
                for j in range(leaf.shape[1]):
                    for node in np.unique(query_leaf[:, j]):
                        ids = np.flatnonzero(query_leaf[:, j] == node)
                        vals = np.sort(bills[support[sup & (leaf[:, j] == node)], k])
                        counts = np.column_stack([np.searchsorted(vals, query[ids], side=side)
                                                  for side in ("left", "right")])
                        totals[ids] += (counts + 12 * prior[ids]) / (len(vals) + 12)
                left, right = (totals / leaf.shape[1]).T
                mid = (left + right) / 2
                measurements.extend([mid, np.log(np.clip(2*np.minimum(mid, 1-mid), .001, 1)),
                                     np.log(np.clip(right-left, .001, 1))])
            parts.append(np.column_stack(measurements))
        signals.append(parts)
    return np.mean(signals, axis=0)


def run():
    row, y, base, old, _, _, _ = panel()
    extra = dict(**NEW, native_carte=("carte", "p"))
    bundle = [predictions(row, y, s, extra) for s in SEEDS]
    historical = np.concatenate([old, np.stack([v[1] for v in bundle])], axis=2)
    base_ok = (base >= .5) == y
    persistent, ever = ~base_ok.any(0), ~base_ok.all(0)
    never = persistent & ~((historical >= .5) == y[None, :, None]).any(axis=(0, 2))
    no_stable = persistent & ~((historical >= .5) == y[None, :, None]).all(0).any(1)
    raw = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c:c not in ("Transported","Name")).iloc[row]
    frame = compact_features(raw)
    x = pd.get_dummies(frame.drop(columns=BILLS),columns=CATEGORICAL,dtype=float).fillna(-1).to_numpy()
    bills = frame[BILLS].fillna(-1).to_numpy()
    groups = raw.PassengerId.str.split("_").str[0].to_numpy()
    out = ART / "nov23_broad_quantile_forest"
    out.mkdir(exist_ok=True)
    hits, results = {}, {}
    for i, (seed, (ref, _, folds)) in enumerate(zip(SEEDS, bundle)):
        np.testing.assert_array_equal(ref, base[i])
        scores = {k:np.full(len(y),np.nan) for k in ("baseline_readout", "pooled", "conditional", "standalone")}
        for f in np.unique(folds):
            t, q = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            assert not np.intersect1d(groups[t],groups[q]).size
            pooled, neg, pos = distribution(x,bills,y,groups,t,q)
            b = logit(ref[q].clip(.001,.999))
            for a, v in GroupKFold(3,shuffle=True,random_state=947).split(q,groups=groups[q]):
                assert not np.intersect1d(groups[q[a]],groups[q[v]]).size
                z = np.column_stack([pooled, pos-neg])
                for name, inputs in (("baseline_readout",b[:,None]), ("pooled",np.column_stack([b,pooled])),
                                     ("conditional",np.column_stack([b,z])), ("standalone",z)):
                    scaler = StandardScaler().fit(inputs[a])
                    model = LogisticRegression(C=.03,max_iter=200).fit(scaler.transform(inputs[a]),y[q[a]])
                    scores[name][q[v]] = model.predict_proba(scaler.transform(inputs[v]))[:,1]
        np.savez_compressed(out / f"{seed}.npz", row=row, y=y, fold=folds, **scores)
        results[seed] = {}
        for name, p in scores.items():
            assert np.isfinite(p).all()
            ok = (p >= .5) == y
            hits.setdefault(name,[]).append(ok)
            metrics = dict(accuracy=ok.mean())
            for k,r in (("baseline",base_ok[i]), ("pooled",(scores["pooled"]>=.5)==y)):
                metrics[k]=[int((ok & ~r).sum()),int((~ok & r).sum())]
            results[seed][name] = metrics
        print(seed,results[seed]["conditional"],flush=True)
    coverage = {}
    for name,h in hits.items():
        a = np.asarray(h)
        coverage[name]=dict(ever_stable=int((a.all(0)&ever).sum()), persistent_stable=int((a.all(0)&persistent).sum()),
                            newly_stable=int((a.all(0)&no_stable).sum()), never_any=int((a.any(0)&never).sum()),never_stable=int((a.all(0)&never).sum()))
    out = ROOT / "reports/nov23_broad_quantile_forest"
    out.mkdir(exist_ok=True)
    (out/"result.json").write_text(json.dumps(dict(results=results,coverage=coverage),indent=2)+"\n")


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
