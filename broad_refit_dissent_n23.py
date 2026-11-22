"""Measure persistent opposing evidence while keeping the complete evaluation fold out of every refit."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_weak_disagreement_n23 import ART, OUT, REPORT, SEEDS, archive
from exaone_validation_data import CATEGORICAL
from relational_error_corrector import write


def features(frame):
    inputs = pd.get_dummies(frame, columns=CATEGORICAL).astype(float)
    for name in BILLS:
        inputs[name] = np.log1p(inputs[name])
    basic = ["Age", "BookingSize", "CabinNumber", *[c for c in inputs if "_" in c]]
    billing = ["Age", "BookingSize", *BILLS]
    return [x.fillna(-2).to_numpy(float) for x in (inputs, inputs[basic], inputs[billing],
            inputs[["Age", "BookingSize", *[c for c in basic if c.startswith(("HomePlanet_", "CryoSleep_", "Destination_", "VIP_"))]]])]


def refit(inputs, y, groups, train, query):
    # Complementary halves of the same random partitions give each booking the
    # same total training weight. No query booking contributes labels.
    assert not np.intersect1d(groups[train], groups[query]).size
    reps = []
    for seed in (193, 557, 941):
        for a, _ in GroupKFold(2, shuffle=True, random_state=seed).split(train, groups=groups[train]):
            use = train[a]
            pred = []
            for j, x in enumerate([inputs[0], *inputs]):
                if j == 0:
                    fit = ExtraTreesClassifier(n_estimators=70, min_samples_leaf=5, max_features=.9,
                                               n_jobs=2, random_state=739)
                else:
                    fit = HistGradientBoostingClassifier(max_iter=95, l2_regularization=3, max_leaf_nodes=12,
                                                          early_stopping=False, random_state=913)
                fit.fit(x[use], y[use])
                pred.append(fit.predict_proba(x[query])[:, 1])
            reps.append(np.column_stack(pred))
    return np.stack(reps)


def evidence(base, members):
    direction = (base >= .5)
    odds = logit(base.clip(.002, .998))[:, None]
    # Every coordinate is oriented toward overturning the baseline's decision.
    contra = np.where(direction[None, :, None], 1-members, members)
    means = contra.mean(0)
    support = (contra >= .5).mean(0)
    stable_weak = (support[:, 2:] == 1).any(1)
    region = stable_weak & ~(support[:, :2] == 1).any(1)
    level = np.maximum(base, 1-base)
    centered = means-.5
    ordinary = np.column_stack([odds, level, centered, centered * (2*direction.astype(float)-1)[:,None]])
    # Compare using the identical six fits with and without their stability,
    # controlling for the mean opinion of each family.
    stability = np.column_stack([support, contra.std(0), np.min(contra, axis=0),
                                 (support[:, :2] == 1).sum(1), (support[:, 2:] == 1).sum(1), region])
    sign = (2*direction.astype(float)-1)[:,None]
    return dict(base=odds, means=ordinary, repeat=np.column_stack([ordinary, stability, stability*sign])), region


def estimate(x, y, a, b):
    scaler = StandardScaler().fit(x[a])
    learner = LogisticRegression(C=.12, max_iter=500)
    learner.fit(scaler.transform(x[a]), y[a])
    return learner.predict_proba(scaler.transform(x[b]))[:,1]


def run():
    row, y, frame, groups, _ = load()
    old_row, old_y, bases, _, cohorts, _, _ = archive()
    np.testing.assert_array_equal(row, old_row)
    np.testing.assert_array_equal(y, old_y)
    inputs = features(frame)
    plans = []
    for i, seed in enumerate(SEEDS):
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"], row)
            np.testing.assert_array_equal(z["base"], bases[i])
            plans.append((str(seed), z["fold"], bases[i], None))
    with np.load(ART/"nov23_incumbent_budget_transfer/oof.npz") as z:
        for field, value in (("row", row), ("y", y)):
            np.testing.assert_array_equal(z[field], value)
        plans.append(("v131", z["fold"], z["v108"], z["v131"]))
    OUT.mkdir(exist_ok=True)
    report, historical = {}, {}
    for name, fold, base, incumbent in plans:
        ref = base >= .5 if incumbent is None else incumbent.astype(bool)
        original = ref == y
        override = ref != (base >= .5)
        pred = {key: np.full(len(y), np.nan) for key in ("base", "means", "repeat", "repeat_only_in_region")}
        region = np.zeros(len(y), bool)
        for k in np.unique(fold):
            q = np.flatnonzero(fold == k)
            t = np.flatnonzero((fold != k) & ~np.isin(groups, groups[q]))
            cache = OUT/f"refits_{name}_{k}.npz"
            if cache.exists():
                with np.load(cache) as z:
                    for f, data in (("t",t), ("q",q), ("row",row)):
                        np.testing.assert_array_equal(z[f], data)
                    members = z["members"]
            else:
                members = refit(inputs, y, groups, t, q)
                np.savez_compressed(cache, t=t, q=q, row=row, members=members)
            x, region[q] = evidence(base[q], members)
            for a,b in GroupKFold(3, shuffle=True, random_state=667).split(q, groups=groups[q]):
                assert not np.intersect1d(groups[q[a]], groups[q[b]]).size
                for method, design in x.items():
                    pred[method][q[b]] = estimate(design, y[q], a, b)
            pred["repeat_only_in_region"][q] = np.where(region[q], pred["repeat"][q], base[q])
            print(name, "fold", k, flush=True)
        result = {}
        target = cohorts["weak409"] | cohorts["mirror1185"]
        for method, p in pred.items():
            assert np.isfinite(p).all()
            hit = np.where(override, ref, p >= .5) == y
            gain = hit.astype(int) - original
            risk = np.where(base >= .5, 1-p, p)
            result[method] = dict(accuracy=float(hit.mean()), fixed=int((gain>0).sum()), harmed=int((gain<0).sum()),
                by_fold=[int(gain[fold==k].sum()) for k in np.unique(fold)],
                cohort_fixed={k:int(((gain>0)&v).sum()) for k,v in cohorts.items()},
                cohort_harmed={k:int(((gain<0)&v).sum()) for k,v in cohorts.items()})
            if incumbent is None:
                result[method]["mirror_auc"] = float(roc_auc_score(cohorts["weak409"][target], risk[target]))
                historical.setdefault(method, []).append(hit)
        report[name] = dict(results=result, region_size=int(region.sum()),
                            in_region={k:int((region & m).sum()) for k,m in cohorts.items()})
        np.savez_compressed(OUT/f"decisions_{name}.npz", row=row, y=y, fold=fold, ref=ref, base_probability=base,
                            overrides=override, region=region, **{"p_"+k:v for k,v in pred.items()})
        write(REPORT/"refits.json", dict(plans=report))
    ref = (bases >= .5) == y
    stable = {}
    for method, hits in historical.items():
        hit = np.stack(hits)
        stable[method] = {k: int((hit.all(0) & m).sum()) for k,m in cohorts.items()}
        stable[method]["paired_gain_ci95_pp"] = interval((hit.astype(float)-ref).mean(0), groups)
    write(REPORT/"refits.json", dict(plans=report, stable=stable))
    print(stable, flush=True)


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
