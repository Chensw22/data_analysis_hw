"""Calibrate the full and amount-ablated predictors on distinct validation bookings."""
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import isotonic_regression
from sklearn.model_selection import GroupKFold

from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART, ROOT, SEEDS, read_panel
from relational_error_corrector import write
from transported_subclasses_n23 import stats


def monotone_surface(means, weights):
    # Dykstra projections give weighted least squares on the intersection of
    # the row/column monotonicity cones. All cells have positive prior weight.
    table = means.copy()
    adjustments = [np.zeros_like(table), np.zeros_like(table)]
    for iteration in range(1500):
        previous = table.copy()
        for axis in (0, 1):
            before = table + adjustments[axis]
            current = before if axis == 0 else before.T
            weight = weights if axis == 0 else weights.T
            projected = np.array([isotonic_regression(a, weights=w).x for a, w in zip(current, weight)])
            table = projected if axis == 0 else projected.T
            adjustments[axis] = before - table
        if np.max(abs(table-previous)) < 1e-9 and all(np.diff(table, axis=a).min() >= -1e-8 for a in (0, 1)):
            return table
    raise RuntimeError("Monotone surface failed to converge")


def knots(values, n):
    return np.unique(np.quantile(values, (np.arange(n)+.5)/n))


def predict(base, second, y, train, query, joint):
    first_axis = knots(base[train], 9)
    second_axis = knots(second[train], 7) if joint else np.array([0., 1.])
    coords = [np.searchsorted((axis[1:]+axis[:-1])/2, val[train]) for axis, val in
              [(first_axis,base), (second_axis,second)]]
    if not joint:
        coords[1][:] = 0
    shape = (len(first_axis),len(second_axis))
    n = np.zeros(shape)
    successes = np.zeros(shape)
    np.add.at(n,tuple(coords),1)
    np.add.at(successes,tuple(coords),y[train])
    if not joint:
        pooled = (successes[:,0]+14*first_axis)/(n[:,0]+14)
        p = isotonic_regression(pooled, weights=n[:,0]+14).x
        return np.interp(base[query],first_axis,p)
    # Match the total prior mass per full-model risk band to the univariate fit.
    prior = 14 / len(second_axis)
    mean = (successes+prior*first_axis[:,None])/(n+prior)
    table = monotone_surface(mean, n+prior)
    inputs = np.column_stack([base[query].clip(first_axis[0],first_axis[-1]),
                              second[query].clip(second_axis[0],second_axis[-1])])
    return RegularGridInterpolator((first_axis,second_axis),table)(inputs)


def permute(values, base, train, query, seed):
    rng = np.random.default_rng(seed)
    result = values.copy()
    bands = (base*20).astype(int)
    for part in (train,query):
        for band in np.unique(bands[part]):
            idx = part[bands[part]==band]
            result[idx]=rng.permutation(values[idx])
    return result


def paired_summary():
    y, hits, _ = read_panel()
    files = ART/"nov23_joint_risk_surface"
    models = ("joint","fixed_quarter","score_only","shuffled")
    accuracy = {name:[] for name in (*models,"base")}
    for seed in SEEDS:
        with np.load(files/f"{seed}.npz") as data:
            row=data["row"]
            np.testing.assert_array_equal(y,data["y"])
            for name in accuracy:
                accuracy[name].append((data[name]>=.5)==y)
    ids=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.str.split("_").str[0]
    _, groups=np.unique(ids.iloc[row],return_inverse=True)
    size=np.bincount(groups)
    baseline=np.array(accuracy["base"])
    np.testing.assert_array_equal(baseline,hits)
    result={}
    for name in models:
        correct=np.array(accuracy[name])
        difference=(correct.astype(float)-baseline).mean(0)
        delta=np.bincount(groups,weights=difference)
        rng=np.random.default_rng(132)
        draws=[]
        for _ in range(150):
            sample=rng.integers(0,len(size),size=(20,len(size)))
            draws.extend(100*delta[sample].sum(1)/size[sample].sum(1))
        stable=correct.all(0)&~hits.any(0)
        result[name]=dict(mean_gain_pp=float(100*difference.mean()),
                          booking_interval_pp=np.quantile(draws,[.025,.975]).tolist(),
                          stable_fixed=int(stable.sum()),stable_false_negative=int((stable&y).sum()),
                          stable_harmed=int((~correct.any(0)&baseline.all(0)).sum()))
    write(ROOT/"reports/nov23_joint_risk_surface/paired_summary.json",
          dict(method="Average three paired accuracy differences per person. Resample entire bookings 3000 times with seed 132, holding fitted predictions fixed.",results=result))


def run():
    row = pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    groups = pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.str.split("_").str[0]
    single = groups.map(groups.value_counts()).to_numpy()[row] == 1
    bookings = groups.to_numpy()[row]
    y, old, others = read_panel()
    persistent = ~old.any(0)
    hard = persistent & ~others.any(axis=(0,1))
    results = {}
    out = ART/"nov23_joint_risk_surface"
    out.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            base,fold = z["base"],z["fold"]
            verify(z,row,y,fold)
        with np.load(ART/f"nov23_simplex_likelihood/{seed}.npz") as z:
            verify(z,row,y,fold)
            np.testing.assert_array_equal(z["base"],base)
            alt = z["budget_prior"]
        scores = {k:np.full(len(y), np.nan) for k in ["score_only","joint","shuffled"]}
        for k in np.unique(fold):
            held = np.flatnonzero(fold==k)
            # Both frozen first-stage predictors excluded this entire fold.
            for a,b in GroupKFold(3,shuffle=True,random_state=145).split(held,groups=bookings[held]):
                t,q = held[a],held[b]
                assert not np.intersect1d(bookings[t],bookings[q]).size
                scores["score_only"][q] = predict(base,alt,y,t,q,False)
                scores["joint"][q] = predict(base,alt,y,t,q,True)
                scrambled = permute(alt,base,t,q,seed+int(k)*11+int(b[0]))
                scores["shuffled"][q] = predict(base,scrambled,y,t,q,True)
        scores["fixed_quarter"] = .75*base+.25*alt
        results[seed]={}
        for name, p in scores.items():
            assert np.isfinite(p).all()
            m=stats(p>=.5,base>=.5,y,persistent,hard,single)
            m["brier"]=float(np.mean((p-y)**2))
            m["base_brier"]=float(np.mean((base-y)**2))
            results[seed][name]=m
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**scores)
        print(seed, {n:(m["fixed"],m["harmed"]) for n,m in results[seed].items()},flush=True)
    write(ROOT/"reports/nov23_joint_risk_surface/result.json",dict(results=results))


if __name__ == "__main__":
    run()
    paired_summary()
