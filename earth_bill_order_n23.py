"""Measure and predict with all binary rules monotone in the five bills within a manifest stratum."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from archive_error_recovery_n23 import ART, ROOT, SEEDS
from relational_error_corrector import write


BILLS = ["FoodCourt", "ShoppingMall", "RoomService", "Spa", "VRDeck"]


def dominated_by(a, b):
    return np.all(a[:, None, :] <= b[None, :, :], axis=2)


def compatible_set(bad):
    # Alternating paths from unmatched left vertices give a minimum vertex cover
    # (left outside the reachable set, right inside). Keep its complement.
    graph = csr_matrix(bad)
    left, right = bad.shape
    if not left or not right:
        return np.ones(left, bool), np.ones(right, bool), 0
    match = maximum_bipartite_matching(graph, perm_type="column")
    paired = match >= 0
    assert np.all(bad[np.flatnonzero(paired), match[paired]])
    reached = ~paired
    while True:
        visited = np.asarray(graph[reached].getnnz(axis=0)).ravel() > 0
        updated = reached | (paired & visited[match])
        if np.array_equal(updated, reached):
            break
        reached = updated
    assert (~reached).sum()+visited.sum() == paired.sum()
    assert not bad[np.ix_(reached, ~visited)].any()
    return reached, ~visited, int(paired.sum())


def fit_rule(x, y, train, query):
    pos, neg = train[y[train]], train[~y[train]]
    conflict = dominated_by(x[pos], x[neg])
    a, b, cost = compatible_set(conflict)
    c, d, reverse_cost = compatible_set(conflict.T)
    assert reverse_cost == cost
    recommendations = []
    for p_keep, n_keep in [(a, b), (d, c)]:
        yes = dominated_by(x[pos[p_keep]], x[query]).any(axis=0)
        no = dominated_by(x[query], x[neg[n_keep]]).any(axis=1)
        assert not (yes & no).any()
        recommendations.append(yes.astype(int)-no.astype(int))
    first, second = recommendations
    return cost, np.where(first == second, first, 0)


def strata(raw, sizes):
    # Compare bills only with identical missingness patterns. A zero placeholder
    # compares only with the same missing bill, never with an observed zero.
    pattern = raw[BILLS].isna().to_numpy() @ (2 ** np.arange(5))
    cabin = raw.Cabin.str.split("/", expand=True)
    fields = pd.DataFrame(dict(missing=pattern, sleep=raw.CryoSleep, deck=cabin[0], side=cabin[2],
                               destination=raw.Destination, age=np.digitize(raw.Age.fillna(-1), [0, 13, 25, 45]),
                               party=sizes > 1), index=raw.index)
    presets = {"bills": ["missing"], "berth": ["missing", "sleep", "deck", "side"],
               "registration": list(fields.columns)}
    return {name: fields.groupby(cols, dropna=False, sort=False).ngroup().to_numpy() for name, cols in presets.items()}


def run():
    with np.load(ART/f"nov23_group_kernel/{SEEDS[0]}.npz") as z:
        row, y = z["row"], z["y"]
    raw = pd.read_csv(ROOT/"data/raw/train.csv", usecols=lambda s: s not in {"Name", "Transported"})
    bookings = raw.PassengerId.str.split("_").str[0]
    sizes = bookings.map(bookings.value_counts()).to_numpy()[row]
    groups = bookings.to_numpy()[row]
    raw = raw.iloc[row].reset_index(drop=True)
    earth = raw.HomePlanet.eq("Earth").to_numpy()
    x = raw[BILLS].fillna(0).to_numpy()*[1, 1, -1, -1, -1]
    roster = np.flatnonzero(earth)
    designs = strata(raw, sizes)
    data = {}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(row, z["row"])
            np.testing.assert_array_equal(y, z["y"])
            data[seed] = z["base"], z["fold"]
    repeat = np.logical_and.reduce([(b >= .5) != y for b, _ in data.values()])
    out = ART/"nov23_earth_bill_order"
    out.mkdir(exist_ok=True)
    bounds, results = {}, {}
    for name, cells in designs.items():
        membership = [roster[cells[roster] == s] for s in np.unique(cells[roster])]
        empty = np.empty(0, dtype=int)
        unavoidable = sum(fit_rule(x,y,pool,empty)[0] for pool in membership)
        _, duplicates = np.unique(np.column_stack([cells[roster], x[roster]]), axis=0, return_inverse=True)
        positives = np.bincount(duplicates, weights=y[roster]).astype(int)
        negatives = np.bincount(duplicates)-positives
        ties = int(np.minimum(positives, negatives).sum())
        assert 0 <= ties <= unavoidable
        bounds[name] = dict(earth=int(earth.sum()), strata=len(membership), minimum_mistakes=unavoidable,
                            mistakes_from_identical_inputs=ties, additional_from_monotonicity=unavoidable-ties,
                            maximum_earth_accuracy=1-unavoidable/earth.sum())
        results[name] = {}
        for seed, (base, fold) in data.items():
            unchanged = base >= .5
            rule = np.zeros(len(y), dtype=int)
            upper = int(earth.sum()-unavoidable+((unchanged==y) & ~earth).sum())
            for f in np.unique(fold):
                assert not np.intersect1d(groups[fold==f], groups[fold!=f]).size
                for pool in membership:
                    q = pool[fold[pool] == f]
                    if q.size:
                        _, rule[q] = fit_rule(x, y, pool[fold[pool]!=f], q)
            new = np.where(rule == 0, unchanged, rule == 1)
            fixed = (new==y) & (unchanged!=y)
            harmed = (new!=y) & (unchanged==y)
            results[name][seed] = dict(whole_oracle_ceiling=upper/len(y), whole_accuracy=float((new==y).mean()),
                recommendations=int((rule!=0).sum()), corrections=int(fixed.sum()), new_errors=int(harmed.sum()),
                fixed_false_alarms=int((fixed&~y).sum()), fixed_misses=int((fixed&y).sum()),
                repeat_fixed=int((earth&repeat&(new==y)).sum()))
            np.savez_compressed(out/f"{name}_{seed}.npz",row=row,y=y,fold=fold,base=base,direction=rule)
    write(ROOT/"reports/nov23_earth_bill_order/result.json", dict(bounds=bounds,heldout=results))


if __name__ == "__main__":
    run()
