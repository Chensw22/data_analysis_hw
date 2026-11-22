"""Learn systematic probability errors over overlapping large passenger populations."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS
from population_error_accounting_n23 import BILLS
from relational_error_corrector import write
from unlabelled_passenger_neighbours_n23 import evaluate


def conditions(raw, p, sizes, include_fields):
    rules = {"all": np.ones(len(raw), bool)}
    for cut in (.15, .30, .50, .70, .85):
        rules[f"score_ge_{cut}"] = p >= cut
        rules[f"score_lt_{cut}"] = p < cut
    if include_fields:
        for key in ("HomePlanet", "CryoSleep", "Destination", "Deck", "Side"):
            for value in raw[key].dropna().unique():
                rules[f"{key}_{value}"] = raw[key].eq(value).to_numpy()
        continuous = [(raw.Age, "age", [15, 25, 40]), (sizes, "booking_size", [2, 3]),
                      (raw[BILLS].gt(0).sum(axis=1), "active_bills", [1, 2, 3]),
                      (raw[BILLS].sum(axis=1, min_count=1), "total_bill", [1, 800, 1600]),
                      (raw[BILLS].isna().sum(axis=1), "missing_bill", [1])]
        continuous.extend((raw[key], key, [1, 200]) for key in BILLS)
        for values, name, cuts in continuous:
            for cut in cuts:
                rules[f"{name}_ge_{cut}"] = np.asarray(values >= cut)
                rules[f"{name}_lt_{cut}"] = np.asarray(values < cut)
    labels = list(rules)
    for a, b in combinations(labels[1:], 2):
        joined = rules[a] & rules[b]
        if joined.mean() >= .12:
            rules[f"{a}&{b}"] = joined
    # Identical coverage makes equivalent predicates interchangeable.
    seen = set()
    labels, arrays = [], []
    for label, arr in rules.items():
        key = np.packbits(arr).tobytes()
        if key not in seen:
            seen.add(key)
            labels.append(label)
            arrays.append(arr)
    return np.stack(arrays), labels


def audit(bank, names, base, y, discover, confirm, q):
    cover = bank[:, discover].mean(1)
    keep = (cover >= .15) & (bank[:, confirm].mean(1) >= .15)
    name = np.array(names)[keep]
    tables = [np.ascontiguousarray(bank[keep][:, ids], dtype=float) for ids in (discover, confirm, q)]
    parts = [discover, confirm, q]
    fitted = [logit(base[idx].clip(.0001, .9999)) for idx in parts]
    history = []
    for _ in range(8):
        p = [expit(v) for v in fitted]
        residual = tables[0] @ (y[discover]-p[0])
        variance = tables[0] @ (p[0]*(1-p[0])) + 1
        i = int(np.argmax(abs(residual)/np.sqrt(variance)))
        observed = tables[1][i] @ (y[confirm]-p[1])
        if residual[i] * observed <= 0:
            break
        precision = variance[i] + tables[1][i] @ (p[1]*(1-p[1])) + 8
        change = float(np.clip((residual[i]+observed)/precision, -.65, .65)*.5)
        for v, table in zip(fitted, tables):
            v += change*table[i]
        history.append(dict(rule=str(name[i]), discovery_rows=int(tables[0][i].sum()),
                            confirmation_rows=int(tables[1][i].sum()), query_rows=int(tables[2][i].sum()),
                            logit_change=change))
    return expit(fitted[-1]), history


def run():
    with np.load(ART/f"nov23_group_kernel/{SEEDS[0]}.npz") as ref:
        row,y=ref["row"],ref["y"]
    raw = pd.read_csv(ROOT/"data/raw/train.csv", usecols=lambda s: s not in {"Name", "Transported"})
    booking = raw.PassengerId.str.split("_").str[0]
    sizes = booking.map(booking.value_counts()).to_numpy()[row]
    group = booking.to_numpy()[row]
    raw = raw.iloc[row].copy()
    raw["Deck"], _, raw["Side"] = raw.Cabin.str.split("/", expand=True).to_numpy().T
    earth = raw.HomePlanet.eq("Earth").to_numpy()
    data = {}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"], row)
            np.testing.assert_array_equal(z["y"], y)
            data[seed] = z["base"], z["fold"]
    repeat = np.logical_and.reduce([(p >= .5) != y for p, _ in data.values()])
    result = {}
    out = ART/"nov23_broad_probability_audit"
    out.mkdir(exist_ok=True)
    for seed, (base, fold) in data.items():
        estimates, scores, logs = {}, {}, {}
        for name, fields in [("score_only", False), ("broad_fields", True)]:
            p = np.full(len(y), np.nan)
            logs[name] = []
            for f in np.unique(fold):
                held = np.flatnonzero(fold == f)
                # Build the label-free dictionary on this fold only. Base scores here
                # come from one model that excluded all of its passengers.
                bank, names = conditions(raw.iloc[held], base[held], sizes[held], fields)
                for t, q in GroupKFold(3, shuffle=True, random_state=142).split(held, groups=group[held]):
                    a, b = next(GroupShuffleSplit(1, train_size=.5, random_state=173).split(t, groups=group[held[t]]))
                    d, c = t[a], t[b]
                    assert not (set(group[held[d]]) & set(group[held[c]]) or set(group[held[t]]) & set(group[held[q]]))
                    p[held[q]], steps = audit(bank, names, base[held], y[held], d, c, q)
                    logs[name].append(steps)
            assert np.isfinite(p).all()
            estimates[name] = p
            scores[name] = evaluate(p, base, y, earth, repeat)
            scores[name]["brier_change"] = float(np.mean((p-y)**2-(base-y)**2))
        np.savez_compressed(out/f"{seed}.npz", row=row, y=y, base=base, fold=fold, **estimates)
        result[seed] = dict(scores=scores, rules=logs)
    write(ROOT/"reports/nov23_broad_probability_audit/result.json", result)


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
