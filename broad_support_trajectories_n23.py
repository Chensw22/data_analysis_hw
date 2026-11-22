"""Paired passenger learning curves with balanced subsets of whole bookings."""
from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold

from exaone_validation_data import CATEGORICAL, compact_features

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
OUT = ART / "nov23_broad_support_trajectories"
REPORT = ROOT / "reports/nov23_broad_support_trajectories"
SEEDS = (20261112, 20261115, 20261116)
SUBSETS = [c for k in range(1, 5) for c in combinations(range(4), k)]
LEVEL = np.array(list(map(len, SUBSETS)))


def fit_bank(frame, groups, y, folds, seed):
    bank = np.full((len(SUBSETS), len(y)), np.nan)
    sizes = []
    for k in np.unique(folds):
        train, query = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        assert not np.intersect1d(groups[train], groups[query]).size
        blocks = []
        for _, block in GroupKFold(4, shuffle=True, random_state=seed + int(k)).split(train, groups=groups[train]):
            blocks.append(train[block])
        np.testing.assert_array_equal(np.sort(np.concatenate(blocks)), train)
        for i, subset in enumerate(SUBSETS):
            selected = np.sort(np.concatenate([blocks[j] for j in subset]))
            model = CatBoostClassifier(iterations=350, depth=6, learning_rate=.065, verbose=False,
                                       random_seed=1923, thread_count=3, allow_writing_files=False)
            model.fit(frame.iloc[selected], y[selected].astype(int), cat_features=CATEGORICAL)
            bank[i, query] = model.predict_proba(frame.iloc[query])[:, 1]
        sizes.append([len(block) for block in blocks])
        print(f"{seed} fold {k} complete", flush=True)
    assert np.isfinite(bank).all()
    return bank, sizes


def slope(scores, fractions):
    x = 1 / np.sqrt(fractions)
    return (x - x.mean()) @ scores / np.sum((x - x.mean()) ** 2)


def paired(correct, base):
    return dict(accuracy=float(correct.mean()), fixed=int((correct & ~base).sum()),
                harmed=int((~correct & base).sum()))


def run():
    inv = pd.read_csv(ART / "nov23_global_error_inventory.csv")
    rows, y = inv.row.to_numpy(), inv.target.to_numpy(bool)
    raw = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c not in ("Transported", "Name")).iloc[rows]
    groups = raw.PassengerId.str.split("_").str[0].to_numpy()
    frame = compact_features(raw)
    OUT.mkdir(exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    hits, bases, details = {}, [], {}
    for seed in SEEDS:
        with np.load(ART / f"nov23_group_kernel/{seed}.npz") as ref:
            np.testing.assert_array_equal(ref["row"], rows)
            np.testing.assert_array_equal(ref["y"], y)
            folds, base = ref["fold"], ref["base"]
        bank, sizes = fit_bank(frame, groups, y, folds, seed)
        np.savez_compressed(OUT / f"{seed}.npz", row=rows, fold=folds, y=y, bank=bank, subsets=LEVEL)
        means = np.array([bank[LEVEL == k].mean(axis=0) for k in range(1, 5)])
        scores = logit(np.clip(means, 1e-7, 1 - 1e-7))
        fractions = np.arange(1, 5) / 4
        transfer = slope(scores, fractions) * (1 / np.sqrt(1.5) - 1)
        estimate = scores[2] + slope(scores[:3], fractions[:3]) * (1 - 1 / np.sqrt(.75))
        probs = {"quarter": means[0], "half": means[1], "three_quarters": means[2],
                 "full": means[3], "full_extrapolated": expit(scores[3] + transfer),
                 "baseline_plus_trajectory": expit(logit(np.clip(base, 1e-7, 1 - 1e-7)) + transfer)}
        baseline_ok = (base >= .5) == y
        bases.append(baseline_ok)
        result = {}
        for name, p in probs.items():
            correct = (p >= .5) == y
            hits.setdefault(name, []).append(correct)
            result[name] = paired(correct, baseline_ok)
        details[seed] = dict(models=result, baseline=float(baseline_ok.mean()), block_sizes=sizes,
                             full_vs_quarter=paired(hits["full"][-1], hits["quarter"][-1]),
                             held_size_backtest_mae=dict(carry_forward=float(np.abs(means[2] - means[3]).mean()),
                                                         extrapolated=float(np.abs(expit(estimate) - means[3]).mean())))
    bases = np.asarray(bases)
    persistent, ever = ~bases.any(0), ~bases.all(0)
    coverage = {}
    for name, values in hits.items():
        a = np.array(values)
        coverage[name] = dict(ever_wrong_now_correct_all=int((a.all(0) & ever).sum()),
                              persistent_correct_all=int((a.all(0) & persistent).sum()),
                              persistent_correct_once_or_more=int((a.any(0) & persistent).sum()),
                              always_correct_now_wrong_all=int((~a.any(0) & bases.all(0)).sum()))
    q, f = np.asarray(hits["quarter"]), np.asarray(hits["full"])
    def transitions(mask):
        return dict(size=int(mask.sum()), fixes=((~q & f)[:, mask]).sum(1).tolist(),
                    harms=((q & ~f)[:, mask]).sum(1).tolist(),
                    fixed_every_time=int(((~q & f).all(0) & mask).sum()),
                    harmed_every_time=int(((q & ~f).all(0) & mask).sum()))
    result = dict(rows=len(y), ever_wrong=int(ever.sum()), persistent=int(persistent.sum()),
                  split_results=details, coverage=coverage,
                  quarter_to_full=dict(total=transitions(np.ones(len(y), bool)),
                                       ever_wrong=transitions(ever), persistent=transitions(persistent)))
    (REPORT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["quarter_to_full"]), flush=True)


if __name__ == "__main__":
    run()
