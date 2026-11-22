"""Estimate class-conditional label transitions on training bookings and test soft labels."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold

from exaone_validation_data import CATEGORICAL, compact_features

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
SEEDS = (20261112, 20261115, 20261116)


def predict(x, training, target, query):
    model = CatBoostClassifier(iterations=420, depth=6, learning_rate=.06, loss_function="CrossEntropy",
                               thread_count=2, random_seed=217, allow_writing_files=False, verbose=False)
    model.fit(x.iloc[training], target.astype(float))
    return model.predict_proba(x.iloc[query])[:, 1]


def estimate(p, y):
    # Use each observed class's mean self-confidence as its inclusion threshold.
    # Rescale assigned counts so each row retains its observed class prevalence.
    prob = np.column_stack([1 - p, p])
    threshold = np.array([prob[y == c, c].mean() for c in (0, 1)])
    qualifies = prob >= threshold
    picked = np.where(qualifies, prob, -1).argmax(1)
    valid = qualifies.any(1)
    table = np.zeros((2, 2))
    for observed in (0, 1):
        mask = (y == observed) & valid
        counts = np.bincount(picked[mask], minlength=2)
        assert counts.sum()
        table[observed] = counts / counts.sum() * (y == observed).sum()
    transition = table / table.sum(0)
    false_positive, false_negative = transition[1, 0], transition[0, 1]
    assert 0 <= false_negative + false_positive < 1
    latent = ((p - false_positive) / (1 - false_positive - false_negative)).clip(.001, .999)
    pos = latent * np.where(y, 1 - false_negative, false_negative)
    neg = (1 - latent) * np.where(y, false_positive, 1 - false_positive)
    return pos / (pos + neg), false_positive, false_negative


def run():
    inventory = pd.read_csv(ART / "nov23_global_error_inventory.csv")
    row, y = inventory.row.to_numpy(), inventory.target.to_numpy(bool)
    raw = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c not in ("Transported", "Name")).iloc[row]
    x = pd.get_dummies(compact_features(raw), columns=CATEGORICAL, dtype=float)
    groups = raw.PassengerId.str.split("_").str[0].to_numpy()
    folder = ART / "nov23_broad_noise_transition"
    folder.mkdir(exist_ok=True)
    report = {}
    hits = {}
    for seed in SEEDS:
        with np.load(ART / f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"], row)
            np.testing.assert_array_equal(z["y"], y)
            fold, base = z["fold"], z["base"]
        probs = {key: np.full(len(y), np.nan) for key in ("raw", "adjusted", "shuffled", "transfer")}
        diagnostics = []
        for f in np.unique(fold):
            t, q = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            assert not np.intersect1d(groups[t], groups[q]).size
            train_prob = np.full(len(t), np.nan)
            for a, b in GroupKFold(3, shuffle=True, random_state=119).split(t, groups=groups[t]):
                assert not np.intersect1d(groups[t[a]], groups[t[b]]).size
                train_prob[b] = predict(x, t[a], y[t[a]], t[b])
            assert np.isfinite(train_prob).all()
            soft, fp, fn = estimate(train_prob, y[t])
            rng = np.random.default_rng(127)
            shuffled = soft.copy()
            for label in (False, True):
                mask = y[t] == label
                shuffled[mask] = rng.permutation(soft[mask])
            probs["raw"][q] = predict(x, t, y[t], q)
            for name, target in (("adjusted", soft), ("shuffled", shuffled)):
                latent = predict(x, t, target, q)
                probs[name][q] = fp + (1 - fp - fn) * latent
            probs["transfer"][q] = expit(logit(base[q].clip(.001, .999)) +
                                         logit(probs["adjusted"][q]) - logit(probs["raw"][q].clip(.001, .999)))
            diagnostics.append(dict(fold=int(f), fp=fp, fn=fn, changed=int(((soft >= .5) != y[t]).sum())))
        probs["baseline"] = base
        np.savez_compressed(folder / f"{seed}.npz", row=row, y=y, fold=fold, **probs)
        metrics = {}
        for name, p in probs.items():
            assert np.isfinite(p).all()
            ok = (p >= .5) == y
            hits.setdefault(name, []).append(ok)
            metrics[name] = dict(accuracy=ok.mean())
            for other in ("raw", "baseline", "shuffled"):
                ref = (probs[other] >= .5) == y
                metrics[name][other] = [int((ok & ~ref).sum()), int((~ok & ref).sum())]
        report[seed] = dict(diagnostics=diagnostics, metrics=metrics)
        print(seed, metrics["adjusted"], metrics["transfer"], flush=True)
    baseline = np.asarray(hits["baseline"])
    persistent, ever = ~baseline.any(0), ~baseline.all(0)
    coverage = {}
    for name, records in hits.items():
        correct = np.asarray(records)
        coverage[name] = dict(persistent_fixed_each=(correct & persistent).sum(1).tolist(),
                              persistent_fixed_all=int((correct.all(0) & persistent).sum()),
                              ever_wrong_fixed_all=int((correct.all(0) & ever).sum()),
                              correct_to_wrong_all=int((~correct.any(0) & baseline.all(0)).sum()))
    out = ROOT / "reports/nov23_broad_noise_transition"
    out.mkdir(exist_ok=True)
    (out / "result.json").write_text(json.dumps(dict(split=report, coverage=coverage), indent=2) + "\n")


if __name__ == "__main__":
    run()
