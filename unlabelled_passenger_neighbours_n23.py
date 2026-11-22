"""Test whether label-free neighbourhoods constrain predictions on unseen bookings."""
import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

from archive_error_recovery_n23 import ART, ROOT, SEEDS
from class_analogue_experts_n23 import design
from exaone_validation_data import compact_features
from relational_error_corrector import write


def neighbours(x, bookings):
    possible = NearestNeighbors(n_neighbors=32, algorithm="brute").fit(x).kneighbors(x, return_distance=False)
    selected = []
    for i, candidates in enumerate(possible):
        different = candidates[bookings[i] != bookings[candidates]]
        selected.append(different[:6])
    indices = np.stack(selected)
    assert indices.shape == (len(x), 6)
    return torch.as_tensor(indices)


def fit_model(x, y, t, q, start_test, nearby, mode):
    # Reuse initial weights, supervised batches and unlabeled anchors across all controls.
    torch.manual_seed(914)
    rng = np.random.default_rng(416)
    model = nn.Sequential(nn.Linear(x.shape[1], 96), nn.SiLU(), nn.Linear(96, 64),
                          nn.SiLU(), nn.Linear(64, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=.0025, weight_decay=.002)
    test = x[start_test:]
    for epoch in range(22):
        for batch in np.array_split(rng.permutation(t), 15):
            anchors = rng.choice(len(test), size=128, replace=False)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(x[batch]).flatten(), y[batch])
            if mode != "supervised":
                anchor = test[anchors]
                with torch.no_grad():
                    target = torch.sigmoid(model(anchor))
                other = test[nearby[anchors]]
                between = .5 * (anchor[:, None, :] + other)
                compared = model(between).squeeze(-1)
                costs = F.binary_cross_entropy_with_logits(compared, target.expand_as(compared), reduction="none")
                # The entropy of the detached target is constant over directions.
                penalty = costs.max(dim=1).values if mode == "worst" else costs[:, epoch % 6]
                loss = loss + .8 * penalty.mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return torch.sigmoid(model(x[q])).flatten().numpy()


def learn_reversal(base, worst, standard, earth, group, y, folds):
    initial = base >= .5
    error = initial != y
    direction = np.where(initial, 1., -1.)
    native = direction * logit(base.clip(.001, .999))
    scores = [direction * logit(s.clip(.001, .999)) for s in (worst, standard)]
    probability = {key: base.copy() for key in ("risk_only", "earth_reversal")}
    for f in np.unique(folds):
        for label in (False, True):
            rows = np.flatnonzero((folds == f) & earth & (initial == label))
            for a, b in GroupKFold(3, shuffle=True, random_state=422).split(rows, groups=group[rows]):
                fit, query = rows[a], rows[b]
                assert not np.intersect1d(group[fit], group[query]).size
                for key, features in [("risk_only", [native]), ("earth_reversal", [native, *scores])]:
                    inputs = np.column_stack(features)
                    reg = LogisticRegression(C=.25).fit(inputs[fit], error[fit])
                    p_wrong = reg.predict_proba(inputs[query])[:, 1]
                    probability[key][query] = 1-p_wrong if label else p_wrong
    return probability


def evaluate(p, base, y, earth, repeat):
    correct = (p >= .5) == y
    before = (base >= .5) == y
    fixed, harmed = correct & ~before, ~correct & before
    return dict(accuracy=float(correct.mean()), fixed=int(fixed.sum()), harmed=int(harmed.sum()),
                earth_fixed_missed=int((earth & fixed & y).sum()), earth_fixed_false_alarm=int((earth & fixed & ~y).sum()),
                earth_new_missed=int((earth & harmed & y).sum()), earth_new_false_alarm=int((earth & harmed & ~y).sum()),
                repeat=int((repeat & correct).sum()), earth_repeat=int((earth & repeat & correct).sum()))


def run():
    first = ART/f"nov23_group_kernel/{SEEDS[0]}.npz"
    with np.load(first) as ref:
        row, y = ref["row"], ref["y"]
    def roster(file):
        return pd.read_csv(ROOT/f"data/raw/{file}.csv", usecols=lambda k: k not in {"Name", "Transported"})
    raw = pd.concat([roster("train").iloc[row], roster("test")], ignore_index=True)
    group = raw.PassengerId.str.split("_").str[0].to_numpy()
    n = len(y)
    assert len(raw)-n == 4277 and not np.intersect1d(group[:n], group[n:]).size
    earth = raw.HomePlanet.eq("Earth").to_numpy()[:n]
    frame = compact_features(raw)
    reference = {}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            np.testing.assert_array_equal(row, ref["row"])
            np.testing.assert_array_equal(y, ref["y"])
            reference[seed] = ref["base"], ref["fold"]
    repeat = np.logical_and.reduce([(p >= .5) != y for p, _ in reference.values()])
    assert (repeat.sum(), (repeat & earth).sum()) == (1070, 804)
    labels = torch.tensor(y, dtype=torch.float32)
    folder = ART/"nov23_unlabelled_neighbours"
    folder.mkdir(exist_ok=True)
    results = {}
    for seed, (base, folds) in reference.items():
        predictions = {k: np.full(n, np.nan) for k in ("supervised", "ordinary", "worst")}
        for fold in np.unique(folds):
            t, q = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            assert not np.intersect1d(group[t], group[q]).size
            x, enhanced = design(frame, t)
            nearby = neighbours(x[n:].astype("float32"), group[n:])
            values = torch.tensor(enhanced, dtype=torch.float32)
            for name, out in predictions.items():
                out[q] = fit_model(values, labels, t, q, n, nearby, name)
            print(seed, "completed fold", fold, flush=True)
        for name, p in list(predictions.items()):
            assert np.isfinite(p).all()
            predictions[name + "_quarter"] = .75*base + .25*p
        predictions.update(learn_reversal(base, predictions["worst"], predictions["supervised"], earth, group[:n], y, folds))
        predictions["baseline"] = base
        results[str(seed)] = {k: evaluate(p, base, y, earth, repeat) for k, p in predictions.items()}
        np.savez_compressed(folder/f"{seed}.npz", row=row, y=y, fold=folds, **predictions)
    write(ROOT/"reports/nov23_unlabelled_neighbours/result.json", results)


if __name__ == "__main__":
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
