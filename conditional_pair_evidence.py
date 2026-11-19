"""Learn whether a passenger's spending and location belong together under each outcome."""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar

import relational_error_corrector as ref
from exaone_validation_data import CATEGORICAL
from peer_bill_correction import BILLS
from route_consensus_errors import FAMILIES
from stable_consensus_selector import persistent_mask

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def block_scores(train, query, hypothesis, seed):
    rng = np.random.default_rng(seed)
    train = train.reset_index(drop=True)
    # Shuffling within these strata preserves each class's marginal distribution
    # of spending and of passenger attributes. Only their pairing is broken.
    buckets = train[["HomePlanet", "CryoSleep"]].assign(hypothesis=hypothesis)
    strata = list(buckets.groupby(list(buckets), sort=False).indices.values())
    blocks = [train.assign(hypothesis=hypothesis.astype(int))]
    for _ in range(2):
        source = np.arange(len(train))
        for idx in strata:
            if len(idx) > 1:
                order = rng.permutation(idx)
                source[order] = np.roll(order, 1)
        mixed = train.copy()
        mixed[BILLS] = train.iloc[source][BILLS].to_numpy()
        blocks.append(mixed.assign(hypothesis=hypothesis.astype(int)))
    model = CatBoostClassifier(iterations=260, depth=5, learning_rate=.065, l2_leaf_reg=6,
        cat_features=CATEGORICAL, thread_count=2, random_seed=seed, verbose=False, allow_writing_files=False)
    model.fit(pd.concat(blocks, ignore_index=True), np.repeat([1, 0, 0], len(train)),
              sample_weight=np.repeat([1., .5, .5], len(train)))
    s = [model.predict(query.assign(hypothesis=k), prediction_type="RawFormulaVal", thread_count=2) for k in (0, 1)]
    return s[1] - s[0]


def fit_strength(base, y, signal):
    offset = ref.logit(base)
    def loss(weight):
        z = offset + weight * signal
        return np.mean(np.logaddexp(0, z) - y * z) + .01 * weight ** 2
    out = minimize_scalar(loss, bounds=(-3, 3), method="bounded")
    assert out.success
    return float(out.x)


def run():
    inp, splits, folds, bases, evidence = ref.overlap.load_authenticated()
    y, dev = inp["y"].astype(bool), inp["dev"]
    x = inp["features"].iloc[dev].reset_index(drop=True)
    with np.load(ART / "v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"], dev)
        np.testing.assert_array_equal(z["fold_id"], folds)
        anchor = z["v108"]
    shifts = {k: np.zeros(len(y)) for k in ["true_class", "shuffled_class"]}
    weights = []
    for outer, (a, b) in enumerate(splits):
        teacher = ref.load_inner(outer, a, b, dev, y, evidence)
        inner = {k: np.full(len(a), np.nan) for k in shifts}
        query_scores = {k: [] for k in shifts}
        for j in range(3):
            with np.load(ART / f"error_evidence_challengers/outer{outer}_packet{j}.npz") as z:
                it = np.flatnonzero(np.isin(dev[a], z["teacher_support_ids"]))
                ic = np.flatnonzero(np.isin(dev[a], np.r_[z["D_ids"], z["C_ids"]]))
                np.testing.assert_array_equal(np.sort(np.r_[it, ic]), np.arange(len(a)))
                np.testing.assert_array_equal(z["outer_query_ids"], dev[b])
            t = a[it]
            q = np.r_[a[ic], b]
            seed = 20261119 + 100 * outer + j
            fake = y[t].copy()
            rng = np.random.default_rng(seed)
            for ids in x.iloc[t].groupby(["HomePlanet", "CryoSleep"], sort=False).indices.values():
                fake[ids] = rng.permutation(fake[ids])
            for k, hyp in [("true_class", y[t]), ("shuffled_class", fake)]:
                s = block_scores(x.iloc[t], x.iloc[q], hyp, seed)
                inner[k][ic] = s[:len(ic)]
                query_scores[k].append(s[len(ic):])
        for k in shifts:
            assert np.isfinite(inner[k]).all()
            w = fit_strength(teacher, y[a], inner[k])
            shifts[k][b] = w * np.mean(query_scores[k], axis=0)
            weights.append(dict(outer=outer, type=k, weight=w))
        print("Completed conditional pairs fold", outer, flush=True)
    catalog = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id, dev)
    np.testing.assert_array_equal(catalog.y, y)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp, bases, hard)
    pos = anchor >= .5
    agree = (np.column_stack([bases[k] >= .5 for k in FAMILIES]) == pos[:, None]).all(axis=1)
    methods, cache = {}, {}
    for k, shift in shifts.items():
        for cohort, mask in [("all", np.ones(len(y), bool)), ("consensus", agree)]:
            dec = np.where(mask, ref.logit(anchor) + shift >= 0, pos)
            fixed, harmed = (dec == y) & (pos != y), (dec != y) & (pos == y)
            name = k + "_" + cohort
            methods[name] = dict(fixed=int(fixed.sum()), harmed=int(harmed.sum()), accuracy=float(np.mean(dec == y)),
                fold_net=[int(fixed[b].sum()) - int(harmed[b].sum()) for _, b in splits],
                hard_fixed=int((hard & fixed).sum()), hard_harmed=int((hard & harmed).sum()),
                persistent_fixed=int((persistent & fixed).sum()), persistent_harmed=int((persistent & harmed).sum()))
            cache[name] = dec
    ref.write(ROOT / "reports/conditional_pair_evidence/result.json", dict(weights=weights, methods=methods,
        design="For each held-out teacher packet, contrast real rows with two bill-block permutations within outcome, planet "
               "and sleep strata on the training support only. Score every query under both hypothetical outcomes. "
               "Learn one regularized log-odds weight from within-outer-training OOF scores; apply to exact v108. "
               "Negative control shuffles outcome within planet and sleep before constructing the same task."))
    np.savez_compressed(ART / "conditional_pair_evidence.npz", dev=dev, fold=folds, anchor=pos, **cache)
    print(methods, flush=True)


if __name__ == "__main__":
    run()
