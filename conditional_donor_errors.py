"""Test if a prediction's dependence on one passenger attribute block identifies consensus errors."""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import relational_error_corrector as ref
from exaone_validation_data import CATEGORICAL
from peer_bill_correction import BILLS
from route_consensus_errors import FAMILIES
from stable_consensus_selector import persistent_mask
from teacher_distillation import kumo_inner

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
BLOCKS = [BILLS, ["Deck", "Side", "CabinNumber"]]


def encode(reference, query, cols):
    parts = []
    for c in cols:
        if c in CATEGORICAL:
            levels = np.asarray(reference[c].unique())
            parts.append(query[c].to_numpy()[:, None] == levels)
        else:
            a = np.sort(reference[c].dropna().to_numpy())
            x = query[c].to_numpy()
            ranks = (np.searchsorted(a, x) + np.searchsorted(a, x, side="right")) / (2 * len(a))
            parts.extend([np.where(np.isnan(x), .5, ranks)[:, None], np.isnan(x)[:, None]])
    return np.column_stack(parts).astype(float)


def donor_ids(train, query, train_groups, query_groups, withheld, rng):
    keep = [c for c in train if c not in withheld]
    x, q = encode(train, train, keep), encode(train, query, keep)
    eligible = NearestNeighbors(n_neighbors=36).fit(x).kneighbors(q, return_distance=False)
    matched, random = [], []
    for row, group in zip(eligible, query_groups):
        pool = row[train_groups[row] != group][:24]
        assert len(pool) >= 8
        matched.append(rng.choice(pool, 8, replace=False))
        # Same eight donors per passenger budget, without matching passenger attributes.
        sample = rng.integers(len(train), size=24)
        legal = np.unique(sample[train_groups[sample] != group])
        assert len(legal) >= 8
        random.append(rng.choice(legal, 8, replace=False))
    return np.array(matched), np.array(random)


def response(model, train, query, baseline, cols, indices):
    width = indices.shape[1]
    x = query.iloc[np.repeat(np.arange(len(query)), width)].reset_index(drop=True)
    for c in cols:
        x[c] = train[c].to_numpy()[indices.ravel()]
    p = model.predict_proba(x, thread_count=2)[:, 1].reshape(-1, width)
    signed = (1 - 2 * (baseline >= .5).astype(float))[:, None] * ref.logit(p)
    return np.column_stack([signed.mean(axis=1), signed.std(axis=1), (signed > 0).mean(axis=1)])


def train_scores(raw, witness, intact):
    sign = 1 - 2 * (raw >= .5).astype(float)
    return np.column_stack([raw >= .5, np.abs(ref.logit(raw)), sign * ref.logit(witness), sign * ref.logit(intact)])


def run():
    inp, splits, fold, bases, evidence = ref.overlap.load_authenticated()
    dev, y, groups = inp["dev"], inp["y"].astype(bool), np.asarray(inp["groups"])
    x = inp["features"].iloc[dev].reset_index(drop=True)
    with np.load(ART / "v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(dev, z["dev_ids"])
        np.testing.assert_array_equal(fold, z["fold_id"])
        p = z["v108"]
    anchor = p >= .5
    agree = np.all(np.column_stack([bases[k] >= .5 for k in FAMILIES]) == anchor[:, None], axis=1)
    scores = {k: np.zeros((len(y), 3)) for k in ["original", "random_donors", "matched_donors"]}
    for outer, (a, b) in enumerate(splits):
        teacher = ref.load_inner(outer, a, b, dev, y, evidence)
        witness = kumo_inner(outer, a, b, dev, y, evidence)
        for j in range(3):
            with np.load(ART / f"error_evidence_challengers/outer{outer}_packet{j}.npz") as z:
                support = np.flatnonzero(np.isin(dev[a], z["teacher_support_ids"]))
                packet = np.flatnonzero(np.isin(dev[a], np.r_[z["D_ids"], z["C_ids"]]))
                np.testing.assert_array_equal(np.sort(np.r_[support, packet]), np.arange(len(a)))
                np.testing.assert_array_equal(z["outer_query_ids"], dev[b])
            t, c = a[support], a[packet]
            q = np.r_[c, b]
            seed = 20261119 + 100 * outer + j
            model = CatBoostClassifier(iterations=300, depth=5, learning_rate=.065, l2_leaf_reg=6,
                cat_features=CATEGORICAL, thread_count=2, random_seed=seed, verbose=False, allow_writing_files=False)
            model.fit(x.iloc[t], y[t])
            raw = np.r_[teacher[packet], p[b]]
            other = np.r_[witness[packet], bases["Kumo"][b]]
            basic = train_scores(raw, other, model.predict_proba(x.iloc[q], thread_count=2)[:, 1])
            designs = {k: [basic] for k in scores}
            rng = np.random.default_rng(seed)
            for cols in BLOCKS:
                mat, rnd = donor_ids(x.iloc[t], x.iloc[q], groups[t], groups[q], cols, rng)
                for k, ids in [("matched_donors", mat), ("random_donors", rnd)]:
                    designs[k].append(response(model, x.iloc[t], x.iloc[q], raw, cols, ids))
            eligible = (raw[:len(c)] >= .5) == (other[:len(c)] >= .5)
            target = (teacher[packet] >= .5) != y[c]
            for k, parts in designs.items():
                v = np.column_stack(parts)
                gate = make_pipeline(StandardScaler(), LogisticRegression(C=.15, max_iter=1000))
                gate.fit(v[:len(c)][eligible], target[eligible])
                scores[k][b, j] = gate.predict_proba(v[len(c):])[:, 1]
        print("Completed outer fold", outer, flush=True)
    catalog = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id, dev)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp, bases, hard)
    result, decisions = {}, {}
    for k, risks in scores.items():
        flip = agree & (risks.mean(axis=1) >= .5)
        decisions[k] = anchor ^ flip
        fixed, harmed = flip & (anchor != y), flip & (anchor == y)
        result[k] = dict(accuracy=float((decisions[k] == y).mean()), fixed=int(fixed.sum()), harmed=int(harmed.sum()),
                         fold_net=[int(fixed[b].sum()) - int(harmed[b].sum()) for _, b in splits],
                         hard_fixed=int((hard & fixed).sum()), hard_harmed=int((hard & harmed).sum()),
                         persistent_fixed=int((persistent & fixed).sum()), persistent_harmed=int((persistent & harmed).sum()))
    np.savez_compressed(ART / "conditional_donor_errors.npz", dev=dev, fold=fold, anchor=anchor, **decisions,
                        **{k + "_risk": v for k, v in scores.items()})
    ref.write(ROOT / "reports/conditional_donor_errors/result.json",
              dict(rows=len(y), eligible=int(agree.sum()), methods=result, baseline_correct=int((anchor == y).sum()),
                   design="For each packet, a tree fit only on teacher support scores observed and donor bills/cabins. "
                   "Sample eight of the nearest 24 other-party donors, or eight random donors as a control. "
                   "Fit gates on held-out packet rows with agreeing raw and Kumo teachers. Mean three risks at 0.5; "
                   "only override v108 when all five reference families agree with it. Adaptive development splits."))
    print(result, flush=True)


if __name__ == "__main__":
    run()
