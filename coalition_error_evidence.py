"""Decompose a passenger's evidence into three observed blocks and their interactions."""
import itertools
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import relational_error_corrector as ref
from exaone_validation_data import CATEGORICAL
from peer_bill_correction import BILLS
from route_consensus_errors import FAMILIES, cluster_interval
from teacher_distillation import kumo_inner

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
BLOCKS = [["HomePlanet", "CryoSleep", "Destination", "VIP", "Age"],
          ["Deck", "Side", "CabinNumber"], BILLS]


def full_and_partial(train, y, query, seed):
    # Separate fits estimate each conditional outcome probability directly.
    scores = np.empty((len(query), 8))
    scores[:, 0] = ref.logit((y.sum() + 1.) / (len(y) + 2.))
    for mask in range(1, 8):
        cols = list(itertools.chain.from_iterable(block for j, block in enumerate(BLOCKS) if mask & (1 << j)))
        cat = [c for c in cols if c in CATEGORICAL]
        fit = CatBoostClassifier(iterations=180, depth=5, learning_rate=.07, l2_leaf_reg=6.,
                                 cat_features=cat, thread_count=2, random_seed=seed, verbose=False,
                                 allow_writing_files=False)
        fit.fit(train[cols], y)
        scores[:, mask] = ref.logit(fit.predict_proba(query[cols])[:, 1])
    return scores


def features(base, witness, scores):
    direction = base >= .5
    sign = np.where(direction, -1., 1.)
    terms = scores.copy()
    for bit in (1, 2, 4):
        for mask in range(8):
            if mask & bit:
                terms[:, mask] -= terms[:, mask ^ bit]
    np.testing.assert_allclose(terms.sum(axis=1), scores[:, 7], atol=1e-10)
    basic = np.column_stack([direction, np.abs(ref.logit(base)), sign * ref.logit(witness), sign * scores[:, 7]])
    extras = terms[:, 1:] * sign[:, None]
    # The full margin is already in basic. Omit the redundant three-way term.
    return {"full_only": basic, "block_interactions": np.column_stack([basic, extras[:, :6],
            np.abs(terms[:, [3, 5, 6]]), np.std(scores[:, [3, 5, 6]], axis=1)])}


def run():
    inp, splits, fold, bases, evidence = ref.overlap.load_authenticated()
    y, dev = inp["y"], inp["dev"]
    frame = inp["features"].iloc[dev].reset_index(drop=True)
    with np.load(ART / "v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"], dev)
        np.testing.assert_array_equal(z["fold_id"], fold)
        p = z["v108"]
    with np.load(ART / "consensus_error_routes.npz") as z:
        np.testing.assert_array_equal(z["dev"], dev)
        np.testing.assert_array_equal(z["fold"], fold)
        protected = z["combined_by_historical_agreement"]
    base = (p >= .5) ^ protected
    risks = {k: np.zeros((len(y), 3)) for k in ["full_only", "block_interactions"]}
    for outer, (a, b) in enumerate(splits):
        raw = ref.load_inner(outer, a, b, dev, y, evidence)
        kumo = kumo_inner(outer, a, b, dev, y, evidence)
        for j in range(3):
            rows = ART / f"error_evidence_challengers/outer{outer}_packet{j}.npz"
            with np.load(rows) as z:
                t = np.flatnonzero(np.isin(dev[a], z["teacher_support_ids"]))
                c = np.flatnonzero(np.isin(dev[a], np.r_[z["C_ids"], z["D_ids"]]))
                np.testing.assert_array_equal(np.sort(dev[a[t]]), np.sort(z["teacher_support_ids"]))
                np.testing.assert_array_equal(z["outer_query_ids"], dev[b])
            assert len(t) + len(c) == len(a) and not np.intersect1d(t, c).size
            q = np.r_[a[c], b]
            scores = full_and_partial(frame.iloc[a[t]], y[a[t]], frame.iloc[q], 20261118 + 100 * outer + j)
            train = features(raw[c], kumo[c], scores[:len(c)])
            query = features(p[b], bases["Kumo"][b], scores[len(c):])
            errors = (raw[c] >= .5) != y[a[c]]
            for key in risks:
                fit = make_pipeline(StandardScaler(), LogisticRegression(C=.15, max_iter=1000))
                fit.fit(train[key], errors)
                risks[key][b, j] = fit.predict_proba(query[key])[:, 1]
        print("Finished outer fold", outer, flush=True)
    import pandas as pd
    catalog = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id, dev)
    np.testing.assert_array_equal(catalog.y, y)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    hist_agree = (np.column_stack([bases[k] >= .5 for k in FAMILIES]) == (p >= .5)[:, None]).all(axis=1)
    results, decisions = {}, {}
    for k, votes in risks.items():
        flip = ~protected & (votes.mean(axis=1) >= .5)
        fixed, harmed = flip & (base != y), flip & (base == y)
        decisions[k] = base ^ flip
        results[k] = dict(accuracy=float(np.mean(decisions[k] == y)), fixed=int(fixed.sum()), harmed=int(harmed.sum()),
            hard_fixed=int((fixed & hard).sum()), hard_harmed=int((harmed & hard).sum()),
            five_agree_fixed=int((fixed & hist_agree).sum()), five_agree_harmed=int((harmed & hist_agree).sum()),
            fold_net=[int(fixed[b].sum()) - int(harmed[b].sum()) for _, b in splits],
            party_bootstrap_pp=cluster_interval(flip, base, y, inp["groups"]))
    ref.write(ROOT / "reports/coalition_error_evidence/result.json", results)
    np.savez_compressed(ART / "coalition_error_evidence.npz", dev=dev, fold=fold, base=base, **decisions)
    print(results)


if __name__ == "__main__":
    run()
