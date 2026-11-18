"""Learn whether to override two agreeing predictors when a separate model objects."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import error_evidence_challengers as challengers
import iterate
import relational_error_corrector as ref
from route_consensus_errors import FAMILIES, cluster_interval
from teacher_distillation import kumo_inner

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
PACKETS = ART / "error_evidence_challengers"


def disagreement_features(base, witness, objections):
    direction = base >= .5
    sign = np.where(direction, -1., 1.)
    margins = ref.logit(np.column_stack([base, witness, objections])) * sign[:, None]
    # Opponent strength, agreement between challengers, and asymmetry by base class.
    return np.column_stack([margins, margins.min(axis=1), np.min(margins[:, 2:], axis=1),
                            direction, direction[:, None] * margins])


def risk(train, errors, eligible, query):
    if np.unique(errors[eligible]).size < 2:
        return np.full(len(query), (errors[eligible].sum() + 1) / (eligible.sum() + 2))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.15, max_iter=1000))
    model.fit(train[eligible], errors[eligible])
    return model.predict_proba(query)[:, 1]


def run():
    inp, splits, folds, bases, evidence = ref.overlap.load_authenticated()
    dev, y = inp["dev"], inp["y"].astype(bool)
    roster = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c not in ["Transported", "Name"])
    x, _ = iterate.build_features(roster, False)
    with np.load(ART / "v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"], dev)
        np.testing.assert_array_equal(z["fold_id"], folds)
        p = z["v108"]
    with np.load(ART / "consensus_error_routes.npz") as z:
        np.testing.assert_array_equal(z["dev"], dev)
        np.testing.assert_array_equal(z["fold"], folds)
        protected = z["combined_by_historical_agreement"]
    with np.load(PACKETS / "oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"], dev)
        np.testing.assert_array_equal(z["fold_id"], folds)
        members = z["members"]
    pos = pd.Index(dev)
    direction = p >= .5
    base = direction ^ protected
    unanimous = np.all(np.column_stack([bases[k] >= .5 for k in FAMILIES]) == direction[:, None], axis=1)
    designs = {k: np.zeros((len(y), 3)) for k in ["pooled", "agreement_trained"]}
    counts = []
    with threadpool_limits(limits=2):
        for outer, (a, b) in enumerate(splits):
            raw = ref.load_inner(outer, a, b, dev, y, evidence)
            witness = kumo_inner(outer, a, b, dev, y, evidence)
            outer_witness = bases["Kumo"][b]
            for j in range(3):
                stem = f"outer{outer}_packet{j}"
                with np.load(PACKETS / (stem + ".npz")) as z:
                    ids = np.r_[z["D_ids"], z["C_ids"]]
                    train_idx = pd.Index(dev[a]).get_indexer(ids)
                    assert (train_idx >= 0).all() and len(np.unique(ids)) == len(ids)
                    np.testing.assert_array_equal(z["outer_query_ids"], dev[b])
                    assert not np.intersect1d(np.r_[ids, dev[b]], z["teacher_support_ids"]).size
                    np.testing.assert_array_equal(raw[train_idx], np.r_[z["D_base"], z["C_base"]])
                    c_ids, c_probs = z["C_ids"], z["C_challengers"]
                models = joblib.load(PACKETS / (stem + "_challengers.joblib"))
                forest = models["forest"].predict_proba(x.loc[ids])[:, 1]
                encoded = models["latent_encoder"].transform(x.loc[ids, challengers.latent.COLS])
                objections = np.column_stack([forest, models["latent"].predict(encoded)])
                np.testing.assert_allclose(objections[-len(c_ids):], c_probs, rtol=0, atol=1e-12)
                anchor, other = raw[train_idx], witness[train_idx]
                active = np.any((objections >= .5) != (anchor[:, None] >= .5), axis=1)
                agree = (anchor >= .5) == (other >= .5)
                errors = (anchor >= .5) != y[pos.get_indexer(ids)]
                train = disagreement_features(anchor, other, objections)
                query = disagreement_features(p[b], outer_witness, members[b, j])
                challenged = np.any((members[b, j] >= .5) != direction[b, None], axis=1)
                for key, mask in [("pooled", active), ("agreement_trained", active & agree)]:
                    designs[key][b, j] = risk(train, errors, mask, query) * challenged
                counts.append(dict(outer=outer, packet=j, disputed_agreements=int((active & agree).sum()),
                                   errors_in_disputed_agreements=int((errors & active & agree).sum())))
    catalog = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id, dev)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    results, decisions = {}, {}
    for key, risks in designs.items():
        for cohort, mask in [("two_agree", (bases["Kumo"] >= .5) == direction), ("five_agree", unanimous)]:
            change = mask & ~protected & (risks.mean(axis=1) >= .5)
            fixed, broken = change & (base != y), change & (base == y)
            name = key + "_" + cohort
            decisions[name] = base ^ change
            results[name] = dict(accuracy=float(np.mean(decisions[name] == y)), fixed=int(fixed.sum()),
                harmed=int(broken.sum()), hard_fixed=int((hard & fixed).sum()), hard_harmed=int((hard & broken).sum()),
                fold_net=[int(fixed[b].sum()) - int(broken[b].sum()) for _, b in splits],
                party_bootstrap_pp=cluster_interval(change, base, y, inp["groups"]))
    ref.write(ROOT / "reports/consensus_dispute_specialist/result.json",
              dict(methods=results, train_packets=counts, base_correct=int((base == y).sum())))
    np.savez_compressed(ART / "consensus_dispute_specialist.npz", dev=dev, fold=folds, base=base, **decisions)
    print(results)


if __name__ == "__main__":
    run()
