"""Out-of-group sequential covering of teacher errors, with a weight-matched parallel control."""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import error_trained_experts as experts
import relational_error_corrector as ref

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
OUT = ROOT / "reports/sequential_rescue_local"
ROUTES = {"first_only": [0], "parallel": [0, 1], "sequential": [0, 2]}


def predict(frame, cats, y, train, query, weights, seed):
    model = CatBoostClassifier(iterations=140, depth=3, learning_rate=.05, l2_leaf_reg=10,
                               cat_features=[c for c in cats if c in frame], thread_count=2,
                               random_seed=seed, verbose=False, allow_writing_files=False)
    model.fit(frame.iloc[train], y[train].astype(int), sample_weight=weights)
    return model.predict_proba(frame.iloc[query])[:, 1]


def gate_features(base, probs):
    sign = np.where(base >= .5, -1., 1.)
    margin = logit(np.clip(probs, .001, .999)) * sign[:, None]
    return np.column_stack([np.abs(ref.logit(base)), base >= .5, margin]), (margin > 0).any(axis=1)


def route(cal_base, cal_prob, cal_y, query_base, query_prob):
    x, challenged = gate_features(cal_base, cal_prob)
    xq, active = gate_features(query_base, query_prob)
    errors = (cal_base >= .5) != cal_y
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.15))
    model.fit(x[challenged], errors[challenged])
    return np.where(active, model.predict_proba(xq)[:, 1], 0.)


def run():
    inputs, splits, folds, _, _ = ref.overlap.load_authenticated()
    dev, y = inputs["dev"], inputs["y"].astype(bool)
    groups = np.asarray(inputs["groups"])
    with np.load(ART / "v108_exact_oof/oof.npz") as baseline:
        np.testing.assert_array_equal(baseline["dev_ids"], dev)
        np.testing.assert_array_equal(baseline["fold_id"], folds)
        raw = baseline["v108"]
    with np.load(ART / "consensus_error_routes.npz") as old:
        np.testing.assert_array_equal(old["dev"], dev)
        np.testing.assert_array_equal(old["fold"], folds)
        protected = old["combined_by_historical_agreement"].astype(bool)
    base = (raw >= .5) ^ protected
    assert (base != y).sum() == 1274
    protocol = {
        "first_head": "Bills view; fit all D with triple weight on authenticated rawMitra errors",
        "second_head": "Party features; weight seven for D examples missed by both teacher and group-OOF first head, triple for other teacher errors, unit for correct controls",
        "control": "Identical party head, preserving total weight per label and teacher-error stratum, spread uniformly among teacher errors of that label",
        "arbitration": "Each packet gates on separate C, logistic C=.15 among challenged passengers using prediction direction, baseline strength and signed expert logits. Average three packet risks (inactive=0); switch at .5, retain 87 original v115 changes",
        "scope": "Original 6954 adaptive random-split dev. D/C and first-head crossfit disjoint by party. Teacher and outer split random by passenger. Inner baseline rawMitra, outer v108 scores.",
    }
    ref.write(OUT / "protocol.json", protocol)
    frames, cats = experts.views(inputs)
    loc = pd.Series(np.arange(len(dev)), index=dev)
    member = np.full((len(y), 3, 3), np.nan)
    risks = {k: np.full((len(y), 3), np.nan) for k in ROUTES}
    train_log = []
    for i, (_, b) in enumerate(splits):
        for j in range(3):
            seed = 20261118 + 100 * i + j
            with np.load(ART / f"error_evidence_challengers/outer{i}_packet{j}.npz") as packet:
                d, c = [loc.loc[packet[k]].to_numpy() for k in ("D_ids", "C_ids")]
                d_base, c_base = packet["D_base"], packet["C_base"]
                np.testing.assert_array_equal(dev[b], packet["outer_query_ids"])
                assert not np.intersect1d(np.r_[dev[d], dev[c], dev[b]], packet["teacher_support_ids"]).size
            assert not np.intersect1d(groups[d], groups[c]).size
            assert not np.intersect1d(b, np.r_[d, c]).size
            wrong = (d_base >= .5) != y[d]
            weights = 1. + 2 * wrong
            first_oof = np.full(len(d), np.nan)
            for train, val in GroupKFold(3).split(d, groups=groups[d]):
                assert not np.intersect1d(groups[d[train]], groups[d[val]]).size
                first_oof[val] = predict(frames["bills"], cats, y, d[train], d[val], weights[train], seed)
            assert np.isfinite(first_oof).all()
            missed = wrong & ((first_oof >= .5) != y[d])
            cascade = weights + 4 * missed
            parallel = weights.copy()
            for label in (False, True):
                target = wrong & (y[d] == label)
                parallel[target] = cascade[target].mean()
                assert np.isclose(cascade[target].sum(), parallel[target].sum())
            q = np.r_[c, b]
            preds = np.column_stack([
                predict(frames[view], cats, y, d, q, w, seed)
                for view, w in [("bills", weights), ("matched_party", parallel), ("matched_party", cascade)]
            ])
            member[b, j] = preds[len(c):]
            for name, cols in ROUTES.items():
                selected = preds[:, cols]
                risks[name][b, j] = route(c_base, selected[:len(c)], y[c], raw[b], selected[len(c):])
            train_log.append(dict(fold=i, packet=j, teacher_errors=int(wrong.sum()), uncovered=int(missed.sum())))
    assert np.isfinite(member).all() and all(np.isfinite(v).all() for v in risks.values())
    catalog = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id, dev)
    np.testing.assert_array_equal(catalog.y, y)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    assert hard.sum() == 546
    def summarize(pred):
        out = experts.summarize(y, pred, base, hard, folds)
        out["hard_lost"] = int((hard & (base == y) & (pred != y)).sum())
        return out
    heads = member.mean(axis=1) >= .5
    rescue = (heads == y[:, None]) & (base != y)[:, None]
    decisions = {k: base ^ ((v.mean(1) >= .5) & ~protected) for k, v in risks.items()}
    result = dict(routes={k: summarize(v) for k, v in decisions.items()}, training=train_log,
                  standalone={k: summarize(heads[:, i]) for i, k in enumerate(["first", "parallel_second", "cascade_second"])},
                  oracle_extra_beyond_first={k: dict(all=int((rescue[:, i] & ~rescue[:, 0]).sum()),
                                                      hard=int((rescue[:, i] & ~rescue[:, 0] & hard).sum()))
                                             for k, i in [("parallel", 1), ("sequential", 2)]})
    cache = ART / "sequential_rescue_local.npz"
    np.savez_compressed(cache, dev=dev, folds=folds, member=member, **decisions)
    result["oof_sha256"] = ref.sha(cache)
    ref.write(OUT / "result.json", result)
    print(result["routes"], result["oracle_extra_beyond_first"], flush=True)


if __name__ == "__main__":
    run()
