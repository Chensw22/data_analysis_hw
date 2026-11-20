"""Full-roster inference for sequential error heads and calibrated four-center prototypes."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logsumexp
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import exaone_validation_data as data
import export_consensus_routes as routes
import export_majority_specialist as export
import full_challenger_margin as margin
import prototype_error_specialist as proto
import sequential_rescue_local as seq
from kaggle_workflow import sha256
from relational_error_corrector import logit, write

ART = export.ART
OUT = ART / "nov21_specialists"


def calibrated(x, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=.15, max_iter=500)).fit(x, y)


def fit_heads(frames, cats, d, c, raw, groups, y, seed):
    err = (raw[d] >= .5) != y[d]
    weights = 1. + 2 * err
    first = np.full(len(d), np.nan)
    for t, q in GroupKFold(3).split(d, groups=groups[d]):
        first[q] = seq.predict(frames["bills"], cats, y, d[t], d[q], weights[t], seed)
    assert np.isfinite(first).all()
    cascade = weights + 4 * (err & ((first >= .5) != y[d]))
    control = weights.copy()
    for label in [False, True]:
        mask = err & (y[d] == label)
        if mask.any():
            control[mask] = cascade[mask].mean()
    heads = []
    for name, w in [("bills", weights), ("matched_party", control), ("matched_party", cascade)]:
        view = frames[name]
        model = CatBoostClassifier(iterations=140, depth=3, learning_rate=.05, l2_leaf_reg=10,
                                   cat_features=[k for k in cats if k in view], thread_count=2,
                                   random_seed=seed, verbose=False, allow_writing_files=False)
        model.fit(view.iloc[d], y[d].astype(int), sample_weight=w)
        heads.append((name, model))
    cp = score_heads(heads, frames, c)
    gates = {}
    for key, cols in seq.ROUTES.items():
        fx, active = seq.gate_features(raw[c], cp[:, cols])
        target = (raw[c] >= .5) != y[c]
        gates[key] = make_pipeline(StandardScaler(), LogisticRegression(C=.15)).fit(fx[active], target[active])
    return heads, gates


def score_heads(heads, frames, idx):
    return np.column_stack([m.predict_proba(frames[k].iloc[idx])[:, 1] for k, m in heads])


def fit_clouds(x, positive, errors, seed):
    groups = []
    for direction in (False, True):
        centers = []
        rss, n = 0., 0
        for target in (False, True):
            selected = x[(positive == direction) & (errors == target)]
            assert len(selected) >= 4
            m = KMeans(n_clusters=4, n_init=5, random_state=seed).fit(selected)
            weights = np.bincount(m.labels_, minlength=4) / len(selected)
            keep = weights > 0
            centers.append((m.cluster_centers_[keep], np.log(weights[keep])))
            rss += m.inertia_
            n += selected.size
        groups.append((max(rss/n, 1e-4), centers))
    return groups


def ratios(x, positive, clouds):
    out = np.empty(len(x))
    for direction in (False, True):
        take = positive == direction
        var, centers = clouds[int(direction)]
        scores = [logsumexp(w - cdist(x[take], mu, metric="sqeuclidean") / (2*var), axis=1) for mu, w in centers]
        out[take] = np.clip(scores[1] - scores[0], -12., 12.)
    return out


def fit_prototypes(frame, d, c, raw, kd, kc, y, seed):
    d = d[(raw[d] >= .5) == (kd >= .5)]
    good = (raw[c] >= .5) == (kc >= .5)
    c, kc = c[good], kc[good]
    db, cb = raw[d] >= .5, raw[c] >= .5
    errors = db != y[d]
    prior = np.array([errors[db == s].mean() for s in (False, True)])
    views, signals = [], []
    for name in ("consumption", "passenger_cabin"):
        enc = proto.retrieval.View(name).fit(frame.iloc[d])
        x, q = enc.transform(frame.iloc[d]), enc.transform(frame.iloc[c])
        clouds = fit_clouds(x, db, errors, seed)
        s = ratios(q, cb, clouds)
        np.testing.assert_array_equal(s, proto.ratios(x, db, errors, q, cb, 4, seed))
        signals.append(s)
        views.append((enc, clouds))
    x = proto.design(raw[c], kc, signals)
    err = cb != y[c]
    gate = calibrated(x, err)
    directional = [calibrated(x[cb == sign, 1:], err[cb == sign]) for sign in (False, True)]
    return views, prior, gate, directional


def predict(packet, frames, compact, idx, anchor, witness):
    heads, gates, (views, prior, gate, directional) = packet
    voters = score_heads(heads, frames, idx)
    cb = anchor >= .5
    features = [ratios(enc.transform(compact.iloc[idx]), cb, clouds) for enc, clouds in views]
    x = proto.design(anchor, witness, features)
    by_class = np.empty(len(idx))
    for sign, clf in zip((False, True), directional):
        take = cb == sign
        by_class[take] = clf.predict_proba(x[take, 1:])[:, 1]
    result = dict(voters=voters, proto=gate.predict_proba(x)[:, 1],
                  geometric_risk=expit(logit(prior[cb.astype(int)]) + np.mean(features, axis=0)),
                  directional=by_class)
    for key, cols in seq.ROUTES.items():
        fx, active = seq.gate_features(anchor, voters[:, cols])
        risk = np.where(active, gates[key].predict_proba(fx)[:, 1], 0.)
        result[{"first_only":"first_risk", "sequential":"sequential_risk", "parallel":"parallel_risk"}[key]] = risk
    return result


def run():
    root = Path(__file__).resolve().parent
    train, test = [pd.read_csv(root / "data/raw" / f, usecols=lambda c: c != "Name") for f in ["train.csv", "test.csv"]]
    roster = pd.concat([train.drop(columns="Transported"), test], ignore_index=True)
    ids = test.PassengerId.to_numpy(str)
    y = train.Transported.to_numpy(bool)
    groups = roster.PassengerId.str.split("_").str[0].to_numpy()
    tq = np.arange(len(train), len(roster))
    assert not np.intersect1d(groups[tq], groups[:len(train)]).size
    compact = data.compact_features(roster)
    frames, cats = seq.experts.views(dict(dev=np.arange(len(roster)), passengers=roster.PassengerId.to_numpy()))
    anchor, _ = margin.load_reference(ids)
    witness = export.probabilities(ART / routes.HISTORICAL[3] / "test_probabilities.csv", ids)
    with np.load(export.TEACHER / "raw_probabilities.npz") as z:
        np.testing.assert_array_equal(train.PassengerId, z["train_passenger_ids"])
        raw, fold = z["oof_raw"], z["fold_ids"]
    OUT.mkdir(exist_ok=True)
    packets = []
    for j in range(3):
        with np.load(export.MARGIN / f"packet{j}.npz") as z, np.load(export.OUT / f"kumo_packet{j}.npz") as k:
            d, c = z["D_ids"], z["C_ids"]
            np.testing.assert_array_equal(k["train"], z["T_ids"])
            np.testing.assert_array_equal(k["query"], np.r_[d,c])
            np.testing.assert_array_equal(np.flatnonzero(fold == j), np.sort(k["query"]))
            assert not np.intersect1d(groups[d], groups[c]).size
            kd, kc = k["p"][:len(d)], k["p"][len(d):]
        heads, gates = fit_heads(frames, cats, d, c, raw, groups, y, 20261118+j)
        clouds = fit_prototypes(compact, d, c, raw, kd, kc, y, 20261119+j)
        packets.append((heads, gates, clouds))
    path = OUT / "model.joblib"
    joblib.dump(packets, path)
    def averaged(fitted):
        results = [predict(p, frames, compact, tq, anchor, witness) for p in fitted]
        return {k: np.mean([r[k] for r in results], axis=0) for k in results[0]}
    first = averaged(packets)
    replay = averaged(joblib.load(path))
    for k, a in first.items():
        np.testing.assert_array_equal(a, replay[k])
        assert np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
    dest = OUT / "components.npz"
    np.savez_compressed(dest, test_ids=ids, passenger_ids=ids, **first)
    write(OUT / "manifest.json", dict(rows=len(ids), fields=list(first), reload_difference=0,
          model_sha256=sha256(path), components_sha256=sha256(dest)))
    print(sha256(dest))


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
