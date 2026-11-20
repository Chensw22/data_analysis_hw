"""Full-roster counterpart of the matched-donor error score for the consensus committee."""
import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import conditional_donor_errors as local
from exaone_validation_data import compact_features, CATEGORICAL
from export_majority_specialist import MARGIN, TEACHER, OUT as V116, probabilities
from kaggle_workflow import sha256

ART, ROOT = local.ART, local.ROOT
OUT = ART / "nov21_donor"


def design(model, support, query, raw, other, neighbors):
    basic = local.train_scores(raw, other, model.predict_proba(query, thread_count=2)[:, 1])
    blocks = [local.response(model, support, query, raw, cols, ids)
              for cols, ids in zip(local.BLOCKS, neighbors, strict=True)]
    return np.column_stack([basic, *blocks])


def run():
    train = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c != "Name")
    test = pd.read_csv(ROOT / "data/raw/test.csv", usecols=lambda c: c != "Name")
    roster = pd.concat([train.drop(columns="Transported"), test], ignore_index=True)
    ids = test.PassengerId.to_numpy(str)
    x = compact_features(roster)
    group = roster.PassengerId.str.split("_").str[0].to_numpy()
    test_idx = np.arange(len(train), len(roster))
    y = train.Transported.to_numpy(bool)
    assert (len(train), len(test)) == (8693, 4277)
    raw_path = TEACHER / "raw_probabilities.npz"
    assert sha256(raw_path) == json.loads((TEACHER/"manifest.json").read_text())["raw_component"]["files"][raw_path.name]
    with np.load(raw_path) as raw:
        np.testing.assert_array_equal(raw["train_passenger_ids"], train.PassengerId)
        raw_oof, fold = raw["oof_raw"], raw["fold_ids"]
    anchor = probabilities(ART/"proposal_v108_party_bills/test_probabilities.csv", ids)
    witness = probabilities(ART/"proposal_nov16_kumo/test_probabilities.csv", ids)
    kumo_hashes = json.loads((V116/"manifest.json").read_text())["kumo_packets"]
    OUT.mkdir(exist_ok=True)
    members = []
    records = []
    for j in range(3):
        kpath = V116/f"kumo_packet{j}.npz"
        assert sha256(kpath) == kumo_hashes[j]
        with np.load(MARGIN/f"packet{j}.npz") as packet, np.load(kpath) as kumo:
            t = packet["T_ids"]
            np.testing.assert_array_equal(kumo["train"], t)
            np.testing.assert_array_equal(kumo["query"], np.r_[packet["D_ids"],packet["C_ids"]])
            order = np.argsort(kumo["query"])
            c = kumo["query"][order]
            other = np.r_[kumo["p"][order], witness]
        np.testing.assert_array_equal(c,np.flatnonzero(fold == j))
        np.testing.assert_array_equal(t,np.flatnonzero(fold != j))
        q = np.r_[c, test_idx]
        raw = np.r_[raw_oof[c], anchor]
        support, query = x.iloc[t], x.iloc[q]
        seed = 20261119 + j
        model = CatBoostClassifier(iterations=300, depth=5, learning_rate=.065, l2_leaf_reg=6,
            cat_features=CATEGORICAL, thread_count=2, random_seed=seed, verbose=False, allow_writing_files=False)
        model.fit(support, y[t])
        rng = np.random.default_rng(seed)
        neighbors = [local.donor_ids(support, query, group[t], group[q], block, rng)[0]
                     for block in local.BLOCKS]
        assert all(np.all(group[t][matched] != group[q,None]) for matched in neighbors)
        features = design(model, support, query, raw, other, neighbors)
        eligible = (raw[:len(c)] >= .5) == (other[:len(c)] >= .5)
        errors = (raw[:len(c)] >= .5) != y[c]
        gate = make_pipeline(StandardScaler(), LogisticRegression(C=.15, max_iter=1000))
        gate.fit(features[:len(c)][eligible], errors[eligible])
        risk = gate.predict_proba(features[len(c):])[:,1]
        pack_path = OUT/f"packet{j}.joblib"
        joblib.dump(dict(model=model, gate=gate, test_neighbors=[n[len(c):] for n in neighbors], support=t),pack_path)
        back = joblib.load(pack_path)
        again = design(back["model"], x.iloc[back["support"]], x.iloc[test_idx], anchor, witness, back["test_neighbors"])
        np.testing.assert_array_equal(risk, back["gate"].predict_proba(again)[:,1])
        members.append(risk)
        records.append(dict(packet=j, support=len(t), calibration=int(eligible.sum()), errors=int(errors[eligible].sum()),
                            model_sha256=sha256(pack_path)))
        print(records[-1], flush=True)
    members = np.column_stack(members)
    mean = members.mean(1)
    assert np.isfinite(mean).all() and ((mean >= 0) & (mean <= 1)).all()
    path = OUT/"components.npz"
    np.savez_compressed(path, passenger_ids=ids, donor=mean, member_risk=members)
    with np.load(path) as z:
        np.testing.assert_array_equal(z["passenger_ids"],ids)
        np.testing.assert_array_equal(z["donor"],z["member_risk"].mean(1))
    local.ref.write(ROOT/"reports/full_nov21_donor/result.json", dict(rows=len(ids), packets=records, reload_max_diff=0,
        components_sha256=sha256(path), raw_oof_sha256=sha256(raw_path), kumo_packets=kumo_hashes))


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
