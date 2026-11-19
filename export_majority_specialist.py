"""Full-data export of the evaluated five-family majority and consensus specialist."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
NAME = "proposal_v116_majority_specialist"
OUT = ART / NAME
MARGIN = ART / "full_challenger_margin"
TEACHER = ART / "proposal_ready_nonlinear_residual_v2"


def kumo_packets():
    import kumo_adapter as kumo
    import iterate
    train = pd.read_csv(ROOT / "data/raw/train.csv")
    x, _ = iterate.build_features(train.drop(columns=["Transported", "Name"]), False)
    y = train.Transported.to_numpy(int)
    torch = kumo.backend.torch
    assert torch.backends.mps.is_available()
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.65)
    OUT.mkdir(exist_ok=True)
    for j in range(3):
        with np.load(MARGIN / f"packet{j}.npz") as z:
            t, q = z["T_ids"], np.r_[z["D_ids"], z["C_ids"]]
        np.testing.assert_array_equal(np.sort(np.r_[t, q]), np.arange(len(y)))
        model = kumo.fit_context(x.iloc[t], y[t], t, seed=20261112+j)
        p = model.predict_proba(x.iloc[q])[:, 1]
        np.savez_compressed(OUT / f"kumo_packet{j}.npz", train=t, query=q, p=p)
        model.release()
        del model
        print("Kumo calibration packet", j, "complete", flush=True)


def probabilities(path, ids):
    frame = pd.read_csv(path, dtype={"PassengerId": str}, float_precision="round_trip")
    np.testing.assert_array_equal(ids, frame.PassengerId)
    p = frame.probability.to_numpy()
    np.testing.assert_array_equal(p.astype("float32"), frame.reload_probability.to_numpy("float32"))
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    return p


def ballot(anchor, parent, votes, veto):
    agree = np.sum(votes == anchor[:, None], axis=1)
    return parent ^ ((parent == anchor) & ((agree < 3) | ((agree == 5) & veto))), agree


def validate_development():
    from error_overlap_diagnostic import load_authenticated
    from route_consensus_errors import FAMILIES
    inp, _, fold, models, _ = load_authenticated()
    with np.load(ART / "v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"], inp["dev"])
        np.testing.assert_array_equal(z["fold_id"], fold)
        anchor = z["v108"] >= .5
    with np.load(ART / "composed_local_error_routes.npz") as z:
        np.testing.assert_array_equal(z["dev"], inp["dev"])
        parent, expected = z["base"], z["five_family_majority_with_specialist"]
    with np.load(ART / "consensus_dispute_specialist.npz") as z:
        np.testing.assert_array_equal(z["dev"], inp["dev"])
        veto = z["agreement_trained_five_agree"] != parent
    pred, _ = ballot(anchor, parent, np.column_stack([models[k] >= .5 for k in FAMILIES]), veto)
    np.testing.assert_array_equal(pred, expected)
    assert int((pred == inp["y"]).sum()) == 5711


def decide(anchor, parent, votes, witness, members, gates):
    from consensus_dispute_specialist import disagreement_features
    direction = anchor >= .5
    scores = []
    for j, gate in enumerate(gates):
        obj = members[:, j]
        features = disagreement_features(anchor, witness, obj)
        contested = np.any((obj >= .5) != direction[:, None], axis=1)
        scores.append(gate.predict_proba(features)[:, 1] * contested)
    risk = np.mean(scores, axis=0)
    pred, n = ballot(direction, parent, votes, risk >= .5)
    return pred, risk, n


def export():
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from consensus_dispute_specialist import disagreement_features
    import export_consensus_routes as routes
    import full_challenger_margin as margin
    from kaggle_workflow import validate_submission, sha256
    from relational_error_corrector import write

    validate_development()

    train = pd.read_csv(ROOT / "data/raw/train.csv")
    test = pd.read_csv(ROOT / "data/raw/test.csv")
    ids = test.PassengerId.to_numpy(str)
    y = train.Transported.to_numpy(bool)
    x, _ = margin.iterate.build_features(pd.concat([train, test], ignore_index=True).drop(columns=["Transported","Name"]), False)
    model_path = MARGIN / "model.joblib"
    dependency = json.loads((MARGIN / "manifest.json").read_text())
    assert sha256(model_path) == dependency["model_sha256"]
    assert sha256(TEACHER / "raw_probabilities.npz") == json.loads((TEACHER / "manifest.json").read_text())["raw_component"]["files"]["raw_probabilities.npz"]
    full = joblib.load(model_path)
    with np.load(TEACHER / "raw_probabilities.npz") as z:
        np.testing.assert_array_equal(z["train_passenger_ids"], train.PassengerId)
        raw, folds = z["oof_raw"], z["fold_ids"]
    hist = np.column_stack([routes.official_decisions(k, ids)[0] for k in routes.HISTORICAL])
    parent, parent_hash = routes.official_decisions(routes.NAME, ids)
    anchor, _ = margin.load_reference(ids)
    witness = probabilities(ART / routes.HISTORICAL[3] / "test_probabilities.csv", ids)
    np.testing.assert_array_equal(witness >= .5, hist[:, 3])
    gates, training = [], []
    for j, pack in enumerate(full.packets):
        with np.load(MARGIN / f"packet{j}.npz") as z, np.load(OUT / f"kumo_packet{j}.npz") as k:
            q, c = np.r_[z["D_ids"], z["C_ids"]], z["C_ids"]
            np.testing.assert_array_equal(np.flatnonzero(folds == j), np.sort(q))
            np.testing.assert_array_equal(z["T_ids"], k["train"])
            np.testing.assert_array_equal(k["query"], q)
            np.testing.assert_array_equal(z["C_teacher"], raw[c])
            other = k["p"]
            models = pack["challengers"]
            obj = np.column_stack([models["forest"].predict_proba(x.loc[q])[:, 1],
                models["latent"].predict(models["latent_encoder"].transform(x.loc[q, margin.original.latent.COLS]))])
            np.testing.assert_allclose(obj[-len(c):], z["C_challengers"], rtol=0, atol=1e-12)
        prob = raw[q]
        active = ((prob >= .5) == (other >= .5)) & np.any((obj >= .5) != (prob >= .5)[:, None], axis=1)
        err = (prob >= .5) != y[q]
        gate = make_pipeline(StandardScaler(), LogisticRegression(C=.15, max_iter=1000))
        gate.fit(disagreement_features(prob, other, obj)[active], err[active])
        gates.append(gate)
        training.append(dict(calibration_rows=int(active.sum()), errors=int(err[active].sum())))
    path = OUT / "gates.joblib"
    joblib.dump(gates, path)
    assert sha256(MARGIN / "test_components.npz") == dependency["test_components_sha256"]
    with np.load(MARGIN / "test_components.npz") as z:
        np.testing.assert_array_equal(z["passenger_ids"], ids)
        np.testing.assert_array_equal(z["v108"], anchor)
        result, risks, n = decide(anchor, parent, hist, witness, z["members"], gates)
        replay, again, _ = decide(anchor, parent, hist, witness, z["reload_members"], joblib.load(path))
    np.testing.assert_array_equal(risks, again)
    np.testing.assert_array_equal(result, replay)
    changed = result != parent
    pd.DataFrame(dict(PassengerId=ids, Transported=result)).to_csv(OUT / "submission.csv", index=False)
    # The export is a hard voting policy. Encode labels directly for the release verifier.
    pd.DataFrame(dict(PassengerId=ids, probability=result.astype(float),
                      reload_probability=replay.astype(float))).to_csv(OUT / "test_probabilities.csv", index=False)
    manifest = dict(**validate_submission(OUT / "submission.csv", ROOT / "data/raw/test.csv"),
        version="v116_majority_consensus_specialist", model_sha256=sha256(path), reload_difference=0,
        parent_hash=parent_hash, historical=routes.HISTORICAL, calibration=training, parent_changes=int(changed.sum()),
        majority_changes=int((changed & (n<3)).sum()), consensus_changes=int((changed & (n==5)).sum()),
        vs_v108_changes=int((result != (anchor >= .5)).sum()), full_margin_model_sha256=dependency["model_sha256"],
        scores="Binary labels, not calibrated probabilities.", kumo_packets=[sha256(OUT / f"kumo_packet{j}.npz") for j in range(3)])
    for folder in [OUT, ROOT / "reports" / NAME]:
        write(folder / "manifest.json", manifest)
        write(folder / "selection.json", dict(version=manifest["version"], threshold=.5))
    print(manifest, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["kumo", "export"])
    stage = parser.parse_args().stage
    (kumo_packets if stage == "kumo" else export)()
