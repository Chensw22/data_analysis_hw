"""Transfer previously evaluated correction rules to the Kaggle test roster."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import export_consensus_routes as route
import relational_error_corrector as ref
from exact_fold_error_selector import cached
from exaone_validation_data import compact_features
from kaggle_workflow import sha256, validate_submission
from replicate_error_committee import partitions
from route_consensus_errors import FAMILIES
from stable_consensus_selector import make_context

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
COMMON = ART / "nov21_gate_ensemble"
NAMES = ["v118_error_union", "v119_error_committee", "v120_error_tree", "v121_soft_vote",
         "v122_second_confirms", "v123_matched_donor", "v124_first_head",
         "v125_sequential_heads", "v126_parallel_heads"]


def design(p, hist, voters, critics, donor, proto, geometric, directional, context):
    base = p >= .5
    sign = 1 - 2 * base.astype(float)
    hs, cs = ref.logit(hist) * sign[:, None], ref.logit(critics) * sign[:, None]
    lr = np.column_stack([base, np.abs(ref.logit(p)), hs, ref.logit(voters) * sign[:, None],
                          cs, ref.logit(donor), ref.logit(proto), context])
    tree = np.column_stack([-np.abs(ref.logit(p)), hs.mean(1), hs.min(1),
        ref.logit(voters[:, [0, 2]]) * sign[:, None], cs, donor, geometric, directional, base])
    assert np.isfinite(lr).all() and np.isfinite(tree).all()
    return lr, tree


def fit_gates():
    inp, _, fold, models, _ = ref.overlap.load_authenticated()
    dev, y = inp["dev"], inp["y"].astype(bool)
    def get(file, key):
        return cached(file + ".npz", key, dev, fold)
    p = get("v108_exact_oof/oof", "v108")
    hist = np.column_stack([models[k] for k in FAMILIES])
    voters = get("sequential_rescue_local", "member").mean(1)
    geometric = get("prototype_error_specialist", "uncalibrated_four_risk")
    lr, tree = design(p, hist, voters, get("error_evidence_challengers/oof", "members").mean(1),
        get("conditional_donor_errors", "matched_donors_risk").mean(1), get("prototype_error_specialist", "four_centers_risk"),
        geometric, get("directional_error_selection", "separate_risk"), make_context(inp))
    base = p >= .5
    eligible = ((hist >= .5) == base[:, None]).all(1)
    lp, tp = np.zeros((len(y), 3)), np.zeros((len(y), 3))
    pack = []
    for outer in range(3):
        ids = np.flatnonzero(eligible & (fold == outer))
        for slot, seed, t, q in partitions(ids, inp["groups"], "original", outer):
            classifier = make_pipeline(StandardScaler(), LogisticRegression(C=.15, max_iter=1000))
            classifier.fit(lr[t], base[t] != y[t])
            nonlinear = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=5, min_samples_leaf=32,
                    l2_regularization=10., learning_rate=.06, early_stopping=False, random_state=seed)
            nonlinear.fit(tree[t], base[t] != y[t])
            lp[q, slot] = classifier.predict_proba(lr[q])[:, 1]
            tp[q, slot] = nonlinear.predict_proba(tree[q])[:, 1]
            pack.append((slot, classifier, nonlinear))
    com = (lp >= .5).all(1) & (geometric >= .5) & ((voters[:,2] >= .5) != base)
    tr = tp.mean(1) >= .5
    for key, flips in [("committee", com), ("tree", tr), ("either", com | tr)]:
        np.testing.assert_array_equal(base ^ flips, get("replicate_error_committee", "original_" + key))
    return pack, inp


def test_probabilities(name, ids):
    decision, _ = route.official_decisions(name, ids)
    frame = pd.read_csv(ART / name / "test_probabilities.csv", dtype={"PassengerId": str}, float_precision="round_trip")
    np.testing.assert_array_equal(frame.PassengerId, ids)
    p = frame.probability.to_numpy()
    np.testing.assert_array_equal(p >= .5, decision)
    return p


def apply(pack, inp):
    test = pd.read_csv(ROOT / "data/raw/test.csv")
    ids = test.PassengerId.to_numpy(str)
    p = test_probabilities(route.BASE, ids)
    base = p >= .5
    parent, _ = route.official_decisions(route.NAME, ids)
    previous, _ = route.official_decisions("proposal_v116_majority_specialist", ids)
    hist = np.column_stack([test_probabilities(n, ids) for n in route.HISTORICAL])
    n = ((hist >= .5) == base[:,None]).sum(1)
    protected = parent != base
    with np.load(ART / "full_challenger_margin/test_components.npz") as z:
        np.testing.assert_array_equal(z["passenger_ids"], ids)
        np.testing.assert_array_equal(z["members"], z["reload_members"])
        critics = z["members"].mean(1)
    with np.load(ART / "nov21_specialists/components.npz") as z:
        np.testing.assert_array_equal(z["passenger_ids"], ids)
        voters, proto, geo, direction = (z[k] for k in ["voters", "proto", "geometric_risk", "directional"])
        head_risk = [z[k] for k in ["first_risk", "sequential_risk", "parallel_risk"]]
    with np.load(ART / "nov21_donor/components.npz") as z:
        np.testing.assert_array_equal(z["passenger_ids"], ids)
        donor = z["donor"]
    roster = pd.concat([inp["features"].iloc[inp["dev"]], compact_features(test)], ignore_index=True)
    context = make_context(dict(features=roster, dev=np.arange(len(roster))))
    np.testing.assert_array_equal(context[:len(inp["dev"])], make_context(inp))
    lr, tree = design(p, hist, voters, critics, donor, proto, geo, direction, context[len(inp["dev"]):])
    # Each local query had one eligible held-out predictor per repeat. On new
    # passengers all nine fits per repeat are out of sample, so average their risks.
    linear, nonlinear = [], []
    for slot in range(3):
        gates = [(c, t) for s, c, t in pack if s == slot]
        assert len(gates) == 9
        linear.append(np.mean([c.predict_proba(lr)[:,1] for c, _ in gates], axis=0))
        nonlinear.append(np.mean([t.predict_proba(tree)[:,1] for _, t in gates], axis=0))
    com = (n == 5) & np.all(np.array(linear) >= .5, axis=0) & (geo >= .5) & ((voters[:,2] >= .5) != base)
    tr = (n == 5) & (np.mean(nonlinear, axis=0) >= .5)
    soft = (hist.sum(1) + p) / 6 >= .5
    answers = [base ^ (com | tr), base ^ com, base ^ tr, np.where(protected | (n == 5), previous, soft),
               np.where((n == 0) & ((voters[:,2] >= .5) != previous), parent, previous),
               base ^ ((n == 5) & (donor >= .5))]
    answers.extend(parent ^ (~protected & (r >= .5)) for r in head_risk)
    return ids, base, dict(zip(NAMES, answers, strict=True))


def run():
    COMMON.mkdir(exist_ok=True)
    pack, inp = fit_gates()
    model_path = COMMON / "gates.joblib"
    joblib.dump(pack, model_path)
    ids, base, answers = apply(pack, inp)
    ids_again, _, replay = apply(joblib.load(model_path), inp)
    np.testing.assert_array_equal(ids, ids_again)
    sources = [model_path, ART / "nov21_specialists/components.npz", ART / "nov21_donor/components.npz",
               ART / "full_challenger_margin/test_components.npz"]
    sources += [ART / n / "test_probabilities.csv" for n in [*route.HISTORICAL, route.BASE]]
    for n in [route.NAME, "proposal_v116_majority_specialist"]:
        sources.append(ART / n / "submission.csv")
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    for key, answer in answers.items():
        np.testing.assert_array_equal(answer, replay[key])
        name = "proposal_" + key
        out = ART / name
        out.mkdir(exist_ok=True)
        ref.write(out / "recipe.json", dict(version=key, components=hashes, implementation=sha256(Path(__file__))))
        pd.DataFrame(dict(PassengerId=ids, Transported=answer)).to_csv(out / "submission.csv", index=False)
        pd.DataFrame(dict(PassengerId=ids, probability=answer.astype(float),
                          reload_probability=replay[key].astype(float))).to_csv(out / "test_probabilities.csv", index=False)
        manifest = dict(**validate_submission(out / "submission.csv", ROOT / "data/raw/test.csv"),
                        version=key, model_sha256=sha256(out / "recipe.json"), reload_difference=0,
                        changed_vs_v108=int(np.sum(answer != base)), component_sources=hashes)
        for folder in [out, ROOT / "reports" / name]:
            ref.write(folder / "manifest.json", manifest)
            ref.write(folder / "selection.json", dict(version=key, threshold=.5))
        print(key, manifest["changed_vs_v108"], manifest["sha256"], flush=True)


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
