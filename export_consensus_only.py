"""Evaluate and export the consensus branch alone on the public-best v108."""
from pathlib import Path

import numpy as np
import pandas as pd

import export_consensus_routes as routes
from export_majority_specialist import NAME as V116
from kaggle_workflow import sha256, validate_submission
from relational_error_corrector import write

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
NAME = "proposal_v117_consensus_only"


def run():
    catalog = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    with np.load(ART / "v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"], catalog.row_id)
        np.testing.assert_array_equal(z["fold_id"], catalog.fold)
        base = z["v108"] >= .5
    with np.load(ART / "consensus_error_routes.npz") as z:
        np.testing.assert_array_equal(z["dev"], catalog.row_id)
        old = z["exception_in_historical_consensus"]
        parent = base ^ z["combined_by_historical_agreement"]
    with np.load(ART / "consensus_dispute_specialist.npz") as z:
        np.testing.assert_array_equal(z["dev"], catalog.row_id)
        np.testing.assert_array_equal(z["base"], parent)
        extra = z["agreement_trained_five_agree"] != parent
    assert not np.any(old & extra)
    flip = old | extra
    y = catalog.y.to_numpy(bool)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    repair, harm = flip & (base != y), flip & (base == y)
    history = np.column_stack([np.where(catalog["correct_" + k], y, ~y)
                               for k in ["v44", "v81", "TabDPT", "Kumo", "EXAONE"]])
    agree = (history == base[:, None]).all(axis=1)
    with np.load(ART / "composed_local_error_routes.npz") as z:
        np.testing.assert_array_equal(z["dev"], catalog.row_id)
        np.testing.assert_array_equal(base ^ flip, np.where(agree, z["five_family_majority_with_specialist"], base))
    local = dict(rows=len(y), correct=int(((base ^ flip) == y).sum()), accuracy=float(((base ^ flip) == y).mean()),
                 repaired=int(repair.sum()), harmed=int(harm.sum()), hard_repaired=int((hard & repair).sum()),
                 hard_harmed=int((hard & harm).sum()), fold_net=[int(repair[catalog.fold == f].sum()) -
                 int(harm[catalog.fold == f].sum()) for f in range(3)])
    ids = pd.read_csv(ROOT / "data/raw/test.csv", usecols=["PassengerId"]).PassengerId.to_numpy()
    sources = {}
    def get_decision(name):
        values, sources[name] = routes.official_decisions(name, ids)
        return values
    anchor = get_decision(routes.BASE)
    child = get_decision(V116)
    votes = np.column_stack([get_decision(k) for k in routes.HISTORICAL])
    take = (votes == anchor[:, None]).all(axis=1)
    decision = np.where(take, child, anchor)
    out = ART / NAME
    report = ROOT / "reports" / NAME
    out.mkdir(exist_ok=True)
    spec = dict(anchor=routes.BASE, consensus_candidate=V116, families=routes.HISTORICAL, source_csv_sha256=sources)
    write(out / "router.json", spec)
    import json
    spec = json.loads((out / "router.json").read_text())
    def reload(name):
        assert sha256(ART / name / "submission.csv") == spec["source_csv_sha256"][name]
        return get_decision(name)
    original = reload(spec["anchor"])
    same = np.logical_and.reduce([reload(k) == original for k in spec["families"]])
    replay = np.where(same, reload(spec["consensus_candidate"]), original)
    np.testing.assert_array_equal(decision, replay)
    pd.DataFrame(dict(PassengerId=ids, Transported=decision)).to_csv(out / "submission.csv", index=False)
    pd.DataFrame(dict(PassengerId=ids, probability=decision.astype(float),
                      reload_probability=replay.astype(float))).to_csv(out / "test_probabilities.csv", index=False)
    manifest = dict(**validate_submission(out / "submission.csv", ROOT / "data/raw/test.csv"),
                    version="v117_v108_consensus_only", model_sha256=sha256(out / "router.json"), reload_difference=0,
                    changed_vs_v108=int((decision != anchor).sum()), changes_when_divided=int(((decision != anchor) & ~take).sum()),
                    local=local, selection_context="Isolate both consensus specialists following public evaluation of v116.")
    for folder in [out, report]:
        write(folder / "manifest.json", manifest)
        write(folder / "selection.json", dict(version=manifest["version"], threshold=.5))
    print(manifest)


if __name__ == "__main__":
    run()
