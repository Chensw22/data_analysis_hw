"""Two tests of how far the confirmed Earth sleeper correction transfers."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import export_consensus_routes as routes
import relational_error_corrector as ref
from confirm_sleeper_scope import NAME as INCUMBENT, confirm
from exact_fold_error_selector import counts
from exaone_validation_data import compact_features
from sleeper_consensus_scope import ART, NAME as SCOPE, ROOT, cohort
from sleeper_manifest_risk import design
from stable_consensus_selector import persistent_mask

FOLDER = "nov21_sleep_checks"


def decisions(base, scoped, simple, berth, eligible, relative):
    incumbent = confirm(base, scoped, berth)
    # v129 asks if cabin position is needed to adjudicate the original challengers.
    # v130 adds cases without challenger support only if both outcome models agree.
    return incumbent, {"v129": confirm(base, scoped, simple),
                       "both_without_challenger": incumbent & ~(eligible & ~simple & ~berth),
                       "v130": confirm(base, scoped, relative)}


def relative_risk(frame, y, groups):
    matrix = design(frame, True)
    ix = np.flatnonzero(cohort(frame))
    parties = groups[ix]
    order, group = np.unique(parties.astype(int), return_inverse=True)
    block = np.minimum(8, 9 * group // len(order))
    splits = {key: list(GroupKFold(3, shuffle=True, random_state=seed).split(ix, groups=parties))
              for key, seed in [("party_a", 1121), ("party_b", 2121)]}
    splits["boarding_ranges"] = [(np.flatnonzero((block-i)%9 >= 3), np.flatnonzero((block-i)%9 < 3))
                                 for i in [0,3,6]]
    answers = {}
    for key, pairs in splits.items():
        relative = np.ones(len(y), dtype=bool)
        for t, q in pairs:
            t, q = ix[t], ix[q]
            assert not np.intersect1d(groups[t], groups[q]).size
            model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000))
            model.fit(matrix[t], y[t])
            relative[q] = model.predict_proba(matrix[q])[:,1] >= y[t].mean()
        answers[key] = relative
    return answers


def evaluate():
    inp, _, folds, models, _ = ref.overlap.load_authenticated()
    rows = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(rows.row_id, inp["dev"])
    y = inp["y"].astype(bool)
    np.testing.assert_array_equal(rows.y, y)
    hard = rows.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp, models, hard)
    results = {}
    relative = relative_risk(inp["features"].iloc[inp["dev"]], y, np.asarray(inp["groups"]))
    with np.load(ART / "sleeper_consensus_scope.npz") as scope, np.load(ART / "sleeper_manifest_risk.npz") as risk:
        for archive in [scope, risk]:
            np.testing.assert_array_equal(archive["dev"], inp["dev"])
            np.testing.assert_array_equal(archive["fold"], folds)
        np.testing.assert_array_equal(scope["base"], risk["base"])
        for split, check in [("original", "party_a"), ("new_party_partitions", "party_b"),
                             ("boarding_blocks", "boarding_ranges")]:
            incumbent, versions = decisions(scope["base"], scope[split], risk["manifest_" + check],
                                            risk["manifest_berth_" + check], scope["eligible"], relative[check])
            results[split] = {k: counts(v, incumbent, y, folds, hard, persistent) for k,v in versions.items()}
    ref.write(ROOT / "reports" / FOLDER / "local.json", results)
    print(results["original"])


def export():
    train, test = [pd.read_csv(ROOT / f"data/raw/{p}.csv", usecols=lambda c: c != "Name") for p in ["train","test"]]
    x, q = compact_features(train), compact_features(test)
    ids = test.PassengerId.to_numpy(str)
    base, _ = routes.official_decisions(routes.BASE, ids)
    scoped, _ = routes.official_decisions(SCOPE, ids)
    historical = np.column_stack([routes.official_decisions(n, ids)[0] for n in routes.HISTORICAL])
    eligible = base & historical.all(axis=1) & cohort(q)
    domain = cohort(x)
    predictions = []
    for expanded in [False, True]:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000))
        model.fit(design(x, expanded)[domain], train.Transported.to_numpy(bool)[domain])
        predictions.append(model.predict(design(q, expanded)))
    relative = model.predict_proba(design(q, True))[:,1] >= train.Transported.to_numpy(bool)[domain].mean()
    incumbent, versions = decisions(base, scoped, *predictions, eligible, relative)
    np.testing.assert_array_equal(incumbent, routes.official_decisions(INCUMBENT, ids)[0])
    out = ART / FOLDER
    out.mkdir(exist_ok=True)
    for k, v in versions.items():
        if k == "both_without_challenger":
            print(k, "changed vs incumbent:", np.count_nonzero(v != incumbent))
            continue
        pd.DataFrame(dict(PassengerId=ids, Transported=v)).to_csv(out / (k + ".csv"), index=False)
        print(k, "changed vs incumbent:", np.count_nonzero(v != incumbent))


if __name__ == "__main__":
    evaluate()
    export()
