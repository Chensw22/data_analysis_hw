"""Test monotone combinations of independent challenges on an exact frozen reference."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from exact_fold_error_selector import ART, ROOT, cached, counts
from route_consensus_errors import FAMILIES
from stable_consensus_selector import persistent_mask
import relational_error_corrector as ref


def run():
    inp, _, folds, bases, _ = ref.overlap.load_authenticated()
    dev, groups, y = inp["dev"], np.asarray(inp["groups"]), inp["y"].astype(bool)
    p = cached("v108_exact_oof/oof.npz", "v108", dev, folds)
    base = p >= .5
    sign = 1 - 2 * base.astype(float)
    hist = np.column_stack([bases[k] for k in FAMILIES])
    eligible = ((hist >= .5) == base[:, None]).all(axis=1)
    # These signals are computed using the original outer training rows only.
    # Every predictor is oriented so larger values indicate evidence of an error.
    strong = sign[:, None] * ref.logit(hist)
    heads = cached("sequential_rescue_local.npz", "member", dev, folds).mean(axis=1)
    challengers = cached("error_evidence_challengers/oof.npz", "members", dev, folds).mean(axis=1)
    signals = np.column_stack([-np.abs(ref.logit(p)), strong.mean(axis=1), strong.min(axis=1),
        sign[:, None] * ref.logit(heads[:, [0, 2]]), sign[:, None] * ref.logit(challengers),
        cached("conditional_donor_errors.npz", "matched_donors_risk", dev, folds).mean(axis=1),
        cached("prototype_error_specialist.npz", "uncalibrated_four_risk", dev, folds),
        cached("directional_error_selection.npz", "separate_risk", dev, folds), base])
    monotone = [1] * (signals.shape[1] - 1) + [0]
    wrong = base != y
    catalog = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id, dev)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp, bases, hard)
    results, preds = {}, {}
    for mode, constraints in [("unconstrained", None), ("monotone", monotone)]:
        repeat = []
        for seed in (119, 220, 321):
            risk = np.zeros(len(y))
            for outer in range(3):
                ids = np.flatnonzero((folds == outer) & eligible)
                for t, q in GroupKFold(3, shuffle=True, random_state=seed+outer).split(ids, groups=groups[ids]):
                    t, q = ids[t], ids[q]
                    assert not np.intersect1d(groups[t], groups[q]).size
                    model = HistGradientBoostingClassifier(monotonic_cst=constraints, max_iter=100, max_leaf_nodes=5,
                        min_samples_leaf=32, l2_regularization=10., learning_rate=.06, early_stopping=False, random_state=seed)
                    model.fit(signals[t], wrong[t])
                    risk[q] = model.predict_proba(signals[q])[:, 1]
            pred = base ^ (risk >= .5)
            results[f"{mode}_{seed}"] = counts(pred, base, y, folds, hard, persistent)
            repeat.append(risk)
        decision = base ^ (np.mean(repeat, axis=0) >= .5)
        results[mode + "_mean"] = counts(decision, base, y, folds, hard, persistent)
        preds[mode + "_mean"] = decision
    np.savez_compressed(ART / "monotone_error_confirmation.npz", dev=dev, fold=folds, base=base, **preds)
    ref.write(ROOT / "reports/monotone_error_confirmation/result.json", dict(results=results,
        scope="Only where five references support v108. All evidence scores are original held-out predictions. "
              "Fit error selectors within the original held-out fold on other parties, fixing the teacher. "
              "Compare monotone and unconstrained trees with identical hyperparameters, including repeats."))
    print(results)


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
