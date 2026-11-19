"""Calibrate against the same frozen model's real mistakes on other held-out parties."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import relational_error_corrector as ref
from route_consensus_errors import FAMILIES
from stable_consensus_selector import make_context, persistent_mask

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def cached(file, key, dev, folds):
    with np.load(ART / file) as z:
        np.testing.assert_array_equal(z["dev"] if "dev" in z else z["dev_ids"], dev)
        fold_key = next(k for k in ("fold", "folds", "fold_id") if k in z)
        np.testing.assert_array_equal(z[fold_key], folds)
        return z[key]


def counts(pred, anchor, y, fold, hard, persistent):
    fixed, harmed = (anchor != y) & (pred == y), (anchor == y) & (pred != y)
    return dict(accuracy=float((pred == y).mean()), fixed=int(fixed.sum()), harmed=int(harmed.sum()),
                fold_net=[int(fixed[fold == k].sum()) - int(harmed[fold == k].sum()) for k in range(3)],
                hard_fixed=int((hard & fixed).sum()), hard_harmed=int((hard & harmed).sum()),
                persistent_fixed=int((persistent & fixed).sum()), persistent_harmed=int((persistent & harmed).sum()))


def run():
    inp, _, folds, bases, _ = ref.overlap.load_authenticated()
    dev, y, groups = inp["dev"], inp["y"].astype(bool), np.asarray(inp["groups"])
    p = cached("v108_exact_oof/oof.npz", "v108", dev, folds)
    pos = p >= .5
    sign = 1 - 2 * pos.astype(float)
    hist = np.column_stack([bases[k] for k in FAMILIES])
    eligible = ((hist >= .5) == pos[:, None]).all(axis=1)
    cascade = cached("sequential_rescue_local.npz", "member", dev, folds).mean(axis=1)
    challenge = cached("error_evidence_challengers/oof.npz", "members", dev, folds).mean(axis=1)
    donor = cached("conditional_donor_errors.npz", "matched_donors_risk", dev, folds).mean(axis=1)
    prototype = cached("prototype_error_specialist.npz", "four_centers_risk", dev, folds)
    simple = np.column_stack([pos, np.abs(ref.logit(p)), ref.logit(hist) * sign[:, None]])
    evidence = np.column_stack([simple, sign[:, None] * ref.logit(cascade), sign[:, None] * ref.logit(challenge),
                                ref.logit(donor), ref.logit(prototype)])
    views = {"confidence": simple, "expert_evidence": evidence,
             "expert_and_manifest": np.column_stack([evidence, make_context(inp)])}
    hard = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(hard.row_id, dev)
    np.testing.assert_array_equal(hard.y, y)
    hard = hard.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp, bases, hard)
    calibration, results, arrays = [], {}, {}
    for seed in (119, 220, 321):
        risks = {}
        for outer in range(3):
            ids = np.flatnonzero(eligible & (folds == outer))
            # All covariates for these rows came from one frozen outer training set.
            # Only split inside that model's held-out rows. Crossing original folds
            # here would expose a calibration predictor trained on the assessment labels.
            parts = GroupKFold(3, shuffle=True, random_state=seed+outer)
            for block, (t, q) in enumerate(parts.split(ids, groups=groups[ids])):
                t, q = ids[t], ids[q]
                assert not np.intersect1d(groups[t], groups[q]).size
                for label, wrong in [("v108", pos != y), ("proxy", (bases["rawMitra"] >= .5) != y)]:
                    for view, x in views.items():
                        key = label + "_" + view
                        if key not in risks:
                            risks[key] = np.zeros(len(y))
                        model = make_pipeline(StandardScaler(), LogisticRegression(C=.15, max_iter=1000))
                        model.fit(x[t], wrong[t])
                        risks[key][q] = model.predict_proba(x[q])[:, 1]
                calibration.append(dict(seed=seed, outer=outer, block=block, train=len(t), query=len(q)))
        for name, risk in risks.items():
            key = f"{name}_{seed}"
            pred = pos ^ (eligible & (risk >= .5))
            results[key] = counts(pred, pos, y, folds, hard, persistent)
            arrays[key] = pred
    np.savez_compressed(ART / "exact_fold_error_selector.npz", dev=dev, fold=folds, anchor=pos, **arrays)
    ref.write(ROOT / "reports/exact_fold_error_selector/result.json", dict(rows=len(y), eligible=int(eligible.sum()),
        reference_label_disagreements=int(np.sum(eligible & (pos != (bases["rawMitra"] >= .5)))),
        design="Within each original teacher's held-out fold, partition eligible parties into three blocks. Fit the gate "
               "on two blocks, score on the third. Keep predictors identical in the v108 versus raw-teacher target ablation. "
               "Three repeated partitions of the same labels; no tuning on the repeats. The full-data baseline would need "
               "OOF-to-full transfer or a separately labeled calibration sample.",
        calibration=calibration, results=results))
    print({k: (v["fixed"], v["harmed"], v["hard_fixed"], v["hard_harmed"]) for k, v in results.items()})


if __name__ == "__main__":
    run()
