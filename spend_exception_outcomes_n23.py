"""Test equivalent outcome codings and separate models across the whole manifest."""
import numpy as np
import pandas as pd

from boarding_sequence_n23 import ART, ROOT, SEEDS
from exaone_validation_data import compact_features
from grouped_five_way_coverage_n23 import verify
from nested_manifest_expressions_n23 import BILLS, fit
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write


def run():
    rows = pd.read_csv(ART / "nov23_global_error_inventory.csv").row.to_numpy()
    y, base_hits, challengers = read_panel()
    persistent = ~base_hits.any(axis=0)
    unresolved = persistent & ~challengers.any(axis=(0, 1))
    raw = pd.concat([pd.read_csv(ROOT / f"data/raw/{source}.csv",
                                 usecols=lambda c: c not in {"Name", "Transported"})
                     for source in ["train", "test"]], ignore_index=True)
    groups = raw.PassengerId.str.split("_").str[0]
    x = compact_features(raw).assign(PartySize=groups.map(groups.value_counts())).iloc[rows].reset_index(drop=True)
    usual = ~raw.iloc[rows][BILLS].gt(0).any(axis=1).to_numpy()
    groups = groups.iloc[rows].to_numpy()
    x["no_positive_bill"] = usual.astype(int)
    exceptions = y != usual
    out = ART / "nov23_spend_exceptions"
    out.mkdir(exist_ok=True)
    results = {}
    for seed in SEEDS:
        with np.load(ART / f"nov23_group_kernel/{seed}.npz") as cache:
            fold = cache["fold"]
            base = cache["base"]
            verify(cache, rows, y, fold)
        predictions = {name: np.full(len(y), np.nan) for name in ["direct", "exception", "two_experts"]}
        for k in np.unique(fold):
            t = np.flatnonzero(fold != k)
            q = np.flatnonzero(fold == k)
            assert not np.intersect1d(groups[t], groups[q]).size
            predictions["direct"][q] = fit(x, y, t, 155).predict_proba(x.iloc[q])[:, 1]
            p = fit(x, exceptions, t, 155).predict_proba(x.iloc[q])[:, 1]
            predictions["exception"][q] = np.where(usual[q], 1-p, p)
            for flag in [True, False]:
                a, b = t[usual[t] == flag], q[usual[q] == flag]
                predictions["two_experts"][b] = fit(x, y, a, 155).predict_proba(x.iloc[b])[:, 1]
        old = (base >= .5) == y
        metrics = {}
        for name, score in predictions.items():
            assert np.isfinite(score).all()
            for label, prob in [(name, score), (name+"_blend", .75*base+.25*score)]:
                correct = (prob >= .5) == y
                metrics[label] = dict(accuracy=float(correct.mean()), fixed=int((correct & ~old).sum()),
                                      harmed=int((~correct & old).sum()), persistent1070=int(correct[persistent].sum()),
                                      unresolved466=int(correct[unresolved].sum()),
                                      unresolved_false_negative=int((correct & y & unresolved).sum()),
                                      unresolved_false_positive=int((correct & ~y & unresolved).sum()))
        results[seed] = metrics
        np.savez_compressed(out / f"{seed}.npz", row=rows, fold=fold, y=y, base=base, **predictions)
        print(seed, metrics, flush=True)
    write(ROOT / "reports/nov23_spend_exceptions/result.json", dict(results=results,
          definition="Usual outcome = no positive observed bill. Includes partially missing bills. Identical inputs for direct and recoded targets; two specialists partition every passenger. Diagnostic masks never enter fitting."))


if __name__ == "__main__":
    run()
