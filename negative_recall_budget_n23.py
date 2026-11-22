"""Rank all baseline negatives in each held-out booking fold at fixed promotion budgets."""
import numpy as np
import pandas as pd

from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART, FAMILIES, ROOT, SEEDS, read_panel
from relational_error_corrector import write


MODELS = {
    **FAMILIES,
    "budget_prior": ("simplex_likelihood", "budget_prior"),
    "conditional_composition": ("simplex_likelihood", "full"),
    "class_analogues": ("class_analogues", "class_linear"),
    "gibbs_kernel": ("gibbs_manifest", "adaptive"),
    "household_joint": ("exact_booking_field", "joint"),
    "conditional_view": ("conditional_view_density", "conditional"),
    "spend_exceptions": ("spend_exceptions", "two_experts"),
    "paired_integrity": ("pair_pretext", "integrity"),
}
FRACTIONS = (.01, .025, .05, .10, .20)


def top(rows, score, count):
    # Fixed development index breaks exact ties identically for every score.
    order = np.lexsort((rows, -score[rows]))
    return rows[order[:count]]


def fixed_budget(y, fold, base, alternative, fraction, masks):
    picked = np.zeros(len(y), dtype=bool)
    by_base = picked.copy()
    by_matched_base = picked.copy()
    expected = dict.fromkeys(masks, 0.)
    expected['positive'] = 0.
    fold_counts = []
    cells = np.floor(base / .025).astype(int)
    for i in np.unique(fold):
        pool = np.flatnonzero((fold == i) & (base < .5))
        k = max(1, int(np.floor(len(pool) * fraction + .5)))
        take = top(pool, alternative, k)
        take_base = top(pool, base, k)
        picked[take] = True
        by_base[take_base] = True
        fold_counts.append(k)
        for band in np.unique(cells[take]):
            block = pool[cells[pool] == band]
            need = int((cells[take] == band).sum())
            by_matched_base[top(block, base, need)] = True
            expected['positive'] += need * y[block].mean()
            for name, mask in masks.items():
                expected[name] += need * mask[block].mean()
    assert picked.sum() == by_base.sum() == by_matched_base.sum() == sum(fold_counts)
    ev = lambda m: int((m & y).sum())
    positives = ev(picked)
    return picked, by_base, dict(
        k_folds=fold_counts, promoted=int(picked.sum()), tp=positives,
        fp=int(picked.sum())-positives, baseline_top_tp=ev(by_base),
        probability_matched_random_tp=round(expected.pop('positive'), 3),
        probability_matched_baseline_top_tp=ev(by_matched_base),
        incremental_true_vs_base=ev(picked & ~by_base), displaced_true_vs_base=ev(by_base & ~picked),
        subgroups={name:dict(tp=int((picked & mask).sum()), baseline_top_tp=int((by_base & mask).sum()),
                             additional_vs_baseline_top=int((picked & ~by_base & mask).sum()),
                             probability_matched_random_tp=round(expected[name],3),
                             probability_matched_baseline_top_tp=int((by_matched_base & mask).sum()))
                   for name, mask in masks.items()})


def run():
    panel = pd.read_csv(ART / "nov23_global_error_inventory.csv")
    row = panel.row.to_numpy()
    y, old_hits, history = read_panel()
    assert len(row)==6954 and len(np.unique(row))==len(row)
    np.testing.assert_array_equal(y, panel.target)
    booking = pd.read_csv(ROOT / "data/raw/train.csv", usecols=["PassengerId"]).iloc[row].PassengerId.str.split('_').str[0].to_numpy()
    bases, folds = [], []
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold, base = z['fold'], z['base']
            verify(z,row,y,fold)
            assert pd.DataFrame({'group':booking, 'fold':fold}).groupby('group').fold.nunique().max()==1
            bases.append(base)
            folds.append(fold)
    all_wrong = ~old_hits.any(axis=0)
    unresolved = all_wrong & ~history.any(axis=(0,1))
    high = np.max(bases,axis=0) <= .20
    masks = {"persistent_fn":y & all_wrong, "unresolved_fn":y & unresolved,
             "high_confidence_persistent_fn":y & high & all_wrong, "high_confidence_unresolved_fn":y & high & unresolved}
    assert all_wrong.sum()==1070 and unresolved.sum()==466
    reference = []
    for base in bases:
        negative = base < .5
        reference.append(dict(all_negatives=int(negative.sum()), false_negatives=int((negative & y).sum())))
    details, repeated = {}, {}
    for family, (folder, variant) in MODELS.items():
        details[family], repeated[family] = {}, {}
        selections = {frac:[] for frac in FRACTIONS}
        for seed, base, fold in zip(SEEDS, bases, folds):
            with np.load(ART/f"nov23_{folder}/{seed}.npz") as data:
                verify(data,row,y,fold)
                if 'base' in data:
                    np.testing.assert_array_equal(data['base'],base)
                challenger = data[variant]
                assert challenger.shape==y.shape and np.isfinite(challenger).all()
                assert np.all((0<=challenger)&(challenger<=1))
            curves = {}
            for frac in FRACTIONS:
                picked, base_top, summaries = fixed_budget(y, fold, base, challenger, frac, masks)
                selections[frac].append((picked, base_top))
                curves[str(frac)] = summaries
            details[family][str(seed)] = curves
        for frac, selectors in selections.items():
            picked, base_top = (np.stack(t) for t in zip(*selectors))
            repeated[family][str(frac)] = {
                n:dict(all_three=int((picked.all(axis=0) & mask).sum()),
                       any_split=int((picked.any(axis=0) & mask).sum()),
                       all_three_and_outside_every_baseline_top=int((picked.all(axis=0) & ~base_top.any(axis=0) & mask).sum()))
                for n, mask in masks.items()}
    write(ROOT/'reports/nov23_negative_recall_budget/result.json',dict(
        development=len(row),seeds=SEEDS,denominators_by_seed=reference,
        persistent_false_negatives={n:int(m.sum()) for n,m in masks.items()},fractions=FRACTIONS,
        model_sources=MODELS,definition="Every fold promotes floor(pool*fraction+0.5) baseline negatives. Controls use highest baseline probability overall, uniform expected positives in identical fold and 0.025-wide probability bins, and the highest baseline scores within those same bins. High confidence means baseline probability at most .20 in all three assignments. The same people recur across assignments. Existing scores are evaluated without tuning.",curves=details,repeatability=repeated))


if __name__ == '__main__':
    run()
