"""Identify stable rescues and damage unique to the fee-distribution models."""
import numpy as np

from broad_bill_neighbor_decision_n23 import opinions
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_service_distribution_distance_n23 import LABEL, REPORT
from broad_tree_field_dividends_n23 import hard_cohorts

PRIOR_MODELS = {
    'abstaining_fields': ['personal_first', 'bill_first', 'averaged_order'],
    'dependence_anchors': ['independent', 'single', 'double'],
    'opposing_passengers': ['matched_conditional'],
    'service_bag': ['pooled_logit', 'fixed', 'learned'],
    'service_bag_trees': ['initial', 'mixture'],
    'subspace_discriminants': ['diagonal_average', 'qda_average'],
    'known_manifest_codes': ['shared_binary', 'code_mean', 'code_conditional', 'code_marginal',
                             'ordinary_conditional', 'ordinary_marginal'],
}


def cached(seed, name, keys, expected):
    with np.load(ART / ('nov23_' + name) / (seed + '.npz')) as data:
        for key, array in expected.items():
            np.testing.assert_array_equal(data[key], array)
        return np.column_stack([data[key] for key in keys])


def main():
    row, y, frame, _, _ = load()
    _, _, hard, correct = hard_cohorts(row, y)
    latest = ['budget', 'unordered_fees', 'sliced_identity', 'aligned_services']
    baseline, candidate = [], []
    for seed in map(str, SEEDS):
        fold, _, ref, bill, near = opinions(seed, row, y, frame.BookingSize.to_numpy() > 1)
        expected = dict(row=row, y=y, fold=fold, ref=ref)
        previous = [bill, near]
        for source, keys in PRIOR_MODELS.items():
            previous.append(cached(seed, source, keys, expected))
        baseline.append((np.column_stack(previous) >= .5) == y[:, None])
        candidate.append((cached(seed, LABEL, latest, expected) >= .5) == y[:, None])
    seen_correct = np.all(baseline, axis=0).any(1)
    seen_damaged = (~np.any(baseline, axis=0)).any(1)
    assert (seen_correct & hard).sum() == 77
    fixed = np.all(candidate, axis=0)
    damaged = ~np.any(candidate, axis=0)
    result = {}
    for name, fixed_rows, damaged_rows in [*zip(latest, fixed.T, damaged.T),
                                            ('union', fixed.any(1), damaged.any(1))]:
        result[name] = dict(hard_stable=int((fixed_rows & hard).sum()),
                            newly_covered=int((fixed_rows & hard & ~seen_correct).sum()),
                            earlier_stable_wrong=int((damaged_rows & correct).sum()),
                            newly_damaged=int((damaged_rows & correct & ~seen_damaged).sum()))
    fresh = np.flatnonzero(fixed.any(1) & hard & ~seen_correct)
    result['new_passengers'] = [dict(train_row=int(row[i]), transported=bool(y[i]),
                                    features=frame.iloc[i].where(frame.iloc[i].notna(), None).to_dict(),
                                    experts=[name for name, hit in zip(latest, fixed[i]) if hit])
                                for i in fresh]
    write(REPORT / 'rescue_overlap.json', result)
    print(result)


if __name__ == '__main__':
    main()
