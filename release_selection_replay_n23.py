"""Use whole-booking splits to measure selection effects conditional on archived predictions."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import historical, write
from forward_release_transfer_n23 import REPORT
from released_correction_audit_n23 import development


def main():
    row, y, _, groups, _ = load()
    labels, votes, _ = development()
    np.testing.assert_array_equal(y, labels)
    # Check passenger alignment, not just label order.
    old_rows, _, _ = historical()
    np.testing.assert_array_equal(row, old_rows)
    early = [108, *range(114, 129)]
    late = list(range(129, 137))
    versions = early+late
    hits = np.column_stack([votes[v] == y for v in versions])
    code, names = pd.factorize(groups)
    sizes = np.bincount(code)
    correct = np.zeros((len(names), len(versions)), dtype=int)
    np.add.at(correct, code, hits)
    rng = np.random.default_rng(976)
    draws = 1000
    results = {k: [] for k in ['select_both', 'fixed_incumbent', 'fixed_challenger', 'both_fixed']}
    selected = np.zeros(len(versions), int)
    for _ in range(draws):
        # Assign complete bookings to local selection, mock public selection and independent scoring.
        blocks = np.array_split(rng.permutation(len(names)), 3)
        scores = np.array([100*correct[b].sum(0)/sizes[b].sum() for b in blocks])
        incumbent = np.flatnonzero(scores[1, :len(early)] == scores[1, :len(early)].max())
        challenger = np.flatnonzero(scores[0, len(early):] == scores[0, len(early):].max()) + len(early)
        a, b = rng.choice(incumbent), rng.choice(challenger)
        selected[[a, b]] += 1
        # v128 and v129 are fixed comparators for this diagnostic.
        for k, i, j in [('select_both', a, b), ('fixed_incumbent', early.index(128), b),
                        ('fixed_challenger', a, len(early)), ('both_fixed', early.index(128), len(early))]:
            results[k].append(scores[:, j]-scores[:, i])
    out = {}
    for name, values in results.items():
        changes = np.asarray(values)
        out[name] = dict(mean_pp=changes.mean(0).tolist(), median_pp=np.median(changes, axis=0).tolist(),
                         positive_fraction=(changes>0).mean(0).tolist(),
                         local_minus_public_pp=float((changes[:,0]-changes[:,1]).mean()),
                         public_minus_independent_pp=float((changes[:,1]-changes[:,2]).mean()))
    write(REPORT/'selection.json', dict(draws=draws, bookings=len(names), passengers=len(y), early=early, late=late,
                                      columns=['local_selection', 'mock_public', 'independent_scoring'],
                                      selected=dict(zip(map(str, versions), selected.tolist())), results=out))
    print({k:v['mean_pp'] for k,v in out.items()})


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
