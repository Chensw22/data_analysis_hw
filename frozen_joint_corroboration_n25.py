"""Compare agreeing corrections from frozen binary, joint outcome, and nonlinear token readouts."""
import json
from pathlib import Path

import numpy as np

from full_roster_conditional_pool_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'artifacts/nov25_mitra_four_outcomes'
TREES = ROOT / 'artifacts/nov25_mitra_token_trees/predictions.npz'


def decisions(scores, tree, native, previous, nine):
    eligible = ((native >= .5) == previous) & (previous == nine)
    binary, marginal, conditional = (scores[:, j] >= .5 for j in range(3))
    readout = tree >= .5
    binary_edit = eligible & (binary != nine)
    joint_edit = eligible & (conditional != nine) & (conditional == marginal)
    tree_edit = eligible & (readout != nine)
    return {
        'binary_confirmed': binary_edit & (joint_edit | tree_edit),
        'joint_and_tree': joint_edit & tree_edit,
        'binary_plus_joint_and_tree': binary_edit | (joint_edit & tree_edit),
        'two_of_three': (binary_edit & joint_edit) | (binary_edit & tree_edit) | (joint_edit & tree_edit),
    }


def run():
    raw, truth, groups, full_fold = data()
    with np.load(ROOT / 'artifacts/nov25_historical_nine/historical.npz') as z:
        rows, folds, nine, previous = [z[k] for k in ('rows', 'fold', 'nine', 'v131')]
    with np.load(TREES) as z:
        np.testing.assert_array_equal(z['rows'], rows)
        np.testing.assert_array_equal(z['fold'], folds)
        np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[rows])
        tree = z['historical'][:, z['model_names'].tolist().index('joint')]
    scores = np.empty((len(rows), 4))
    native = np.empty(len(rows))
    for fold in np.unique(folds):
        ix = folds == fold
        with np.load(CACHE / f'fold{fold}.npz') as z:
            np.testing.assert_array_equal(z['q'], rows[ix])
            scores[ix] = z['scores']
            native[ix] = z['native']
    _, panels = prediction_sets(raw, truth, groups, full_fold, rows, folds)
    agree = ((np.column_stack(panels[1]) >= .5) == nine[:, None]).all(1)
    correct = nine == truth[rows]
    base_edit = ((native >= .5) == previous) & (previous == nine) & ((scores[:, 0] >= .5) != nine)
    baseline_correct = np.logical_xor(correct, base_edit)
    result = {}
    for name, edit in decisions(scores, tree, native, previous, nine).items():
        outcome = np.logical_xor(correct, edit)
        net = outcome.astype(int) - correct
        delta = outcome.astype(int) - baseline_correct
        result[name] = dict(
            corrected=int((net > 0).sum()), harmed=int((net < 0).sum()),
            fold_net=[int(net[folds == f].sum()) for f in np.unique(folds)],
            net_vs_frozen_by_fold=[int(delta[folds == f].sum()) for f in np.unique(folds)],
            consensus_fixed=int((agree & (net > 0)).sum()), consensus_harmed=int((agree & (net < 0)).sum()),
            new_fixes_over_frozen=int(((delta > 0) & ~correct).sum()),
            new_harms_over_frozen=int(((delta < 0) & correct).sum()))
    (ROOT / 'reports/nov23_matched_learning_curve/frozen_joint_corroboration.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print(result)


if __name__ == '__main__':
    run()
