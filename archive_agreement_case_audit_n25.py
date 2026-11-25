"""Locate corrections and newly introduced mistakes across the full historical error distribution."""
import json
from pathlib import Path

import numpy as np

from full_roster_conditional_pool_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from frozen_joint_corroboration_n25 import decisions

ROOT = Path(__file__).resolve().parent


def run():
    raw, y, orders, f = data()
    with np.load(ROOT / 'artifacts/nov25_historical_nine/historical.npz') as z:
        rows, fold, v131, base = [z[k] for k in ('rows', 'fold', 'v131', 'nine')]
    _, panels = prediction_sets(raw, y, orders, f, rows, fold)
    votes = np.column_stack(panels[1]) >= .5
    disagree = (votes != base[:, None]).sum(axis=1)
    n_votes = votes.shape[1]
    with np.load(ROOT / 'artifacts/nov25_mitra_token_trees/predictions.npz') as z:
        np.testing.assert_array_equal(z['rows'], rows)
        tree = z['historical'][:, z['model_names'].tolist().index('joint')]
    native = np.empty(len(rows))
    scores = np.empty((len(rows), 4))
    matched = np.empty(len(rows))
    for k in np.unique(fold):
        use = fold == k
        for study in ('four_outcomes', 'stratified_support'):
            path = ROOT / 'artifacts' / ('nov25_mitra_' + study) / f'fold{k}.npz'
            with np.load(path) as z:
                np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[rows[use]])
                np.testing.assert_array_equal(z['q'], rows[use])
                if study == 'four_outcomes':
                    scores[use], native[use] = z['scores'], z['native']
                else:
                    matched[use] = z['scores'][:, 0]
    free = ((native >= .5) == v131) & (v131 == base)
    edits = decisions(scores, tree, native, v131, base)
    edits['full_binary'] = free & ((scores[:, 0] >= .5) != base)
    edits['full_plus_matched'] = free & ((.5 * (scores[:, 0] + matched) >= .5) != base)
    was_wrong = base != y[rows]
    bins = {
        'none': disagree == 0,
        'up_to_one_tenth': (disagree > 0) & (10 * disagree <= n_votes),
        'up_to_one_half': (10 * disagree > n_votes) & (2 * disagree <= n_votes),
        'majority': 2 * disagree > n_votes,
    }
    np.testing.assert_array_equal(np.stack(list(bins.values())).sum(0), np.ones(len(rows)))
    result = {}
    for label, cases in bins.items():
        counts = {}
        for name, edit in edits.items():
            counts[name] = dict(corrected=int((cases & edit & was_wrong).sum()),
                                harmed=int((cases & edit & ~was_wrong).sum()))
        result[label] = dict(rows=int(cases.sum()), original_errors=int((cases & was_wrong).sum()), methods=counts)
    path = ROOT / 'reports/nov23_matched_learning_curve/archive_agreement_case_audit.json'
    path.write_text(json.dumps(dict(archive_outputs=n_votes, groups=result), indent=2) + '\n')
    print(n_votes, [(k, v['rows'], v['original_errors'], v['methods']['full_plus_matched']) for k, v in result.items()])


if __name__ == '__main__':
    run()
