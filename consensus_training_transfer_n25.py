"""Follow the same 68 archive outputs on the same passengers after changing the validation split."""
import json
from pathlib import Path

import numpy as np

from full_roster_conditional_pool_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'


def run():
    raw, y, groups, full_fold = data()
    with np.load(ART / 'nov25_historical_nine/historical.npz') as z:
        rows, fold, nine = [z[k] for k in ('rows', 'fold', 'nine')]
        np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[rows])
    alt_names, panels = prediction_sets(raw, y, groups, full_fold, rows, fold)
    original = np.column_stack(panels[1]) >= .5
    rerun = np.column_stack(panels[0])[rows] >= .5
    assert original.shape == rerun.shape == (len(rows), 68)
    # The first six columns are the same two archived model families in both protocols.
    names = [f'base_panel_{i+1}' for i in range(6)] + alt_names
    assert len(names) == original.shape[1]
    all_agree = (original == nine[:, None]).all(1)
    wrong = all_agree & (nine != y[rows])
    right = all_agree & (nine == y[rows])
    assert wrong.sum() == 208 and right.sum() == 2676
    shifts = rerun != nine[:, None]
    count = shifts.sum(1)
    seen = np.zeros(len(rows), bool)
    for f in np.unique(fold):
        packet = json.loads((ART / 'v108_exact_oof' / f'fold{f}_packet.json').read_text())
        np.testing.assert_array_equal(rows[fold == f], packet['query_ids'])
        seen[fold == f] = np.isin(groups[rows[fold == f]], groups[packet['train_ids']])
    def transfer(mask):
        n = count[mask]
        return dict(rows=len(n), still_identical=int((n == 0).sum()), any_changed=int((n > 0).sum()),
                    at_least_seven=int((n >= 7).sum()), strict_majority=int((n > 34).sum()),
                    all_changed=int((n == 68).sum()), max_changed=int(n.max(initial=0)))
    result = {}
    for name, scope in [('all', np.ones(len(rows), bool)), ('original_training_saw_order', seen),
                        ('original_training_excluded_order', ~seen)]:
        result[name] = dict(original_errors=transfer(wrong & scope), original_correct=transfer(right & scope))
    per_output = {name:dict(original_errors_now_correct=int(shifts[wrong, j].sum()),
                            original_correct_now_wrong=int(shifts[right, j].sum()))
                  for j, name in enumerate(names)}
    report = dict(archive_outputs=68, original_passengers=len(rows), cohorts=result, outputs=per_output,
                  note='Paired retrospective on historical unanimous cases. Model training size and partition change together.')
    (ROOT / 'reports/nov23_matched_learning_curve/consensus_training_transfer.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print(result, flush=True)


if __name__ == '__main__':
    run()
