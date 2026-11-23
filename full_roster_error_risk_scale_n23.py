"""Calibrate large-training expert margins to smaller-training reference distributions."""
import json

import numpy as np
from scipy.stats import rankdata

from full_roster_agreement_error_risk_n23 import agreement_band, risk_scores
from full_roster_unanimous_error_risk_n23 import ROOT, banks, data, inputs, latest_reference


def matched_margins(half, full, vote, fold, conditional):
    """Use paired predictions on each unlabeled query fold, never its outcomes."""
    sign = np.where(vote, 1, -1)[:, None]
    source = (full - .5) * sign
    target = (half - .5) * sign
    calibrated = source.copy()
    for f in np.unique(fold):
        for direction in (False, True) if conditional else (None,):
            rows = fold == f
            if direction is not None:
                rows &= vote == direction
            # Midranks retain ties and order within the unlabeled query population.
            ranks = rankdata(source[rows], axis=0, method='average') - 1
            for j in range(source.shape[1]):
                calibrated[rows, j] = np.interp(
                    ranks[:, j], np.arange(rows.sum()), np.sort(target[rows, j])
                )
    return calibrated


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    train, train_vote, count = inputs(raw, half)
    query, vote, query_count = inputs(raw, full)
    bands = agreement_band(query_count)
    reference = latest_reference(y, group, fold, names, half, full)
    settings = json.loads((ROOT / 'reports/nov23_matched_learning_curve/nested_error_scope.json').read_text())
    cuts = np.array([row['cutoff'] for row in settings['inner']])
    assert [row['outer_fold'] for row in settings['inner']] == list(range(8))
    final = settings['deployment_scope']['cutoff']
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    report = {}
    margins = [f'ExpertMargin{j}' for j in range(len(names))]
    for mode in ['unmodified', 'rank', 'rank_by_direction', 'smaller_training_scores']:
        feature = query.copy()
        if mode == 'smaller_training_scores':
            feature[margins] = (half - .5) * np.where(vote, 1, -1)[:, None]
        elif mode != 'unmodified':
            feature[margins] = matched_margins(half, full, vote, fold, mode == 'rank_by_direction')
        probability = risk_scores(train, feature, train_vote, count, query_count, y, group, fold, 'nearest')
        for label, allowed in [('all', bands >= 0), ('deployment', bands >= final), ('nested', bands >= cuts[fold])]:
            pick = allowed & (probability >= .5) & (reference == vote)
            delta = np.where(pick, np.where(reference == y, -1, 1), 0)
            report[f'{mode}_{label}'] = dict(
                accuracy=float(((reference == y).sum() + delta.sum())/len(y)),
                fixed=int((delta == 1).sum()), harmed=int((delta == -1).sum()),
                additional_net=int(delta.sum()-delta[dev].sum()),
                fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)],
            )
    baseline = report['unmodified_nested']
    for k in ['accuracy','fixed','harmed','fold_net','additional_net']:
        assert baseline[k] == settings['results']['learn_cutoff'][k]
    (ROOT / 'reports/nov23_matched_learning_curve/error_risk_scale.json').write_text(
        json.dumps(report, indent=2) + '\n')
    for label, row in report.items():
        print(label, round(100*row['accuracy'],3), row['fixed'], row['harmed'], row['additional_net'])


if __name__ == '__main__':
    run()
