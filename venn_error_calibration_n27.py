"""Estimate error rates for every passenger from held-order probability calibration."""
import json
import numpy as np
from sklearn.isotonic import isotonic_regression

from dissent_likelihood_transfer_n27 import splits
from full_roster_booking_validation_n23 import ROOT, data
from repeated_archive_misses_n26 import ART, cohort, count

OUT = ART / 'nov27_venn_errors'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/venn_error_calibration.json'
BINS = 48


def interval(train_bin, errors):
    """Append one hypothetical outcome in each bin and fit both monotone regressions."""
    size = np.bincount(train_bin, minlength=BINS)
    hits = np.bincount(train_bin, weights=errors, minlength=BINS)
    table = np.empty((BINS, 2))
    for k in range(BINS):
        weight = size.copy()
        weight[k] += 1
        keep = weight > 0
        where = np.count_nonzero(keep[:k])
        for label in (0, 1):
            totals = hits.copy()
            totals[k] += label
            fit = isotonic_regression(totals[keep] / weight[keep], sample_weight=weight[keep])
            table[k, label] = fit[where]
    assert np.all(table[:, 0] <= table[:, 1] + 1e-12)
    return table


def run():
    raw, y, groups, fold = data()
    _, blind, known, fresh = cohort(raw, y, fold)
    n = len(y)
    with np.load(ART / 'nov27_dissent_likelihood_transfer/aligned.npz') as anchor:
        ids, first = anchor['passenger'], anchor['first']
        selected = {s: anchor[s + '_median'] for s in ('random', 'contiguous')}
        np.testing.assert_array_equal(anchor['y'], y)
        np.testing.assert_array_equal(anchor['fold'], fold)
    with np.load(ART / 'nov26_latent_passenger_difficulty/panel.npz') as panel:
        np.testing.assert_array_equal(ids, panel['passenger'])
        scores = panel['scores']
        print(panel['names'])
    risk = np.where(first[:, None], 1 - scores, scores)
    source = np.column_stack([np.median(risk[:, :18], axis=1), risk[:, -3:], np.median(risk[:, -3:], axis=1)])
    bins = np.minimum((source * BINS).astype(int), BINS - 1)
    wrong = y != first[:n]
    result = {}
    actions = {}
    for scheme, parent_edits in selected.items():
        split = splits(groups, fold, scheme)
        estimates = np.full((len(ids), source.shape[1], 2), np.nan)
        test_estimates = []
        for k in range(8):
            for j in (0, 1, 2, 3, None):
                train = np.flatnonzero(known & (fold == k) & ((split != j) if j is not None else True))
                query = np.arange(n, len(ids)) if j is None else np.flatnonzero((fold == k) & (split == j))
                if j is not None:
                    assert not np.isin(groups[train], groups[query]).any()
                batch = np.empty((len(query), source.shape[1], 2))
                for side in (False, True):
                    tr = train[first[train] == side]
                    q = first[query] == side
                    for channel in range(source.shape[1]):
                        table = interval(bins[tr, channel], wrong[tr])
                        batch[q, channel] = table[bins[query[q], channel]]
                if j is None:
                    test_estimates.append(batch)
                else:
                    estimates[query] = batch
        estimates[n:] = np.mean(test_estimates, axis=0)
        assert np.isfinite(estimates).all()
        lower, upper = estimates.transpose(2, 0, 1)
        center = upper / (1 - lower + upper)
        rules = {
            'foundation_center': center[:, -1] > .5,
            'foundation_lower': lower[:, -1] > .5,
            'two_foundations_lower': (lower[:, 1:4] > .5).sum(axis=1) >= 2,
            'independent_lower': (lower[:, 0] > .5) & (lower[:, -1] > .5),
        }
        current = first ^ parent_edits
        for key, proposed in rules.items():
            for use in ('additional', 'replacement'):
                edit = (~parent_edits & proposed) if use == 'additional' else (parent_edits ^ proposed)
                name = key + '_' + use
                actions[scheme + '_' + name] = edit
                err = current[:n] != y
                delta = np.where(err, 1, -1) * edit[:n]
                result.setdefault(name, {})[scheme] = {
                    'all': count(edit[:n], known, err), 'fresh': count(edit[:n], fresh & known, err),
                    'prior_unanimous': count(edit[:n], blind, err),
                    'fold_delta': [int(delta[known & (fold == k)].sum()) for k in range(8)],
                    'test_changes': int(edit[n:].sum()),
                }
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / 'aligned.npz', passenger=ids, y=y, fold=fold, **actions)
    REPORT.write_text(json.dumps(result, indent=2) + '\n')
    for key, values in result.items():
        print(key, [(v['all'], v['fresh'], v['test_changes']) for v in values.values()])


if __name__ == '__main__':
    run()
