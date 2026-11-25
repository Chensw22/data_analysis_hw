"""Measure attainable versus realizable corrections with four distinct out-of-fold challengers."""
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from audit_repeated_pair_cases_n25 import near_scores, TAG
from full_roster_conditional_pool_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/shared_error_family_coverage.json'


def record(edit, wrong, scope):
    return dict(fixed=int((scope & edit & wrong).sum()), harmed=int((scope & edit & ~wrong).sum()))


def signature_gate(votes, error, native, anchor, agreement, groups, folds):
    # Each component is frozen outside the current outer fold. Learn edit utility
    # on separate bookings within that fold, with a coarse-to-fine Beta prior.
    signature = votes.astype(int) @ (1 << np.arange(votes.shape[1]))
    direction = np.where(anchor, native, 1 - native)
    cell = anchor.astype(int) + 2 * agreement + 4 * np.digitize(direction, [.6, .8])
    gate = np.zeros(len(votes), bool)
    probability = np.zeros(len(votes))
    for f in np.unique(folds):
        ix = np.flatnonzero(folds == f)
        for t0, q0 in GroupKFold(4, shuffle=True, random_state=691).split(ix, groups=groups[ix]):
            t, q = ix[t0], ix[q0]
            assert not np.intersect1d(groups[t], groups[q]).size
            # Learn the baseline odds of error in each confidence/direction cell.
            n = np.bincount(cell[t], minlength=12)
            k = np.bincount(cell[t], weights=error[t], minlength=12)
            prior = (k + 8 * error[t].mean()) / (n + 8)
            # Now let each set of dissenters supply evidence above its baseline cell.
            code = cell * (1 << votes.shape[1]) + signature
            size = 12 * (1 << votes.shape[1])
            n = np.bincount(code[t], minlength=size)
            k = np.bincount(code[t], weights=error[t], minlength=size)
            probability[q] = (k[code[q]] + 8 * prior[cell[q]]) / (n[code[q]] + 8)
            gate[q] = (signature[q] != 0) & (probability[q] > .5)
    return gate


def run():
    raw, y, group, full_fold = data()
    with np.load(ART / 'nov25_historical_nine/historical.npz') as z:
        rows, fold, base, prev = [z[k] for k in ('rows', 'fold', 'nine', 'v131')]
    _, panels = prediction_sets(raw, y, group, full_fold, rows, fold)
    agree = ((np.column_stack(panels[1]) >= .5) == base[:, None]).all(1)
    with np.load(ART / 'nov25_mitra_fp32_head/oof.npz') as z:
        np.testing.assert_array_equal(z['row'], rows)
        native = z['native']
    free = ((native >= .5) == prev) & (prev == base)
    first, second = [near_scores((tag, 'historical'), rows, fold, group, y) for tag in (TAG, TAG + '_repeat')]
    probs = np.empty((len(rows), 2))
    shared_odds = np.empty(len(rows))
    for f in np.unique(fold):
        use = fold == f
        with np.load(ART / 'nov25_full_context_generations' / f'historical_{f}.npz') as z:
            np.testing.assert_array_equal(z['q'], rows[use])
            probs[use] = np.column_stack([z['v1'], z['v2']])
        with np.load(ART / 'nov25_shared_booking_state' / f'historical_{f}.npz') as z:
            np.testing.assert_array_equal(z['q'], rows[use])
            p = expit(z['states'].astype(float)).mean(axis=(1, 3)).clip(.001, .999)
            shared_odds[use] = logit(p[1]) - logit(p[0])
    families = {
        'paired_references': ((first >= .5) != base) & ((second >= .5) != base),
        'shared_booking_state': free & (((logit(native.clip(.001, .999)) + shared_odds) >= 0) != base),
        'frozen_v1': free & ((probs[:, 0] >= .5) != base),
        'frozen_v2': free & ((probs[:, 1] >= .5) != base),
    }
    votes = np.column_stack(list(families.values()))
    wrong = base != y[rows]
    support = votes.sum(1)
    votes_by_family = {**families, 'union': support > 0, 'two_or_more': support >= 2,
                       'three_or_more': support >= 3, 'all_four': support == 4,
                       'held_out_booking_gate': signature_gate(votes, wrong, native, base, agree, group[rows], fold)}
    out = {}
    for name, scope in [('all', np.ones(len(rows), bool)), ('archive_agrees', agree), ('archive_disagrees', ~agree)]:
        out[name] = dict(rows=int(scope.sum()), errors=int((wrong & scope).sum()),
                         policies={k: record(v, wrong, scope) for k, v in votes_by_family.items()},
                         unique_fixed={k: int((wrong & scope & v & (support == 1)).sum()) for k, v in families.items()})
    target = int(np.floor(.85 * len(rows))) + 1
    correct = int((~wrong).sum())
    report = dict(rows=len(rows), baseline_correct=correct, required_correct_for_above_85=target,
                  minimum_net_gain=target-correct, correct_with_all_unanimous_errors_repaired=correct+int((wrong & agree).sum()),
                  minimum_further_gain_even_with_perfect_unanimous_repair=target-correct-int((wrong & agree).sum()),
                  results=out)
    assert out['archive_agrees']['errors'] == 208
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(out['archive_agrees'], out['all']['policies']['held_out_booking_gate'])


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
