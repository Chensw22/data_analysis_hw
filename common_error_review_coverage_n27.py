"""Measure reachable common mistakes at a label-free review budget within each population."""
import json

import numpy as np
from scipy.stats import rankdata

from full_roster_booking_validation_n23 import ROOT
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART, cohort, count
from unanimous_error_reach_n27 import inputs


def evaluate():
    raw, frame, y, _, folds, first, directional, _, scores, alternatives = inputs()
    n = len(y)
    _, _, known, fresh = cohort(raw, y, folds)
    anchor = first ^ directional
    wrong = anchor[:n] != y
    common = ((scores >= 0.5) == anchor[:, None]).all(axis=1)
    paid = frame[FEES].gt(0).any(axis=1).to_numpy()
    populations = {
        'awake_paid': frame.CryoSleep.eq(False).to_numpy() & paid & common,
        'sleep_zero': frame.CryoSleep.eq(True).to_numpy() & ~paid & common,
    }
    # Distance of the least confident ordinary vote from a tie. Ties share a rank.
    margin = np.min(np.abs(scores - 0.5), axis=1)
    full_known = np.r_[known, np.zeros(len(frame) - n, dtype=bool)]
    partition = np.r_[folds, np.full(len(frame) - n, 8)]
    outside_vote = (alternatives['nearest'] >= 0.5) != anchor
    report = {}
    with np.load(ART / 'nov27_manifest_activity_transitions/aligned.npz') as cache:
        np.testing.assert_array_equal(cache['passenger'], frame.PassengerId.astype(str))
        np.testing.assert_array_equal(cache['fold'], folds)
        np.testing.assert_array_equal(cache['y'], y)
        for name, population in populations.items():
            eligible = population & (full_known | (partition == 8))
            percentile = np.full(len(frame), np.inf)
            for fold in range(9):
                rows = np.flatnonzero(eligible & (partition == fold))
                percentile[rows] = rankdata(margin[rows]) / len(rows)
            counts = {}
            for share in (0.10, 0.25, 0.50, 1.0):
                reviewed = eligible & (percentile <= share)
                entry = dict(reviewed=int(reviewed[:n].sum()), errors=int((reviewed[:n] & wrong).sum()),
                             test_reviewed=int(reviewed[n:].sum()), protocols={})
                for scheme in ('random', 'ordered'):
                    challenger = (cache[scheme + '_asymmetric'] >= 0.5) != anchor
                    control = (cache[scheme + '_additive'] >= 0.5) != anchor
                    rules = {'nearest': outside_vote, 'transition': challenger,
                             'both': outside_vote & challenger, 'nearest_and_additive': outside_vote & control}
                    entry['protocols'][scheme] = {}
                    for rule, mask in rules.items():
                        edited = mask & reviewed
                        entry['protocols'][scheme][rule] = dict(
                            all=count(edited[:n], known, wrong), fresh=count(edited[:n], known & fresh, wrong),
                            test=int(edited[n:].sum()))
                counts[str(share)] = entry
            report[name] = counts
    assert report['awake_paid']['1.0']['errors'] == 476
    assert report['sleep_zero']['1.0']['errors'] == 300
    output = ROOT / 'reports/nov23_matched_learning_curve/common_error_review_coverage.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    for name, results in report.items():
        for share, entry in results.items():
            print(name, share, entry['reviewed'], entry['errors'],
                  {scheme: rules['both']['all'] for scheme, rules in entry['protocols'].items()})


if __name__ == '__main__':
    evaluate()
