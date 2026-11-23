"""Assess the passenger-pair candidate with an independent zero-spend positive override."""
import json

import numpy as np

from audit_donor_direction_n23 import REPORT, counts, ranked_flips
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_service_pairs_n23 import REPORT as PAIR_REPORT
from service_pair_change_budget_n23 import changed


def additional(base, anchor_vote, pair_vote, matched, row, fraction):
    candidates = np.flatnonzero(np.isfinite(matched) & ~anchor_vote)
    num = int(np.floor(fraction * len(candidates)))
    score = .75 * base + .25 * matched
    choices = ranked_flips(candidates, score, row, num, len(row))
    return choices & (pair_vote == anchor_vote)


def main():
    row, y, _, _, _ = load()
    weight = json.loads((PAIR_REPORT / 'test_budget.json').read_text())['test']['change_150']['weight']
    result = {}
    for seed in [*map(str, SEEDS), 'v131']:
        with np.load(ART/f'nov23_zero_spend_analogues/{seed}.npz') as donors, \
             np.load(ART/f'nov23_service_pairs/{seed}.npz') as pairs:
            for key in ('row', 'y', 'ref', 'base', 'fold'):
                np.testing.assert_array_equal(donors[key], pairs[key])
            base, ref = donors['base'], donors['ref']
            np.testing.assert_array_equal(row, donors['row'])
            np.testing.assert_array_equal(y, donors['y'])
            pair = changed(pairs['same_passenger_both'], base, ref, weight)
            result[seed] = {}
            for fraction in (.05, .10):
                by_method = {}
                for key in ('matched','random'):
                    extra = additional(base, ref, pair, donors[key], row, fraction)
                    combined = pair | extra
                    by_method[key] = dict(increment=counts(extra, pair != y),
                                          combined=counts(combined != ref, ref != y))
                result[seed][str(fraction)] = by_method
    write(REPORT/'overlay.json', result)
    print(result)


if __name__ == '__main__':
    main()
