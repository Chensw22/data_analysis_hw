"""Measure overlap in solo repairs from disjoint training thirds and cross-origin corrections."""
import json

import numpy as np
from threadpoolctl import threadpool_limits

from distinct_challenger_error_reach_n26 import population
from full_roster_booking_validation_n23 import ROOT
from tabnet_attention_transfer_n26 import RESULT, collect
from target_mix_expert_routing_n26 import ticker


def run():
    raw, y, _, f, reference, reference_common, reach, _, _, _, solo, _ = population()
    with np.load(ROOT / 'artifacts/nov26_replicated_solo_training_errors/scores.npz') as z:
        np.testing.assert_array_equal(z['passenger'], raw.PassengerId)
        np.testing.assert_array_equal(z['fold'], f)
        third = z['real'] >= .5
    with np.load(ROOT / 'artifacts/nov26_nested_environment_residual/scores.npz') as z:
        np.testing.assert_array_equal(z['ids'], raw.PassengerId)
        np.testing.assert_array_equal(z['y'], y)
        np.testing.assert_array_equal(z['fold'], f)
        regional = z['departure_maximin'].mean(0) >= .5
        shuffled = z['shuffled_maximin'].mean(0) >= .5
    report = {}
    for tag, items in collect()[0].items():
        ids, common, parent = (np.concatenate([b[i] for b in items]) for i in range(3))
        eligible = common & solo[ids] & ~np.concatenate([b[3]['actual_and_scores'] for b in items])
        wrong = y[ids] != parent
        stable = (third[ids] != parent[:, None]).all(1)
        region = regional[ids] != parent
        control = shuffled[ids] != parent
        choices = dict(both=stable & region, only_disjoint=stable & ~region, only_region=region & ~stable,
                       neither=~stable & ~region, with_shuffled=stable & control, real_only=stable & region & ~control)
        report[tag] = {}
        no_dissent = reference_common[ids] & (reference[ids] == parent) & (reach[ids] == 0)
        for name, mask in [('solo_common', eligible), ('without_old_dissent', eligible & no_dissent)]:
            cells = {}
            for label, action in choices.items():
                table = ticker(mask, wrong, action)
                delta = mask * action * np.where(wrong, 1, -1)
                table['net_by_fold'] = [int(delta[f[ids] == k].sum()) for k in range(8)]
                cells[label] = table
            for count in ('fixed', 'harmed'):
                total = sum(cells[c][count] for c in ('both', 'only_disjoint', 'only_region', 'neither'))
                assert total == int((mask & (wrong if count == 'fixed' else ~wrong)).sum())
            report[tag][name] = cells
        print(tag, report[tag]['solo_common']['both'], flush=True)
    (RESULT.parent / 'agreement_of_solo_corrections.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
