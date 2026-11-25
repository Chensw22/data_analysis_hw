"""Pair every November 25–26 queued release with the same scored incumbent and measure their joint reach."""
import json
import numpy as np
import pandas as pd

from arbitration_against_scored_nine_n26 import champions
from evaluate_nested_release_n26 import stages
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import ROOT, data
from held_expert_portfolios_n26 import DECK, REVIEW, panel_predictions, sources, splits
from joint_candidate_coverage_n26 import branch_decisions
from price_implied_deck_n26 import FEES
from six_way_against_scored_nine_n26 import test_decisions

RELEASES = {
    'nine': 'proposal_n24_selected_dissent',
    'fields': 'proposal_n25_confirmed_random_fields',
    'awake': 'proposal_n25_awake_personal_tabicl',
    'fields_awake': 'proposal_n25_confirmed_awake_union',
    'filtered': 'proposal_n25_awake_error_confirmation',
    'band': 'proposal_n26_dispute_band',
    'all_three': 'proposal_n26_joint_dispute_awake',
    'review': 'proposal_n26_consensus_review',
    'fare': 'proposal_n26_fare_deck',
    'six': 'proposal_n26_six_way_dispute',
}


def paired():
    raw, y, g, f = data()
    nine, known, _, _ = champions()
    _, _, _, _, names, scores, _, native, _, ref = panels_and_reference()
    with np.load(ROOT/'artifacts/nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        prior = z['prior']
        np.testing.assert_array_equal(f, z['fold'])
        np.testing.assert_array_equal(ref[0], z['original_prior'])
    branches = branch_decisions(np.arange(len(y)), f, scores, names, native, prior, ref[1], ref[2],
                                raw.CryoSleep.eq(False).to_numpy(), 'full', 891)
    local = {}
    for name in list(RELEASES)[:7]:
        a, b = branches[name]
        np.testing.assert_array_equal(a[known], b[known])
        local[name] = a.copy()
    np.testing.assert_array_equal(local['nine'][known], nine[known])
    panels = sources()[-1]
    full = panels['full']
    bank = panel_predictions(full, 'full')
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled = z['eligible']
    mask = np.zeros(len(y), bool)
    for name in ['review', 'fare', 'six']:
        local[name] = mask.copy()
    for key, _, dst, _, q in splits(panels, g):
        if key.startswith('full_'):
            assert not mask[q].any()
            mask[q] = True
            with np.load(REVIEW / f'pred_{key}.npz') as z:
                np.testing.assert_array_equal(z['q'], q)
                risk = np.column_stack([z['identities_error'], z['identities_outcome']])
            anchor, *rest = stages(dst, bank, q, enabled[q], risk)
            np.testing.assert_array_equal(anchor, local['all_three'][q])
            for name, stage in zip(['review','fare','six'], rest, strict=True):
                local[name][q] = stage
    np.testing.assert_array_equal(known, mask)
    assert int(((local['six'] == y).astype(int) - (nine == y))[known].sum()) == 81
    return raw, y, g, f, known, np.column_stack(list(local.values()))


def run():
    raw, y, g, f, known, matrix = paired()
    test = pd.read_csv(ROOT / 'data/raw/test.csv', usecols=['PassengerId'])
    hard = np.column_stack([test_decisions(n, test.PassengerId)[0] for n in RELEASES.values()])
    assert matrix.shape[1] == hard.shape[1] == 10
    # Union recipes operate on changes relative to the scored release in both class directions.
    for block in [hard, matrix[known]]:
        changes = block != block[:, :1]
        np.testing.assert_array_equal(changes[:,3], changes[:,1] | changes[:,2])
        np.testing.assert_array_equal(changes[:,6], changes[:,3] | changes[:,5])
        assert not (changes[:,4] & ~changes[:,2]).any()
    change = hard != hard[:, :1]
    disagreement = np.any(matrix != matrix[:, :1], axis=1)
    base = matrix[:,0]
    paid = raw[FEES].gt(0).any(axis=1).to_numpy()
    masks = dict(all=known, solo=known & (raw.groupby(g).PassengerId.transform('size').to_numpy() == 1),
                 awake_paid_miss=known & ~base & raw.CryoSleep.eq(False).to_numpy() & paid,
                 sleeper_false_alarm=known & base & raw.CryoSleep.eq(True).to_numpy() & ~paid)
    report = dict(versions=RELEASES, test_rows=len(test), test_any_changed=int(change.any(1).sum()), scopes={}, by_candidate={})
    for key, subset in masks.items():
        error = subset & (base != y)
        unanimous = subset & ~disagreement
        report['scopes'][key] = dict(rows=int(subset.sum()), incumbent_errors=int(error.sum()),
                    any_candidate_correct=int((error & disagreement).sum()),
                    unanimous_rows=int(unanimous.sum()), all_wrong=int((error & unanimous).sum()),
                    all_wrong_per_fold=np.bincount(f[error & unanimous], minlength=8).tolist())
    for j, name in enumerate(RELEASES):
        correct = (matrix[:,j] == y)
        report['by_candidate'][name] = dict(test_edits=int(change[:,j].sum()),
            additional_test_rows=int((change[:,j] & ~change[:,:j].any(1)).sum()),
            local_fixed=int((known & (base != y) & correct).sum()), local_harmed=int((known & (base == y) & ~correct).sum()))
    print(report, flush=True)
    path = ROOT / 'reports/nov23_matched_learning_curve/frozen_queue_shared_errors.json'
    path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    run()
