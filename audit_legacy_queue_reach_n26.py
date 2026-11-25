"""Compare seven frozen older files with current releases on matching historical folds."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_export_score_limits_n23 import all_candidate_votes
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS
from evaluate_nested_release_n26 import stages
from family_identity_transfer_n26 import panels_and_reference
from frozen_queue_shared_errors_n26 import RELEASES
from full_roster_booking_validation_n23 import data
from held_expert_portfolios_n26 import DECK, REVIEW, panel_predictions, sources, splits
from joint_candidate_coverage_n26 import branch_decisions
from pending_submission_diversity_n23 import PAIRS
from six_way_against_scored_nine_n26 import test_decisions

OLD = ['all_booking_union', 'budget_quarter_incumbent', 'large_booking_union', 'pending_family_support',
       'service_bag_tree_quarter', 'service_pair_change_150', 'service_pair_corroborated']
NAMES = ['proposal_n23_' + n for n in OLD]
OUT = ROOT/'reports/nov23_matched_learning_curve/legacy_queue_reach.json'


def saved(path):
    return json.loads((ROOT/'reports'/path).read_text())


def archived(n, seed, i):
    if n in ('all_booking_union', 'large_booking_union'):
        path = 'nov23_booking_influence/combined.json'
        m = saved(path)['plans'][seed]['union' if n == OLD[0] else 'large_union']['vs_reference']
    elif n == 'pending_family_support':
        path = 'nov23_pending_family_coverage/coverage.json'
        m = saved(path)['plans'][seed]['rules']['queued_or_two']
    elif n == 'budget_quarter_incumbent':
        path = 'nov23_budget_quarter_replay/result.json'
        m = saved(path)['original_v131' if seed == 'v131' else seed]
    else:
        r = saved('nov23_recent_family_transfer/pending_diversity.json')['files']['proposal_n23_' + n]
        return r['source'], r['net_correct'][i]
    return 'reports/' + path, m['fixed'] - m['harmed']


def contemporary(row, ref, groups):
    raw, y, g, _ = data()
    _, _, _, ids, names, _, h, _, old_native, _ = panels_and_reference()
    np.testing.assert_array_equal(row, ids)
    np.testing.assert_array_equal(groups, g[row].astype(int))
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        np.testing.assert_array_equal(ref, z['v131'])
        np.testing.assert_array_equal(z['rows'], row)
        nine, fold = z['nine'], z['fold']
    with np.load(ART/'nov23_broad_partition_evidence/v131.npz') as z:
        np.testing.assert_array_equal(z['fold'], fold)
    br = branch_decisions(row, fold, h, names, old_native, ref, nine, nine,
                          raw.CryoSleep.eq(False).to_numpy(), 'historical', 891)
    chosen = {}
    for n in ('nine', 'fields', 'awake', 'fields_awake', 'band', 'all_three'):
        np.testing.assert_array_equal(br[n][0], br[n][1])
        chosen[n] = br[n][0]
    banks = sources()[-1]
    v = banks['historical']
    np.testing.assert_array_equal(v['ids'], row)
    np.testing.assert_array_equal(v['anchor'], chosen['all_three'])
    p = panel_predictions(v, 'historical')
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled = z['eligible'][row]
    visited = np.zeros(len(row), bool)
    chosen.update({n: visited.copy() for n in ('review', 'fare', 'six')})
    for key, _, _, t, q in splits(banks, g):
        if not key.startswith('historical_') or key == 'historical_to_new':
            continue
        assert not visited[q].any() and not np.isin(groups[q], groups[t]).any()
        visited[q] = True
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'], row[q])
            risk = np.column_stack([z['identities_error'], z['identities_outcome']])
        _, *versions = stages(v, p, q, enabled[q], risk)
        for n, a in zip(('review', 'fare', 'six'), versions, strict=True):
            chosen[n][q] = a
    assert visited.all()
    np.testing.assert_array_equal(y[row], v['wrong'] ^ v['anchor'])
    cold = np.zeros(len(row), bool)
    for k in np.unique(fold):
        m = fold == k
        cold[m] = ~np.isin(groups[m], groups[~m])
    return chosen, fold, cold


def stats(v, before, y, mask, fold, g):
    changed = mask & (v != before)
    fixed = changed & (v == y)
    harmed = changed & (v != y)
    gain = fixed.astype(int) - harmed
    _, group = np.unique(g[mask], return_inverse=True)
    counts = np.bincount(group)
    by_group = np.bincount(group, weights=gain[mask])
    rng = np.random.default_rng(2624)
    samples = []
    for _ in range(20):
        draw = rng.integers(len(counts), size=(100, len(counts)))
        samples.extend(100 * by_group[draw].sum(1)/counts[draw].sum(1))
    return dict(fixed=int(fixed.sum()), harmed=int(harmed.sum()), net=int(gain.sum()),
                net_by_fold=[int(gain[fold == k].sum()) for k in np.unique(fold)],
                paired_booking_interval_pp=np.quantile(samples, [.025, .975]).tolist())


def run():
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=['PassengerId'])
    current = {k: test_decisions(n, test.PassengerId)[0] for k, n in RELEASES.items()}
    assert not ((current['filtered'] != current['nine']) & (current['awake'] == current['nine'])).any()
    held = np.column_stack(list(current.values()))
    unanimous = (held == held[:, :1]).all(1)
    receipts = {json.loads(f.read_text()).get('file_sha256') for f in (ROOT/'reports').glob('*/online_attempts/*.json')}
    result = dict(test_rows=len(test), current_test_unanimous=int(unanimous.sum()), candidates={})
    test_new = []
    for n, proposal in zip(OLD, NAMES, strict=True):
        b, digest = test_decisions(proposal, test.PassengerId)
        assert digest not in receipts
        new = unanimous & (b != held[:, 0])
        test_new.append(new)
        result['candidates'][n] = dict(sha256=digest, test_distances={k: int((b != v).sum()) for k, v in current.items()},
                                      test_new_rows=int(new.sum()), legacy_net_by_plan={})
    result['seven_union_new_test_rows'] = int(np.any(test_new, axis=0).sum())
    row, y, frame, groups, _ = load()
    weight = json.loads((PAIRS/'test_budget.json').read_text())['test']['change_150']['weight']
    for i, seed in enumerate([*map(str, SEEDS), 'v131']):
        ref, votes = all_candidate_votes(frame, groups, row, y, seed, weight)
        for n, key in zip(OLD, NAMES, strict=True):
            path, archived_net = archived(n, seed, i)
            v = votes[key]
            net = int((v == y).sum() - (ref == y).sum())
            assert net == archived_net
            result['candidates'][n]['legacy_net_by_plan'][seed] = net
            result['candidates'][n]['legacy_metric_source'] = path
    chosen, fold, cold = contemporary(row, ref, groups)
    current_votes = np.column_stack(list(chosen.values()))
    no_change = (current_votes == current_votes[:, :1]).all(1)
    result['protocol'] = dict(historical_rows=len(y), fully_held_booking_rows=int(cold.sum()),
                              scope='Same historical outer folds and same passenger ids. Only complete-order holdout is used for recommendation; other historical rows can have a booking peer in the outer training fold. Three extra early plans each use a different parent, not the scored nine. This is reused development data.',
                              filtered='The filtered release only accepts edits made by awake, so it cannot affect exact current unanimity. No individual historical predictions for filtered are assumed.')
    pools = dict(whole_order_cold=cold, all_historical=np.ones(len(y), bool))
    result['current_unanimously_wrong'] = {}
    result['seven_union_outside_current_reach'] = {}
    any_old_flip = np.any([votes[key] != chosen['nine'] for key in NAMES], axis=0)
    for scope, m in pools.items():
        result['current_unanimously_wrong'][scope] = int((m & no_change & (chosen['nine'] != y)).sum())
        extra = m & no_change & any_old_flip
        result['seven_union_outside_current_reach'][scope] = dict(
            recoverable_errors=int((extra & (chosen['nine'] != y)).sum()),
            correct_at_risk=int((extra & (chosen['nine'] == y)).sum()))
        for n, key in zip(OLD, NAMES, strict=True):
            v = votes[key]
            result['candidates'][n][scope] = {
                **{f'versus_{comp}':stats(v, chosen[comp], y, m, fold, groups) for comp in ('nine','six')},
                'outside_current_reach':stats(v, chosen['nine'], y, no_change & m, fold, groups)}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    for n, r in result['candidates'].items():
        print(n, r['test_new_rows'], [(k, v['fixed'], v['harmed'], v['net_by_fold'])
                                    for k, v in r['whole_order_cold'].items()])


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
