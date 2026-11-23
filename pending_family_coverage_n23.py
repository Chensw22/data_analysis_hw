"""Measure exact error coverage and evaluate family-level rules over eleven pending releases."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_flip_capacity_n23 import required
from audit_pending_booking_influence_n23 import votes_for
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from export_budget_quarter_n23 import source
from export_five_release_majority_n23 import load_votes
from pending_booking_size_router_n23 import rules as booking_rules
from pending_submission_diversity_n23 import DONOR, LINEAR, PAIR, PAIRS, PERSONAL, THREE, TREE
from service_pair_change_budget_n23 import changed

LARGE = 'proposal_n23_large_booking_union'
ALL = 'proposal_n23_all_booking_union'
DIRECT = 'proposal_n23_registration_tree_quarter'
REPORT = ROOT/'reports/nov23_pending_family_coverage'


def rules(ref, votes):
    changes = {key: vote != ref for key, vote in votes.items()}
    # Combined files repeat component votes. Collapse related models before counting.
    families = np.column_stack([
        changes[PERSONAL] | changes[THREE], changes[LINEAR], changes[TREE],
        changes[DONOR] | changes[PAIR+'quarter'] | changes[PAIR+'change_150'], changes[DIRECT],
    ])
    count = families.sum(1)
    any_vote = np.logical_or.reduce(list(changes.values()))
    np.testing.assert_array_equal(any_vote, count > 0)
    return {'all': any_vote, 'two_families': count >= 2, 'queued_or_tree': changes[DONOR] | changes[DIRECT],
            'queued_or_two': changes[DONOR] | (count >= 2)}, count


def read(path):
    return json.loads(path.read_text())


def development_votes(frame, groups, row, y, seed, weight):
    ref, votes = votes_for(frame, groups, row, y, seed, weight)
    old = booking_rules(votes[DONOR], votes[LINEAR], ref, frame.BookingSize.to_numpy())
    votes.update({LARGE: old['large_union'], ALL: old['union']})
    with np.load(ART/f'nov23_registration_bill_links/{seed}.npz') as cache:
        for key, value in [('row', row), ('y', y), ('ref', ref)]:
            np.testing.assert_array_equal(cache[key], value)
        votes[DIRECT] = changed(cache['direct'], cache['base'], ref, .25)
    return ref, votes


def main():
    row, y, frame, groups, _ = load()
    wrong, _, hard = cohorts(row, y)
    previous = read(ROOT/'reports/nov23_recent_family_transfer/pending_diversity.json')['files']
    names = [*previous, LARGE, ALL, DIRECT]
    assert len(set(names)) == 11
    receipts = {read(p)['file_sha256'] for p in (ROOT/'reports').glob('*/online_attempts/*.json')}
    members = {n: read(ROOT/'reports'/n/'manifest.json')['sha256'] for n in names}
    assert len(set(members.values())) == 11 and not receipts.intersection(members.values())
    weight = read(PAIRS/'test_budget.json')['test']['change_150']['weight']
    combined = read(ROOT/'reports/nov23_booking_influence/combined.json')['plans']
    tree = read(ROOT/'reports/nov23_registration_bill_links/registration_bill_links.json')['plans']
    plans, history = {}, {}
    for i, seed in enumerate([*map(str, SEEDS), 'v131']):
        ref, votes = development_votes(frame, groups, row, y, seed, weight)
        hit = y == ref
        assert set(votes) == set(names)
        for key, vote in votes.items():
            metrics = paired(vote == y, hit)
            if key in previous:
                assert metrics['fixed'] - metrics['harmed'] == previous[key]['net_correct'][i]
            elif key == DIRECT:
                assert metrics == tree[seed]['changes']['quarter_direct']
            else:
                assert metrics == combined[seed]['large_union' if key == LARGE else 'union']['vs_reference']
        choices, count = rules(ref, votes)
        metrics = {k: paired(hit ^ mask, hit) for k, mask in choices.items()}
        attainable = int(hit.sum()) + metrics['all']['fixed']
        plans[seed] = dict(baseline=int(hit.sum()), errors=int((~hit).sum()), reachable_correct=attainable,
                           shortfall=required(len(y))-attainable, rules=metrics,
                           by_family_count=[dict(count=k, wrong=int(((count == k) & ~hit).sum()),
                                                 correct=int(((count == k) & hit).sum())) for k in range(6)])
        if seed != 'v131':
            for key, mask in choices.items():
                history.setdefault(key, []).append(hit ^ mask)
    stability = {}
    for key, outcomes in history.items():
        votes_right = np.stack(outcomes)
        stability[key] = dict(hard_ever=int((hard & votes_right.any(0)).sum()),
                              hard_stable=int((hard & votes_right.all(0)).sum()),
                              common_stable=int((wrong & votes_right.all(0)).sum()))
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=['PassengerId'])
    _, incumbent, _ = source('proposal_v131_party_consensus', test)
    matrix = load_votes({'members': members}, test.PassengerId)
    test_rules, count = rules(incumbent, dict(zip(names, matrix.T)))
    result = dict(plans=plans, stability=stability, passengers=len(y), required=required(len(y)),
                  recurring_common=int(hard.sum()), test_rows=len(test), member_hashes=members,
                  test_changes={k: int(v.sum()) for k, v in test_rules.items()},
                  test_support=np.bincount(count, minlength=6).tolist())
    write(REPORT/'coverage.json', result)
    print(result['test_changes'], stability)
    for seed, p in plans.items():
        print(seed, p['shortfall'], {k: (v['fixed'], v['harmed']) for k, v in p['rules'].items()})


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
