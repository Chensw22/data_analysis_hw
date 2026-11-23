"""Count new error coverage and test corroborating votes from distinct pending models."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_flip_capacity_n23 import required
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_service_pairs_n23 import REPORT
from broad_three_corrections_replay_n23 import scores
from export_budget_quarter_n23 import source
from export_service_bag_n23 import NAME as LINEAR, PERSONAL, THREE
from export_service_bag_tree_n23 import NAME as TREE
from kaggle_workflow import sha256
from service_pair_change_budget_n23 import changed

NAMES = [PERSONAL, THREE, LINEAR, TREE, 'proposal_n23_service_pair_change_150']


def actions(changes):
    personal, three, linear, tree, pair = changes.T
    # Personal and the three-way mix share a parent; both trees use service exposures.
    profile = personal | three
    service = tree | pair
    families = profile.astype(int) + linear + service
    outside = ~changes[:, :4].any(axis=1)
    return {
        'pair': pair,
        'old_union': ~outside,
        'union': ~outside | pair,
        'pair_only_new': pair & outside,
        'others_only': ~outside & ~pair,
        'two_families': families >= 2,
        'pair_plus_corroborated': pair | ((profile.astype(int) + linear + tree) >= 2),
        'pair_plus_independent': pair | (profile & linear),
        'pair_confirmed_by_other_family': pair & (profile | linear),
    }


def replay_votes(frame, groups, row, y, seed, weight):
    three, ref = scores(frame, groups, row, y, seed)
    with np.load(ART / f'nov23_broad_partition_evidence/{seed}.npz') as src:
        base, fold = src['base'], src['fold']
    views = [three[:, 0] >= .5, three.mean(1) >= .5]
    for folder, key, w in [('service_bag', 'fixed', .25), ('service_bag_trees', 'initial', .25),
                           ('service_pairs', 'same_passenger_both', weight)]:
        with np.load(ART / f'nov23_{folder}/{seed}.npz') as z:
            for field, expected in [('row', row), ('y', y), ('ref', ref), ('fold', fold), ('base', base)]:
                np.testing.assert_array_equal(z[field], expected)
            views.append(changed(z[key], base, ref, w))
    return ref, np.column_stack(views)


def replay(frame, groups, row, y, seed, weight):
    ref, views = replay_votes(frame, groups, row, y, seed, weight)
    return ref, actions(views != ref[:, None])


def test_actions():
    test = pd.read_csv(ROOT / 'data/raw/test.csv', usecols=['PassengerId'])
    _, ref, _ = source('proposal_v131_party_consensus', test)
    views = []
    for name in NAMES:
        path = ART / name / 'submission.csv'
        manifest = json.loads((ROOT / 'reports' / name / 'manifest.json').read_text())
        assert sha256(path) == manifest['sha256']
        df = pd.read_csv(path)
        np.testing.assert_array_equal(df.PassengerId, test.PassengerId)
        views.append(df.Transported)
    return ref, actions(np.column_stack(views) != ref[:, None])


def main():
    weight = json.loads((REPORT / 'test_budget.json').read_text())['test']['change_150']['weight']
    row, y, frame, groups, _ = load()
    wrong, right, hard = cohorts(row, y)
    plans, history = {}, {}
    for seed in [*map(str, SEEDS), 'v131']:
        ref, rules = replay(frame, groups, row, y, seed, weight)
        hit = ref == y
        metrics = {k: paired(hit ^ mask, hit) for k, mask in rules.items()}
        gap = required(len(y)) - int(hit.sum()) - metrics['union']['fixed']
        plans[seed] = dict(metrics=metrics, gap_even_if_oracle=gap, baseline_correct=int(hit.sum()))
        assert metrics['old_union']['fixed'] + metrics['pair_only_new']['fixed'] == metrics['union']['fixed']
        if seed != 'v131':
            for k, mask in rules.items():
                history.setdefault(k, []).append(mask)
            history.setdefault('hit', []).append(hit)
    baseline = np.stack(history.pop('hit'))
    stability = {}
    for name, masks in history.items():
        corrected = np.stack(masks) ^ baseline
        stability[name] = dict(
            gain95_pp=interval((corrected.astype(float)-baseline).mean(0), groups),
            hard_stable=int((hard & corrected.all(0)).sum()), hard_ever=int((hard & corrected.any(0)).sum()),
            common_stable=int((wrong & corrected.all(0)).sum()), stable_harms=int((right & ~corrected.any(0)).sum()))
    _, test_rules = test_actions()
    test = {k: int(v.sum()) for k, v in test_rules.items()}
    assert test['pair'] == 150 and test['union'] == 240 and test['pair_only_new'] == 26
    write(REPORT / 'portfolio.json', dict(test=test, plans=plans, stability=stability))
    for k, n in test.items():
        print(k, n, [p['metrics'][k]['fixed'] - p['metrics'][k]['harmed'] for p in plans.values()])


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
