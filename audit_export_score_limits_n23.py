"""Bound frozen exports using only file differences and previously published aggregate scores."""
import json
from decimal import Decimal

import numpy as np
import pandas as pd

from audit_flip_capacity_n23 import ART, REPORT, ROOT, correct_count, required, same_denominator
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from kaggle_workflow import validate_submission
from pending_family_coverage_n23 import PAIRS, SEEDS, development_votes, rules


def local_coverage(names):
    row, y, frame, groups, _ = load()
    _, _, hard = cohorts(row, y)
    weight = json.loads((PAIRS / 'test_budget.json').read_text())['test']['change_150']['weight']
    plans = {}
    for seed in [*map(str, SEEDS), 'v131']:
        ref, votes = development_votes(frame, groups, row, y, seed, weight)
        decisions, _ = rules(ref, votes)
        votes['proposal_n23_pending_family_support'] = ref ^ decisions['queued_or_two']
        if seed == 'v131':
            path = ART / 'nov23_budget_quarter_replay/original.npz'
        else:
            path = ART / f'nov23_joint_risk_surface/{seed}.npz'
        with np.load(path) as cache:
            np.testing.assert_array_equal(row, cache['row'])
            np.testing.assert_array_equal(y, cache['y'])
            if seed == 'v131':
                np.testing.assert_array_equal(ref, cache['ref'])
                budget_vote = cache['vote']
            else:
                np.testing.assert_array_equal(ref, cache['base'] >= .5)
                budget_vote = cache['fixed_quarter'] >= .5
        votes['proposal_n23_budget_quarter_incumbent'] = budget_vote
        assert len(votes) == 13 and set(votes).issubset(names)
        reachable = np.any([v != ref for v in votes.values()], axis=0) & (ref != y)
        best_possible = int((ref == y).sum() + reachable.sum())
        plans[seed] = dict(reachable_errors=int(reachable.sum()), hard_reachable=int((hard & reachable).sum()),
                           oracle_correct=best_possible, shortfall=required(len(y)) - best_possible)
    return plans


def run():
    test = ROOT / 'data/raw/test.csv'
    n = len(pd.read_csv(test, usecols=['PassengerId']))
    scores = json.loads((REPORT / 'primary_scores.json').read_text())['public_scores']
    digits = {int(Decimal(score) * 100000) for score in scores.values()}
    assert same_denominator(digits, n, 'either') == [n]
    files = {}
    for path in sorted(ART.glob('*/submission.csv')):
        name = path.parent.name
        digest = validate_submission(path, test, n)['sha256']
        manifest = ROOT / 'reports' / name / 'manifest.json'
        if manifest.exists():
            assert json.loads(manifest.read_text())['sha256'] == digest
        votes = pd.read_csv(path, usecols=['Transported']).Transported.to_numpy()
        files[name] = (digest, votes, manifest.exists())

    receipts = {}
    for path in (ROOT / 'reports').glob('*/online_attempts/*.json'):
        entry = json.loads(path.read_text())
        ref = str(entry.get('submission_ref'))
        evaluation = entry.get('evaluation', {})
        if ref not in scores or evaluation.get('Status') != 'COMPLETE':
            continue
        name = path.parents[1].name
        digest, votes, _ = files[name]
        assert ref not in receipts and digest == entry['file_sha256']
        assert Decimal(scores[ref]) == Decimal(evaluation['Public Score'])
        receipts[ref] = (name, digest, votes, correct_count(scores[ref], n))
    assert set(receipts) == set(scores)
    scored_digests = {digest for _, digest, _, _ in receipts.values()}

    def tightest(votes, skip=None):
        bounds = []
        for ref, (name, _, old, correct) in receipts.items():
            if ref != skip:
                distance = int(np.count_nonzero(votes != old))
                # At most one extra passenger can become correct per differing decision.
                upper = min(correct + distance, 2 * n - correct - distance)
                bounds.append((upper, distance, ref, name, correct))
        upper, distance, ref, name, correct = min(bounds)
        return dict(max_correct=upper, reference=name, ref=ref, reference_correct=correct,
                    differing_rows=distance, shortfall=max(0, required(n) - upper))

    for ref, (_, _, votes, correct) in receipts.items():
        limit = tightest(votes, skip=ref)['max_correct']
        assert correct <= limit and (limit - correct) % 2 == 0
    unscored = {name: dict(**tightest(votes), sha256=digest, has_manifest=manifest)
                for name, (digest, votes, manifest) in files.items() if digest not in scored_digests}
    baseline = receipts['58653601'][2]
    free = np.any([files[name][1] != baseline for name in unscored], axis=0)
    # A union can undo all reference mistakes on free rows. On fixed rows it can
    # only gain where its decision differs from the reference.
    union_limit, union_reference = min(
        (min(n, correct + int(free.sum()) + int(np.count_nonzero(baseline[~free] != votes[~free]))), name)
        for name, _, votes, correct in receipts.values())
    return dict(rows=n, required_correct=required(n), scored_references=len(receipts),
                scanned_files=len(files), unscored=unscored, union_changed=int(free.sum()),
                union_upper_correct=union_limit, union_reference=union_reference,
                local=local_coverage(unscored))


if __name__ == '__main__':
    result = run()
    path = REPORT / 'export_limits.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    for name, row in result['unscored'].items():
        print(row['max_correct'], name)
