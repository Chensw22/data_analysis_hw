"""Locate novel queued fixes and test complementary votes outside all pending changes."""
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from audit_export_score_limits_n23 import all_candidate_votes
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_tree_field_dividends_n23 import hard_cohorts
from export_five_release_majority_n23 import load_votes
from pending_submission_diversity_n23 import DONOR, PAIRS
from released_correction_audit_n23 import development, release


def counts(mask, original, y, hard):
    fixed = int(np.sum(mask & (original != y)))
    harmed = int(np.sum(mask & (original == y)))
    return dict(fixed=fixed, harmed=harmed, net=fixed-harmed,
                shared_fixed=int(np.sum(mask & (original != y) & hard)))


def adjudicate(groups, fold, y, ref, quiet, base, profile, pair):
    # Fit on other orders scored by the same outer-fold models, keeping the
    # base learners used for gate training blind to the query order's labels.
    eligible = quiet & ((profile >= .5) != ref) & ((pair >= .5) != ref)
    features = logit(np.column_stack([base, profile, pair]).clip(.001, .999))
    features *= np.where(ref[:, None], -1., 1.)
    features = np.column_stack([ref, features])
    prediction = np.zeros((len(y), 2), dtype=bool)
    assigned = np.zeros(len(y), bool)
    for outer in np.unique(fold):
        ids = np.flatnonzero((fold == outer) & eligible)
        for a, b in GroupKFold(3, shuffle=True, random_state=54).split(ids, groups=groups[ids]):
            t, q = ids[a], ids[b]
            assert not np.intersect1d(groups[t], groups[q]).size
            for j, width in enumerate([2, 4]):
                model = LogisticRegression(C=.25).fit(features[t, :width], y[t] != ref[t])
                prediction[q, j] = model.predict(features[q, :width])
            assigned[q] = True
    np.testing.assert_array_equal(assigned, eligible)
    return prediction.T


def run():
    row, y, frame, groups, _ = load()
    _, _, hard, _ = hard_cohorts(row, y)
    archive_y, archive, _ = development()
    np.testing.assert_array_equal(y, archive_y)
    official = json.loads((ROOT / 'reports/nov23_flip_capacity/primary_scores.json').read_text())['public_scores']
    ids = pd.read_csv(ROOT / 'data/raw/test.csv', usecols=['PassengerId']).PassengerId
    historical_test = {v: release(v, ids, official)[1] for v in archive}
    digest = json.loads((ROOT / 'reports' / DONOR / 'manifest.json').read_text())['sha256']
    queued = load_votes({'members': {DONOR: digest}}, ids)[:, 0]
    changed = queued != historical_test[131]
    backed = np.any([v != historical_test[131] for v in historical_test.values()], axis=0)
    test = dict(changes=int(changed.sum()), backed=int((changed & backed).sum()), novel=int((changed & ~backed).sum()))
    weight = json.loads((PAIRS / 'test_budget.json').read_text())['test']['change_150']['weight']
    by_split, common_hits = {}, {}
    for seed in [*map(str, SEEDS), 'v131']:
        ref, votes = all_candidate_votes(frame, groups, row, y, seed, weight)
        assert len(votes) == 13
        quiet = ~np.any([v != ref for v in votes.values()], axis=0)
        queued_flip = votes[DONOR] != ref
        assert not (quiet & queued_flip).any()
        with np.load(ART / f'nov23_competing_affine_profiles/{seed}.npz') as a, np.load(
                ART / f'nov23_opposing_passengers/{seed}.npz') as b:
            for z in [a, b]:
                for k, expected in [('ref', ref), ('row', row), ('y', y)]:
                    np.testing.assert_array_equal(z[k], expected)
            np.testing.assert_array_equal(a['fold'], b['fold'])
            profiles = (a['averaged'] >= .5) != ref
            pairs = (b['matched_conditional'] >= .5) != ref
            shuffled = (b['shuffled_conditional'] >= .5) != ref
            risk_gate, full_gate = adjudicate(groups, a['fold'], y, ref, quiet, a['base'],
                                              a['averaged'], b['matched_conditional'])
        extra = quiet & profiles & pairs
        arms = dict(queued=queued_flip, outside_profiles=quiet & profiles, outside_pairs=quiet & pairs,
                    both_outside=extra, shuffled_control=quiet & profiles & shuffled, queued_plus_both=queued_flip | extra,
                    risk_gate=risk_gate, full_gate=full_gate, queued_plus_gate=queued_flip | full_gate)
        if seed == 'v131':
            np.testing.assert_array_equal(archive[131], ref)
            supported = np.any([v != ref for v in archive.values()], axis=0)
            arms.update(queued_backed=queued_flip & supported, queued_novel=queued_flip & ~supported,
                        both_outside_archive=extra & ~supported)
        by_split[seed] = {key: counts(mask, ref, y, hard) for key, mask in arms.items()}
        if seed != 'v131':
            for key in ('both_outside', 'shuffled_control', 'full_gate'):
                common_hits.setdefault(key, []).append(hard & (ref != y) & arms[key])
    recurring = {k: dict(ever=int(np.any(hits, axis=0).sum()), every_split=int(np.all(hits, axis=0).sum()))
                 for k, hits in common_hits.items()}
    return dict(historical_versions=sorted(archive), test=test, development=by_split, shared_recovery=recurring)


if __name__ == '__main__':
    result = run()
    write(ROOT / 'reports/nov23_booking_influence/queued_origins.json', result)
    for split, values in result['development'].items():
        print(split, values)
    print(result['test'])
