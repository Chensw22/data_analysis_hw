"""Resolve the incumbent on grouped folds using exact peer offsets and bounds on unknown committee decisions."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

import peer_bill_correction as peer
from certified_tree_release_n25 import structural_inputs
from export_budget_quarter_n23 import source
from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_conditional_pool_n23 import banks, data
from full_roster_selected_tree_replay_n24 import original_prediction, prepared_vote
from grouped_peer_teacher_n25 import OUT, OUTER
from party_negative_rescue import family_contrasts, risk
from relational_error_corrector import logit
from sleeper_manifest_risk import design

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/grouped_peer_release_replay.json'


def limits(native, patch, known, peer_shift, sleeping, witness, awake_party, family):
    # v106 copies native outside patched queries. All mathematically unresolved inputs
    # are permitted both signs; a changed v128 sign cannot activate the p108<.5 family rule.
    # The deployment reads the shortest float32 decimal from the v106 CSV as float64.
    p = native.astype(np.float32).astype(str).astype(float)
    p[known] = expit(logit(p[known]) + peer_shift[known])
    base = p >= .5
    may_lower = base & sleeping & (witness < .5)
    may_raise = ~base & awake_party & (family >= .60)
    return p, (~(may_lower | may_raise) & ~patch), base


def edit_effect_bounds(candidate, native_sign, v131, low_nine, high_nine, truth):
    possible = []
    for side, branch in ((False, low_nine), (True, high_nine)):
        parent = np.where(v131 >= 0, v131, side)
        edited = np.where((native_sign == parent) & (branch == parent), candidate, branch)
        possible.append((edited == truth).astype(int) - (branch == truth))
    return np.minimum(*possible), np.maximum(*possible)


def run(folds=OUTER):
    raw, y, groups, fold = data()
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=lambda k: k != 'Name')
    roster = pd.concat([raw, test], ignore_index=True)
    features, control, size, x = family_contrasts(roster)
    awake = x.CryoSleep.ne('True').to_numpy()
    patch, known, sleeping, awake_party = structural_inputs(roster)
    ids, peer_known, _, fields = peer.context()
    np.testing.assert_array_equal(ids, roster.PassengerId)
    np.testing.assert_array_equal(known, peer_known)
    wx = design(x, True)
    names, _, full = banks(raw, y, fold)
    opinions = full[:, [names.index(k) for k in (*FAMILIES, 'mitra96', *EVIDENCE)]]
    tree = original_prediction(np.arange(len(y)), fold, groups, y, 'full')
    with np.load(ART/'nov25_production_anchor_cpu/oof.npz') as z:
        np.testing.assert_array_equal(z['passenger'], raw.PassengerId)
        np.testing.assert_array_equal(z['fold'], fold)
        native, old_cert, old_nine = [z[k] for k in ('v96','certified_v131_equals_v96','certified_nine')]
        old_w, old_family = z['witness'], z['family']
    test_native = source('proposal_online_first_counts15', test)[0]
    test_106 = source('proposal_nov18_peer_cabin', test)[0]
    test_108 = source('proposal_v108_party_bills', test)[0]
    saves = np.arange(len(raw), len(roster))
    unchanged = ~patch[saves]
    np.testing.assert_array_equal(test_native.astype(np.float32).astype(str).astype(float)[unchanged], test_106[unchanged])
    fit_test = joblib.load(ART/'proposal_v108_party_bills/peer_coefficients.joblib')
    use = known[saves]
    replay = test_106.copy()
    replay[use] = expit(logit(replay[use]) + peer.apply(fit_test, fields[saves[use]]))
    np.testing.assert_allclose(replay, test_108, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(replay >= .5, test_108 >= .5)
    output = {}
    for k in folds:
        q, t = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
        with np.load(OUT / f'outer{k}.npz') as z:
            for key, val in [('t',t), ('q',q), ('labels',y[t]), ('passenger',raw.PassengerId.iloc[t])]:
                np.testing.assert_array_equal(z[key], val)
            raw_teacher = z['p']
        assert not np.intersect1d(groups[q], groups[t]).size
        fit = peer.fit_offset(fields[t][known[t]], raw_teacher[known[t]], y[t][known[t]].astype(int))
        shift = np.zeros(len(q))
        shift[known[q]] = peer.apply(fit, fields[q][known[q]])
        witness = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000))
        learn = t[sleeping[t]]
        witness.fit(wx[learn], y[learn])
        wp = witness.predict_proba(wx[q])[:, 1]
        fp = risk(features, control, y, t, q, size, awake)[:, 0]
        np.testing.assert_allclose(wp, old_w[q], atol=1e-10, rtol=0)
        np.testing.assert_allclose(fp, old_family[q], atol=1e-10, rtol=0)
        p108, cert, v131 = limits(native[q], patch[q], known[q], shift, sleeping[q], wp, awake_party[q], fp)
        assert np.all(cert[old_cert[q]])
        nine = prepared_vote(tree[q], v131, opinions[q])
        np.testing.assert_array_equal(nine[old_cert[q]], old_nine[q][old_cert[q]])
        lo = prepared_vote(tree[q], np.where(cert, v131, False), opinions[q])
        hi = prepared_vote(tree[q], np.where(cert, v131, True), opinions[q])
        fixed_nine = lo == hi
        prior_precision = native[q].astype(float).copy()
        prior_precision[known[q]] = expit(logit(prior_precision[known[q]]) + shift[known[q]])
        legacy_changes = int((~patch[q] & ((p108 >= .5) != (prior_precision >= .5))).sum())
        assert np.all(fixed_nine[cert]) and np.all(lo <= hi)
        # Keep ambiguous values explicitly marked so callers cannot use the placeholder sign.
        path = OUT / f'release{k}.npz'
        with path.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, q=q, ids=raw.PassengerId.iloc[q].to_numpy(str), y=y[q],
                                p108=np.where(patch[q], np.nan, p108), native=native[q],
                                v131=np.where(cert, v131, -1), nine=np.where(fixed_nine, lo, -1),
                                original_cert=old_cert[q], party=size[q] > 1, shift=shift,
                                nine_on_low_branch=lo, nine_on_high_branch=hi)
        path.with_suffix('.tmp').replace(path)
        output[str(k)] = dict(rows=len(q), original=int(old_cert[q].sum()), v131_resolved=int(cert.sum()),
                              nine_resolved=int(fixed_nine.sum()), nine_only=int((fixed_nine & ~cert).sum()),
                              serialization_changes_decisions=legacy_changes,
                              party_rows=int((size[q] > 1).sum()), party_v131=int((cert & (size[q]>1)).sum()),
                              party_nine=int((fixed_nine & (size[q]>1)).sum()),
                              native_to_v131_edits=int((cert & (v131 != (native[q]>=.5))).sum()),
                              v131_to_nine_edits=int((cert & (nine != v131)).sum()))
        print(k, output[str(k)], flush=True)
    target = REPORT if tuple(folds) == OUTER else OUT / ('release_summary_' + '_'.join(map(str, folds)) + '.json')
    target.write_text(json.dumps(dict(official_test_replay=dict(unchanged_v106=int(unchanged.sum()), v108_rows=len(test),
                                                                   max_abs_error=float(abs(replay-test_108).max())),
                                      folds=output, totals={key:sum(part[key] for part in output.values())
                                                            for key in next(iter(output.values()))}), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=OUTER)
    args = parser.parse_args()
    with threadpool_limits(1):
        run(OUTER if args.fold is None else (args.fold,))
