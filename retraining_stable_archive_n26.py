"""Learn archive weights constrained by paired score changes after refitting on different bookings."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import ROOT, data
from kaggle_workflow import sha256
from repeated_archive_misses_n26 import ART, cohort, count

DIR = ART / 'nov26_retraining_stable_archive'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/retraining_stable_archive.json'


def features(p):
    odds = logit(p.clip(.01, .99))
    # One coefficient per configuration, centered on each passenger's average expert.
    return np.column_stack([np.ones(len(p)), odds.mean(1), odds - odds.mean(1, keepdims=True)])


def fit(x, y, pairs):
    # If a passenger's training bank changes, penalize changes in their combined log-odds.
    penalty = 14 * np.eye(x.shape[1])
    penalty[0, 0] = .5
    if len(pairs):
        penalty += len(y) * (pairs.T @ pairs) / len(pairs)
    def loss(beta):
        z = x @ beta
        return (np.logaddexp(0, z).sum() - y @ z + .5 * beta @ penalty @ beta,
                x.T @ (expit(z) - y) + penalty @ beta)
    ans = minimize(loss, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=400, ftol=1e-10))
    assert ans.success, ans.message
    return ans.x


def run():
    raw, y, g, f = data()
    base, blind, known, fresh = cohort(raw, y, f)
    yy, gg, ff, rows, names, full, historical, *_ = panels_and_reference()
    for a, b in [(y, yy), (g, gg), (f, ff)]:
        np.testing.assert_array_equal(a, b)
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        oldfold = z['fold']
        np.testing.assert_array_equal(z['rows'], rows)
    manifest = json.loads((ROOT/'reports/proposal_n26_dispute_band/manifest.json').read_text())
    freeze = ART/'proposal_n26_dispute_band/frozen_votes.npz'
    assert sha256(freeze) == manifest['model_sha256']
    with np.load(freeze) as z:
        np.testing.assert_array_equal(z['names'], names)
        test_ids, test = z['passenger'], z['p']
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'][len(y):], test_ids)
        test_base = z['anchor'][len(y):]
    assert not np.isin(np.char.partition(test_ids, '_')[:, 0], g).any()
    x, h, xt = map(features, (full, historical, test))
    # A query booking present historically is always assigned to a historical fold that excluded it.
    historic_group = pd.Series(oldfold, index=g[rows]).groupby(level=0).min()
    hslot = pd.Series(g).map(historic_group)
    unique_old = np.unique(oldfold)
    unknown_slot = unique_old[partition(g, 'unseen_archive_bank', len(unique_old))]
    hslot = np.where(hslot.isna(), unknown_slot, hslot).astype(int)
    inner = partition(g, 'paired_score_retrain', 3)
    controls = ('ordinary', 'paired', 'permuted')
    oof = {mode:np.full(len(y), np.nan) for mode in controls}
    test_models = {mode:[] for mode in controls}
    for k in np.unique(f):
        outer = np.flatnonzero(f == k)
        for old in unique_old:
            # The historical score used in any pair must itself exclude every held booking.
            available = (f[rows] == k) & (oldfold == old)
            for split in [0, 1, 2, None]:
                query = outer[(inner[outer] == split) & (hslot[outer] == old)] if split is not None else []
                t = outer[known[outer] & (inner[outer] != split)] if split is not None else outer[known[outer]]
                at = np.flatnonzero(available & np.isin(rows, t))
                if split is not None:
                    assert not np.isin(g[t], g[query]).any()
                    old_fit = rows[(oldfold != old) & ~np.isin(g[rows], g[rows[oldfold == old]])]
                    assert not np.isin(g[old_fit], g[query]).any()
                assert len(at) > 10
                rng = np.random.default_rng(81030 + k * 100 + old)
                # Match permuted pairs on the true label inside calibration data, preserving class-average shifts.
                other = at.copy()
                for label in (False, True):
                    loc = y[rows[at]] == label
                    other[loc] = rng.permutation(at[loc])
                pair_sets = dict(ordinary=[], paired=x[rows[at]]-h[at], permuted=x[rows[at]]-h[other])
                for mode in controls:
                    beta = fit(x[t], y[t], pair_sets[mode])
                    if split is None:
                        test_models[mode].append(expit(xt @ beta))
                    else:
                        oof[mode][query] = expit(x[query] @ beta)
    scores = {}
    for mode in controls:
        assert np.isfinite(oof[mode]).all()
        scores[mode] = np.r_[oof[mode], np.mean(test_models[mode], axis=0)]
    current = np.r_[base, test_base]
    flags = {mode:(p >= .5) != current for mode, p in scores.items()}
    flags['only_paired'] = flags['paired'] & ~flags['ordinary'] & ~flags['permuted']
    flags['only_permuted'] = flags['permuted'] & ~flags['ordinary'] & ~flags['paired']
    flags['paired_with_ordinary'] = flags['paired'] & flags['ordinary']
    scopes = dict(known=known, blind=blind, fresh=known & fresh)
    report = dict(accuracy={k:float(((p[:len(y)] >= .5) == y).mean()) for k,p in scores.items()}, rules={})
    for k, v in flags.items():
        edit = v[:len(y)]
        delta = edit * np.where(base != y, 1, -1)
        report['rules'][k] = dict(test_changes=int(v[len(y):].sum()),
            by_fold=[int(delta[known & (f == i)].sum()) for i in np.unique(f)],
            scopes={s:count(edit, mask, base != y) for s, mask in scopes.items()})
    DIR.mkdir(exist_ok=True)
    np.savez_compressed(DIR/'aligned.npz', passenger=np.r_[raw.PassengerId.to_numpy(str),test_ids], y=y, fold=f, **scores)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['accuracy'], [(k,v['scopes']['known'],v['scopes']['blind']) for k,v in report['rules'].items()])


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
