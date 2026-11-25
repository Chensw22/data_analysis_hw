"""Ask whether another member's model disagreement helps decide a passenger's own case."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_conditional_pool_n23 import banks, data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_teacher_n25 import OUT, OUTER
from grouped_peer_release_replay_n25 import edit_effect_bounds
from monotone_disagreement_n25 import cell_ids, predict as monotone_predict, views
from new_order_error_transfer_n25 import fit_predict, odds

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/companion_disagreement.json'


def summaries(opinions, native, fold, groups, planet):
    members, row, count = np.unique(groups, return_inverse=True, return_counts=True)
    size = count[row]
    native_odds = odds(native)
    delta = (odds(opinions) - native_odds[:, None]).clip(-5, 5)
    personal = np.column_stack([delta, np.median(delta, axis=1)])
    values = np.column_stack([personal, native_odds, np.abs(native - .5)])
    totals = np.zeros((len(members), values.shape[1]))
    np.add.at(totals, row, values)
    other = (totals[row] - values) / np.maximum(size - 1, 1)[:, None]
    # Same-size/planet/outer-fold placebo assigns a different booking's peer summary.
    donors = np.arange(len(row))
    rng = np.random.default_rng(519)
    for f in np.unique(fold):
        for home in np.unique(planet):
            for n in np.unique(size[size > 1]):
                ix = np.flatnonzero((fold == f) & (planet == home) & (size == n))
                for i in ix:
                    options = ix[groups[ix] != groups[i]]
                    if not len(options):
                        options = np.flatnonzero((fold == f) & (size > 1) & (groups != groups[i]))
                    donors[i] = rng.choice(options)
    multi = size > 1
    assert not np.any(groups[donors[multi]] == groups[multi])
    def join(peer):
        interactions = personal * np.tanh(peer[:, :personal.shape[1]])
        return np.column_stack([personal, peer, interactions])
    return {'personal': personal, 'actual_companions': join(other), 'unrelated_companions': join(other[donors])}, multi


def score(pred, baseline, truth, mask, folds, groups):
    delta = (pred == truth).astype(int) - (baseline == truth)
    return dict(rows=int(mask.sum()), original_errors=int((mask & (truth != baseline)).sum()),
                fixed=int((delta[mask] > 0).sum()), harmed=int((delta[mask] < 0).sum()),
                net_interval_pp=interval(delta[mask], groups[mask]),
                fold_net=[int(delta[mask & (folds == f)].sum()) for f in np.unique(folds[mask])])


def run(release):
    raw, y, groups, fold = data()
    names, _, bank = banks(raw, y, fold)
    strong = bank[:, [names.index(k) for k in (*FAMILIES, 'mitra96')]]
    native = strong[:, 4]
    matrices, party = summaries(strong[:, :4], native, fold, groups, raw.HomePlanet.fillna('?').to_numpy(str))
    answer = {}
    for name, x in matrices.items():
        out = native >= .5
        for f in np.unique(fold):
            ix = np.flatnonzero((fold == f) & party)
            for a, b in GroupKFold(4, shuffle=True, random_state=814).split(ix, groups=groups[ix]):
                t, q = ix[a], ix[b]
                assert not np.intersect1d(groups[t], groups[q]).size
                out[q] = fit_predict(x[t], x[q], native[t], y[t], native[q])
        answer[name] = out
    rows = np.arange(len(y))
    with np.load(ART / 'nov25_historical_nine/historical.npz') as z:
        h, fh = z['rows'], z['fold']
    labels, panels = prediction_sets(raw, y, groups, fold, h, fh)
    archive = np.column_stack(panels[0])
    labels = ['response/'+str(i) for i in range(3)] + ['projection/'+str(i) for i in range(3)] + labels
    cells, shape = cell_ids(views(strong, archive, labels)['neural_and_trees'])
    answer['monotone_disagreement'] = monotone_predict(fold, cells, shape, y, groups)
    for col in ('v1', 'v2'):
        out = np.empty(len(y), bool)
        for f in np.unique(fold):
            with np.load(ART / 'nov25_full_context_generations' / f'eight_{f}.npz') as z:
                np.testing.assert_array_equal(z['q'], rows[fold == f])
                out[z['q']] = z[col] >= .5
        answer['frozen_'+col] = out
    native_sign = native >= .5
    agree = ((archive >= .5) == native_sign[:, None]).all(1)
    base = {name:dict(vs_native_party=score(p, native_sign, y, party, fold, groups),
                      vs_native_unanimous_party=score(p, native_sign, y, party & agree, fold, groups))
            for name, p in answer.items()}
    if release:
        v131, nine = np.full(len(y), -1, int), np.full(len(y), -1, int)
        original = np.zeros(len(y), bool)
        low_nine, high_nine = np.zeros(len(y), bool), np.zeros(len(y), bool)
        evaluated = np.isin(fold, OUTER)
        for f in OUTER:
            with np.load(OUT / f'release{f}.npz') as z:
                q = z['q']
                np.testing.assert_array_equal(q, rows[fold == f])
                np.testing.assert_array_equal(z['y'], y[q])
                np.testing.assert_array_equal(z['party'], party[q])
                v131[q], nine[q], original[q] = z['v131'], z['nine'], z['original_cert']
                low_nine[q], high_nine[q] = z['nine_on_low_branch'], z['nine_on_high_branch']
        resolved = (v131 >= 0) & (nine >= 0)
        free = resolved & (v131 == native_sign) & (nine == v131)
        same = ((archive >= .5) == nine[:, None]).all(1)
        for name, p in answer.items():
            pred = np.where(free, p, nine)
            lo, hi = edit_effect_bounds(p, native_sign, v131, low_nine, high_nine, y)
            assert (lo[resolved] == hi[resolved]).all()
            np.testing.assert_array_equal(lo[resolved], (pred[resolved] == y[resolved]).astype(int) - (nine[resolved] == y[resolved]))
            for scope, population in [('all_three_folds', evaluated), ('all_parties_three_folds', evaluated & party)]:
                base[name][scope] = dict(rows=int(population.sum()), unresolved_impacts=int((population & (lo != hi)).sum()),
                    net_min=int(lo[population].sum()), net_max=int(hi[population].sum()),
                    min_per_fold=[int(lo[population & (fold == f)].sum()) for f in OUTER],
                    max_per_fold=[int(hi[population & (fold == f)].sum()) for f in OUTER])
            for key, mask in [('resolved',resolved), ('party',resolved & party), ('solo',resolved & ~party),
                              ('unanimous_party',resolved & party & same), ('previously_resolved', original)]:
                base[name][key] = score(pred, nine, y, mask, fold, groups)
    path = REPORT if release else OUT / 'native_screen.json'
    path.write_text(json.dumps(base, indent=2) + '\n')
    key = 'party' if release else 'vs_native_party'
    print({name:(v[key]['fixed'], v[key]['harmed']) for name, v in base.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', action='store_true')
    with threadpool_limits(1):
        run(parser.parse_args().release)
