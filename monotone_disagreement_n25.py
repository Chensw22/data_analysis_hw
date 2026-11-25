"""Learn accuracy-optimal monotone decision regions across aligned passenger model probabilities."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import FAMILIES
from full_roster_conditional_pool_n23 import banks, data
from full_roster_six_failure_archive_n24 import prediction_sets
from broad_booking_policy_replay_n23 import interval

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/monotone_disagreement.json'
NATIVE_EDGES = np.array([.15, .3, .4, .46, .5, .54, .6, .7, .85])
OPINION_EDGES = np.array([.25, .4, .5, .6, .75])


def family_average(scores, labels):
    families = np.array([label.split('/')[0] for label in labels])
    return np.median(np.column_stack([np.median(scores[:, families == family], axis=1)
                                      for family in np.unique(families)]), axis=1)


def cell_ids(columns):
    edges = [NATIVE_EDGES] + [OPINION_EDGES] * (len(columns) - 1)
    shape = tuple(len(e) + 1 for e in edges)
    bins = [np.searchsorted(e, score, side='right') for score, e in zip(columns, edges, strict=True)]
    return np.ravel_multi_index(bins, shape), shape


def fit_region(cells, shape, truth):
    # A source-side cell predicts True. Infinite upward arcs enforce monotonicity.
    total = int(np.prod(shape))
    grid = np.arange(total).reshape(shape)
    centers = np.convolve(np.r_[0., NATIVE_EDGES, 1.], [.5, .5], mode='valid')
    # Four prior observations per cell at the native confidence, expressed in integer units.
    prior = 4 * (2 * centers - 1)
    reward = np.bincount(cells, weights=np.where(truth, 1, -1), minlength=total)
    reward += np.repeat(prior, total // shape[0])
    reward = np.rint(1000 * reward).astype(np.int64)
    source, sink = total, total + 1
    pos, neg = np.flatnonzero(reward > 0), np.flatnonzero(reward < 0)
    origins = [np.full(len(pos), source), neg]
    targets = [pos, np.full(len(neg), sink)]
    capacity = [reward[pos], -reward[neg]]
    inf = int(abs(reward).sum()) + 1
    for axis in range(len(shape)):
        lo = np.take(grid, np.arange(shape[axis] - 1), axis=axis).ravel()
        hi = np.take(grid, np.arange(1, shape[axis]), axis=axis).ravel()
        origins.append(lo)
        targets.append(hi)
        capacity.append(np.full(len(lo), inf))
    graph = coo_matrix((np.concatenate(capacity), (np.concatenate(origins), np.concatenate(targets))),
                       shape=(total + 2, total + 2)).tocsr()
    flow = maximum_flow(graph, source, sink)
    residual = graph - flow.flow
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    assert sink not in reachable
    decision = np.zeros(total, bool)
    decision[reachable[reachable < total]] = True
    assert all((np.diff(decision.reshape(shape).astype(int), axis=i) >= 0).all() for i in range(len(shape)))
    assert reward[decision].sum() == reward[pos].sum() - flow.flow_value
    return decision


def predict(split, cells, shape, y, groups):
    pred = np.zeros(len(y), bool)
    for f in np.unique(split):
        outer = np.flatnonzero(split == f)
        for a, b in GroupKFold(4, shuffle=True, random_state=814).split(outer, groups=groups[outer]):
            t, q = outer[a], outer[b]
            assert not np.intersect1d(groups[t], groups[q]).size
            pred[q] = fit_region(cells[t], shape, y[t])[cells[q]]
    return pred


def effect(pred, base, y, mask, fold, g):
    d = (pred == y).astype(int) - (base == y)
    return dict(rows=int(mask.sum()), original_errors=int((mask & (base != y)).sum()),
                corrected=int((d[mask] > 0).sum()), harmed=int((d[mask] < 0).sum()),
                delta_interval_pp=interval(d[mask], g[mask]),
                fold_net=[int(d[mask & (fold == f)].sum()) for f in np.unique(fold)])


def views(p, archive, names):
    base = p[:, 4]
    tree = np.median(p[:, 1:4], axis=1)
    four = np.median(p[:, :4], axis=1)
    broad = family_average(archive, names)
    return {'native_only': [base], 'strong_four': [base, four],
            'neural_and_trees': [base, p[:, 0], tree], 'strong_and_archive': [base, four, broad]}


def run():
    raw, y, g, f8 = data()
    with np.load(ART / 'nov25_historical_nine/historical.npz') as z:
        rows, fh, hp, hb, previous = [z[k] for k in ('rows', 'fold', 'scores', 'nine', 'v131')]
        np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[rows])
    alt_names, panels = prediction_sets(raw, y, g, f8, rows, fh)
    labels = ['response/' + str(j) for j in range(3)] + ['projection/' + str(j) for j in range(3)] + alt_names
    archive_e, archive_h = [np.column_stack(v) for v in panels]
    names, _, bank = banks(raw, y, f8)
    ep = bank[:, [names.index(k) for k in (*FAMILIES, 'mitra96')]]
    with np.load(ART / 'nov25_production_anchor_cpu/oof.npz') as z:
        np.testing.assert_array_equal(z['v96'], ep[:, 4])
        np.testing.assert_array_equal(z['fold'], f8)
        eb, cert = z['certified_nine'].astype(bool), z['certified_v131_equals_v96']
    hbase, ebase = hp[:, 4] >= .5, ep[:, 4] >= .5
    hfree = (hbase == previous) & (previous == hb)
    new = ~np.isin(g, g[rows])
    assert new.sum() == 1044 and (new & cert).sum() == 841
    ha = ((archive_h >= .5) == hb[:, None]).all(1)
    ea = ((archive_e >= .5) == ebase[:, None]).all(1)
    historic, eight = views(hp, archive_h, labels), views(ep, archive_e, labels)
    out = {}
    for name in historic:
        hc, shape = cell_ids(historic[name])
        ec, eshape = cell_ids(eight[name])
        assert shape == eshape
        ph = predict(fh, hc, shape, y[rows], g[rows])
        pe = predict(f8, ec, shape, y, g)
        rh = np.where(hfree, ph, hb)
        re = np.where(ebase == eb, pe, eb)
        items = dict(historical=effect(rh, hb, y[rows], np.ones(len(rows), bool), fh, g[rows]),
                     historical_disputed=effect(rh, hb, y[rows], ~ha, fh, g[rows]),
                     historical_unanimous=effect(rh, hb, y[rows], ha, fh, g[rows]),
                     eight_vs_native=effect(pe, ebase, y, np.ones(len(y), bool), f8, g),
                     eight_vs_nine=effect(re, eb, y, cert, f8, g))
        # Evaluate the same new queries with (a) smaller-context historic training
        # and (b) same-context calibration orders inside each query's own outer fold.
        old = fit_region(hc, shape, y[rows])[ec]
        same = np.zeros(len(y), bool)
        for f in np.unique(f8):
            q = np.flatnonzero(new & (f8 == f))
            t = rows[f8[rows] == f]
            assert not np.intersect1d(g[t], g[q]).size
            same[q] = fit_region(ec[t], shape, y[t])[ec[q]]
        for source, pred in [('historical_training', old), ('same_context_training', same)]:
            exact = np.where(ebase == eb, pred, eb)
            items[source] = dict(new_vs_native=effect(pred, ebase, y, new, f8, g),
                                 new_vs_nine=effect(exact, eb, y, new & cert, f8, g),
                                 disputed_vs_native=effect(pred, ebase, y, new & ~ea, f8, g),
                                 unanimous_vs_native=effect(pred, ebase, y, new & ea, f8, g))
        out[name] = items
        print(name, [(items[k]['corrected'], items[k]['harmed']) for k in ('historical', 'eight_vs_nine')],
              [(items[s]['new_vs_nine']['corrected'],items[s]['new_vs_nine']['harmed']) for s in ('historical_training','same_context_training')],flush=True)
    REPORT.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
