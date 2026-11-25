"""Fit correction weights on old validation passengers and test on entirely different bookings."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from audit_repeated_pair_cases_n25 import near_scores, TAG
from broad_booking_policy_replay_n23 import interval
from full_roster_conditional_pool_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
DEST = ROOT / 'reports/nov23_matched_learning_curve/new_order_error_transfer.json'
FEES = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


def odds(p):
    return logit(p.clip(.002, .998))


def inputs(tag, rows, folds, raw, y, groups, native):
    logits = np.empty((len(rows), 4))
    generations = np.empty((len(rows), 2), bool)
    pairs = [near_scores((kind, 'historical' if tag == 'historical' else 'full'), rows, folds, groups, y)
             for kind in (TAG, TAG + '_repeat')]
    for f in np.unique(folds):
        ix = folds == f
        for folder in ('nov25_full_context_generations', 'nov25_shared_booking_state'):
            with np.load(ART / folder / f'{tag}_{f}.npz') as z:
                np.testing.assert_array_equal(z['q'], rows[ix])
                np.testing.assert_array_equal(z['labels'], y[z['t']])
                if tag == 'historical':
                    assert np.isin(z['t'], rows).all()
                else:
                    assert not np.intersect1d(groups[z['t']], groups[rows[ix]]).size
                if 'states' in z:
                    p = expit(z['states'].astype(float)).mean(axis=(1, 3))
                    logits[ix, 3] = odds(p[1]) - odds(p[0])
                else:
                    generations[ix] = np.column_stack([z['v1'] >= .5, z['v2'] >= .5])
                    logits[ix, 0] = odds(z['v1']) - odds(native[ix])
                    logits[ix, 1] = odds(z['v2']) - odds(native[ix])
    logits[:, 2] = odds(.5 * (pairs[0] + pairs[1])) - odds(native)
    bills = raw.iloc[rows][FEES]
    size = raw.groupby(groups).PassengerId.transform('size').to_numpy()[rows]
    context = np.column_stack([np.log(size), bills.gt(0).any(axis=1), raw.CryoSleep.eq(True).to_numpy()[rows]])
    # Votes encode proposed edits to the native model; release protection is applied when it can be verified.
    sign = native >= .5
    votes = np.column_stack([generations != sign[:, None], ((odds(native) + logits[:, 3] >= 0) != sign),
                             ((pairs[0] >= .5) != sign) & ((pairs[1] >= .5) != sign)])
    return logits, context, votes


def design(delta, context, varying):
    delta = delta.clip(-5, 5)
    if not varying:
        return delta
    return np.column_stack([delta, (delta[:, :, None] * context[:, None, :]).reshape(len(delta), -1)])


def fit_predict(x, z, native, outcome, query_native):
    # A fixed offset preserves the native probability when the expert evidence is zero.
    scale = x.std(0).clip(.25)
    a, b = x / scale, z / scale
    offset = odds(native)
    def loss(beta):
        eta = offset + a @ beta
        return np.sum(np.logaddexp(0, eta) - outcome * eta) + 40 * (beta @ beta), a.T @ (expit(eta) - outcome) + 80 * beta
    fit = minimize(loss, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B')
    assert fit.success, fit.message
    return odds(query_native) + b @ fit.x >= 0


def shuffle(data, p, folds, seed):
    # Within-confidence shuffle preserves rough class and confidence distributions.
    answer = data.copy()
    cells = 8 * folds + 4 * (p >= .5).astype(int) + np.digitize(abs(p - .5), [.1, .25, .4])
    rng = np.random.default_rng(seed)
    for k in np.unique(cells):
        where = np.flatnonzero(cells == k)
        answer[where] = data[rng.permutation(where)]
    return answer


def run():
    raw, y, groups, f8 = data()
    with np.load(ART / 'nov25_historical_nine/historical.npz') as z:
        rows, hist_fold, hist_inc, prev = [z[k] for k in ('rows', 'fold', 'nine', 'v131')]
    with np.load(ART / 'nov25_mitra_fp32_head/oof.npz') as z:
        np.testing.assert_array_equal(z['row'], rows)
        hp = z['native']
    hdelta, hctx, hvotes = inputs('historical', rows, hist_fold, raw, y, groups, hp)
    with np.load(ART / 'nov25_production_anchor_cpu/oof.npz') as z:
        np.testing.assert_array_equal(z['passenger'], raw.PassengerId)
        np.testing.assert_array_equal(z['fold'], f8)
        ep, einc, certified = z['v96'], z['certified_nine'].astype(bool), z['certified_v131_equals_v96']
    edelta, ectx, evotes = inputs('eight', np.arange(len(y)), f8, raw, y, groups, ep)
    q = np.flatnonzero(~np.isin(groups, groups[rows]))
    assert len(q) == 1044 and not np.intersect1d(groups[q], groups[rows]).size
    _, panels = prediction_sets(raw, y, groups, f8, rows, hist_fold)
    agree = ((np.column_stack(panels[0])[q] >= .5) == (ep[q, None] >= .5)).all(1)
    hist_agree = ((np.column_stack(panels[1]) >= .5) == hist_inc[:, None]).all(1)
    def evaluate(pred, mask, anchor, truth, f, orders):
        diff = pred != anchor
        effect = np.where(diff & mask, np.where(anchor != truth, 1, -1), 0)
        return dict(rows=int(mask.sum()), errors=int((mask & (anchor != truth)).sum()),
                    fixed=int((effect > 0).sum()), harmed=int((effect < 0).sum()),
                    net_interval_pp=interval(effect[mask], orders[mask]),
                    fold_net=[int(effect[f == k].sum()) for k in np.unique(f)])
    hfree = ((hp >= .5) == prev) & (prev == hist_inc)
    proposals = {}
    hist_proposals = {}
    for label, n in [('votes_three', 3), ('votes_two', 2)]:
        proposals[label] = (ep[q] >= .5) ^ (evotes[q].sum(1) >= n)
        hist_proposals[label] = (hp >= .5) ^ (hvotes.sum(1) >= n)
    for label, vary in [('pooled', False), ('by_context', True), ('shuffled_context', True)]:
        h, e = hdelta, edelta[q]
        if label == 'shuffled_context':
            h = shuffle(h, hp, hist_fold, 827)
            e = shuffle(e, ep[q], f8[q], 1712)
        x, test = design(h, hctx, vary), design(e, ectx[q], vary)
        proposals[label] = fit_predict(x, test, hp, y[rows], ep[q])
        oof = np.zeros(len(rows), bool)
        for k in np.unique(hist_fold):
            ix = np.flatnonzero(hist_fold == k)
            for a, b in GroupKFold(4, shuffle=True, random_state=801).split(ix, groups=groups[rows[ix]]):
                t, ask = ix[a], ix[b]
                assert not np.intersect1d(groups[rows[t]], groups[rows[ask]]).size
                oof[ask] = fit_predict(x[t], x[ask], hp[t], y[rows[t]], hp[ask])
        hist_proposals[label] = oof
    result = {}
    for name, pred in proposals.items():
        hist = np.where(hfree, hist_proposals[name], hist_inc)
        exact = np.where((ep[q] >= .5) == einc[q], pred, einc[q])
        result[name] = dict(
            historical=evaluate(hist, np.ones(len(rows), bool), hist_inc, y[rows], hist_fold, groups[rows]),
            historical_unanimous=evaluate(hist, hist_agree, hist_inc, y[rows], hist_fold, groups[rows]),
            new_orders_vs_native=evaluate(pred, np.ones(len(q), bool), ep[q] >= .5, y[q], f8[q], groups[q]),
            new_orders_native_unanimous=evaluate(pred, agree, ep[q] >= .5, y[q], f8[q], groups[q]),
            new_orders_exact_nine=evaluate(exact, certified[q], einc[q], y[q], f8[q], groups[q]))
    for name, n_votes in [('votes_three', 3), ('votes_two', 2)]:
        hard = (ep >= .5) ^ (evotes.sum(1) >= n_votes)
        protected = np.where((ep >= .5) == einc, hard, einc)
        result[name]['eight_exact_nine'] = evaluate(protected, certified, einc, y, f8, groups)
        result[name]['eight_vs_native'] = evaluate(hard, np.ones(len(y), bool), ep >= .5, y, f8, groups)
    DEST.write_text(json.dumps(dict(training_rows=len(rows), query_rows=len(q), query_orders=len(np.unique(groups[q])),
                                   exact_nine_rows=int(certified[q].sum()), results=result), indent=2) + '\n')
    print([(k, v['historical']['fixed'] - v['historical']['harmed'], v['new_orders_vs_native']['fixed'],
            v['new_orders_vs_native']['harmed'], v['new_orders_exact_nine']['fixed'], v['new_orders_exact_nine']['harmed'])
           for k, v in result.items()])


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
