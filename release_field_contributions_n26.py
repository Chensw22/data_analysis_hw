"""Measure the incremental value of observable fields for the actual frozen release's errors."""
import json
from math import comb
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from full_roster_booking_validation_n23 import ROOT
from evaluate_nested_release_n26 import DECK, OUT, REVIEW, outcome_scores, panel_predictions, protocols, replay, sources, stages
from price_implied_deck_n26 import FEES

NAMES = ['amount', 'allocation', 'location', 'traveller']
REPORT = 'reports/nov23_matched_learning_curve/release_field_contributions.json'


def blocks(raw, groups):
    fees = raw[FEES]
    total = fees.sum(axis=1, min_count=1)
    amount = pd.DataFrame(dict(total=np.log1p(total), positive=fees.gt(0).sum(axis=1),
                               missing=fees.isna().sum(axis=1)))
    allocation = fees.div(total.where(total > 0), axis=0)
    location = pd.get_dummies(raw[['Deck', 'Side', 'HomePlanet', 'Destination']], dtype=float)
    location['cabin'] = raw.CabinNumber
    location['booking'] = groups.astype(int)
    traveller = pd.get_dummies(raw[['VIP', 'CryoSleep']], dtype=float)
    traveller['age'] = raw.Age
    traveller['party'] = raw.booking_size
    return [x.to_numpy(float) for x in (amount, allocation, location, traveller)]


def evidence(v, scores, pred, idx, risk, enabled):
    anchor, review, fare, parent = stages(v, pred, idx, enabled, risk)
    p = scores[idx]
    agree = ((p >= .5) == parent[:, None]).all(axis=1)
    signed = logit(p.clip(.001, .999)) * np.where(parent, -1, 1)[:, None]
    native = v['off'][idx] * np.where(anchor == parent, 1, -1)
    base = np.column_stack([parent, agree, native, np.quantile(signed, [.05,.25,.5,.75,.95], axis=1).T,
                            (signed > 0).mean(axis=1), anchor != review, review != fare, fare != parent, risk])
    return parent, agree, base


def measures(risk, wrong, ids):
    decision = risk > .5
    high = np.lexsort((ids, -risk))[:int(np.ceil(len(ids) * .1))]
    return dict(fixed=int((decision & wrong).sum()), harmed=int((decision & ~wrong).sum()),
                log_loss=float(-np.where(wrong, np.log(risk), np.log1p(-risk)).mean()),
                auc=float(roc_auc_score(wrong, risk)), top_tenth_errors=int(wrong[high].sum()))


def summarize(items):
    ids, parent, agree, truth, predictions = [np.concatenate(v) for v in zip(*items, strict=True)]
    assert len(ids) == len(np.unique(ids)) and predictions.shape == (len(ids), 16)
    wrong = parent != truth
    report = {}
    for name, mask in [('all', np.ones(len(ids), bool)), ('unanimous', agree), ('contested', ~agree)]:
        scores = [measures(p[mask], wrong[mask], ids[mask]) for p in predictions.T]
        marginal = {}
        for j, field in enumerate(NAMES):
            loss = net = 0.
            for subset in range(16):
                if subset & (1 << j):
                    continue
                before, after = scores[subset], scores[subset | (1 << j)]
                weight = 1 / (4 * comb(3, subset.bit_count()))
                loss += weight * (before['log_loss'] - after['log_loss'])
                net += weight * ((after['fixed'] - after['harmed']) - (before['fixed'] - before['harmed']))
            marginal[field] = dict(log_loss_reduction=loss, net_correct_gain=net)
        np.testing.assert_allclose(sum(v['log_loss_reduction'] for v in marginal.values()),
                                   scores[0]['log_loss'] - scores[15]['log_loss'], atol=1e-12)
        active = (predictions[mask, 15] > .5) & (predictions[mask, 0] <= .5)
        retired = (predictions[mask, 15] <= .5) & (predictions[mask, 0] > .5)
        report[name] = dict(rows=int(mask.sum()), wrong=int(wrong[mask].sum()), marginal=marginal,
                            subsets=scores, new_fixed=int((active & wrong[mask]).sum()),
                            new_harmed=int((active & ~wrong[mask]).sum()),
                            retired_fixes=int((retired & wrong[mask]).sum()),
                            retired_harms=int((retired & ~wrong[mask]).sum()))
    return report


def run():
    raw, y, g, panels = sources()
    inputs = blocks(raw, g)
    p = {tag: panel_predictions(v, tag) for tag, v in panels.items()}
    scores = {tag: outcome_scores(v, p[tag]) for tag, v in panels.items()}
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled = z['eligible']
    collected = {}
    for key, src, dst, t, q in protocols(panels, g):
        a, b = src['ids'][t], dst['ids'][q]
        assert not np.isin(g[a], g[b]).any()
        with np.load(OUT/f'{key}.npz') as z:
            np.testing.assert_array_equal(z['ids'], a)
            np.testing.assert_array_equal(z['outer'], b)
            train_risk = z['risk']
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'], b)
            query_risk = np.column_stack([z['identities_error'], z['identities_outcome']])
        old, _, xt = evidence(src, scores['historical'], p['historical'], t, train_risk, enabled[a])
        current, agree, xq = evidence(dst, scores['full'], p['full'], q, query_risk, enabled[b])
        np.testing.assert_array_equal(current, replay(key, q, dst, p['full'], enabled))
        risk = []
        for subset in range(16):
            selected = [field for j, field in enumerate(inputs) if subset & (1 << j)]
            tr = np.column_stack([xt, *[field[a] for field in selected]])
            te = np.column_stack([xq, *[field[b] for field in selected]])
            model = HistGradientBoostingClassifier(max_iter=65, max_leaf_nodes=5, min_samples_leaf=24,
                       l2_regularization=6, learning_rate=.045, early_stopping=False, random_state=821)
            risk.append(model.fit(tr, old != y[a]).predict_proba(te)[:, 1])
        tag = 'bookingmix' if key.startswith('bookingmix') else key
        collected.setdefault(tag, []).append((b, current, agree, y[b], np.column_stack(risk)))
        print(key, flush=True)
    report = {k: summarize(items) for k, items in collected.items()}
    assert report['bookingmix']['unanimous']['wrong'] == 136 and report['bookingmix']['all']['wrong'] == 1040
    (ROOT/REPORT).write_text(json.dumps({'field_bit_order': NAMES, 'protocols': report}, indent=2) + '\n')
    for tag, result in report.items():
        print(tag, {k: (v['subsets'][0], v['subsets'][15], v['marginal']) for k,v in result.items()}, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
