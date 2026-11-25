"""Standardize held release accuracy and gain to passenger-weighted test booking sizes."""
import json

import numpy as np
import pandas as pd

from held_expert_portfolios_n26 import DECK, REVIEW, ROOT, SIX, fare_model, outcome_scores
from held_expert_portfolios_n26 import panel_predictions, six_model, sources, splits


def summarize(rows, parent, child, unanimous, truth):
    err = child[rows] != truth[rows]
    old = parent[rows] != truth[rows]
    return dict(rows=len(rows), errors=int(err.sum()), unanimous=int(unanimous[rows].sum()),
                unanimous_errors=int((unanimous[rows] & err).sum()),
                fixed=int((old & ~err).sum()), harmed=int((~old & err).sum()))


def standardize(rows, size, target, group, truth, parent, child, cap):
    bands = np.minimum(size, cap)
    weights = np.bincount(np.minimum(target, cap), minlength=cap+1)[1:] / len(target)
    draws = np.zeros((2000, 2))
    point = np.zeros(2)
    rng = np.random.default_rng(546)
    for band, weight in enumerate(weights, 1):
        if weight == 0:
            continue
        ids = rows[bands[rows] == band]
        if not len(ids):
            return dict(unsupported_test_booking_size=band)
        labels = truth[ids]
        right = (child[ids] == labels).astype(float)
        delta = right - (parent[ids] == labels)
        _, ix = np.unique(group[ids], return_inverse=True)
        counts = np.bincount(ix)
        totals = np.array([np.bincount(ix, weights=a) for a in (right,delta)]).T
        point += weight * totals.sum(0)/counts.sum()
        for j in range(0,len(draws),100):
            sampled = rng.integers(len(counts),size=(100,len(counts)))
            draws[j:j+100] += weight * totals[sampled].sum(1)/counts[sampled].sum(1)[:,None]
    return {key:dict(point_pct=100*point[j],cluster_95_pct=(100*np.quantile(draws[:,j],[.025,.975])).tolist())
            for j,key in enumerate(('accuracy','gain'))}


def run():
    _,y,g,panels = sources()
    v = panels['full']
    np.testing.assert_array_equal(v['ids'],np.arange(len(y)))
    test = pd.read_csv(ROOT/'data/raw/test.csv')
    tg = test.PassengerId.str.split('_').str[0].to_numpy()
    groups = pd.Series(np.r_[g,tg])
    sizes = groups.map(groups.value_counts()).to_numpy()
    size,target = sizes[:len(y)],sizes[len(y):]
    train_counts = groups.map(pd.Series(g).value_counts()).fillna(0).to_numpy(int)
    assert not np.isin(tg,g).any()
    with np.load(ROOT/'artifacts/nov26_competition_expert_panel/scores.npz') as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        tu = ((z['scores']>=.5)==z['prediction'][:,None]).all(1)
    p = panel_predictions(v,'full')
    votes = outcome_scores(v,p)>=.5
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled = z['eligible']
    report = dict(test={},scopes={},booking_overlap_with_training=int(np.isin(tg,g).sum()),
                  note="Held-out decisions exclude labels of the entire query booking. Training roster companions denote labels available in the source file only; test bookings have no training companions. Intervals resample bookings within size bands and condition on existing fitted predictions.")
    for s in np.unique(target):
        selected = target==s
        report['test'][str(s)] = dict(rows=int(selected.sum()),unanimous=int((selected&tu).sum()),
            with_training_companion=int((selected&(train_counts[len(y):]>0)).sum()))
    for tag in ('full','bookingmix'):
        used = np.zeros(len(y),bool)
        parent,child = used.copy(),used.copy()
        for key,src,dst,t,q in splits(panels,g):
            if not key.startswith(tag+'_'):
                continue
            assert not used[q].any()
            used[q] = True
            with np.load(REVIEW/f'pred_{key}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                review = v['anchor'][q]^((z['identities_error']>.5)&(z['identities_outcome']>.5))
            fares = fare_model(review,p['neural_deck_raw'][q],p['neural_deck_fine'][q],enabled[q])
            parent[q] = fares>=.5
            child[q] = six_model(fares,np.column_stack([p[n][q] for n in SIX]),v['votes'][q,:18],enabled[q])>=.5
        if tag=='full':
            np.testing.assert_array_equal(used,v['known'])
        ids = np.flatnonzero(used)
        unanimous = (votes==child[:,None]).all(1)
        record = dict(all=summarize(ids,parent,child,unanimous,y),by_size={})
        for s in np.unique(size):
            here = ids[size[ids]==s]
            record['by_size'][str(s)] = dict(**summarize(here,parent,child,unanimous,y),
                with_other_label_in_source_file=int((train_counts[here]>1).sum()),other_label_in_fit=0)
        for cap in (4,int(max(sizes))):
            record[f'standardized_size_cap_{cap}'] = standardize(ids,size,target,g,y,parent,child,cap)
        report['scopes'][tag] = record
    assert [(r['all']['rows'],r['all']['fixed'],r['all']['harmed']) for r in report['scopes'].values()] == [(8532,25,8),(6055,19,7)]
    (ROOT/'reports/nov23_matched_learning_curve/release_booking_composition.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,r in report['scopes'].items():
        print(name, r['all'],r['standardized_size_cap_4'])


if __name__=='__main__':
    run()
