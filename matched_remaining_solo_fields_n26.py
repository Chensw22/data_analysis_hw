"""Separate mistakes from correct solo countervotes matched on the observed opinions."""
import json

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from price_implied_deck_n26 import FEES
from solo_remaining_countervotes_n26 import context
from tabnet_attention_transfer_n26 import RESULT
from target_mix_expert_routing_n26 import ticker


def matched(rows, parent, opinion, wrong):
    cases, controls = [], []
    for label in (False, True):
        same = rows[parent[rows] == label]
        a, b = same[wrong[same]], same[~wrong[same]]
        if len(a) and len(b):
            i, j = linear_sum_assignment(cdist(opinion[a], opinion[b], 'cityblock'))
            cases.extend(a[i])
            controls.extend(b[j])
    return np.asarray(cases, dtype=int), np.asarray(controls, dtype=int)


def raw_fields(raw, parent):
    bill = raw[FEES]
    amount = bill.fillna(0).to_numpy()
    total = amount.sum(axis=1)
    share = amount / total.clip(1)[:, None]
    cash = np.column_stack([np.log1p(amount)/8, amount > 0, share, bill.isna(), np.log1p(total)/8])
    berth = raw.Cabin.str.split('/', expand=True)
    number = berth[1].astype(float).fillna(-1).to_numpy()/1800
    cabin = np.column_stack([*[berth[0].eq(v) for v in 'ABCDEFGT'], berth[0].isna(),
                             berth[2].eq('S'), number, np.maximum(number-.5,0)])
    # Match within direction; the same field can imply opposite signed error risk
    # for a predicted positive and a predicted negative.
    direction = np.where(parent, -1., 1.)[:, None]
    return {key: np.column_stack([block, block * direction]) for key, block in
            [('bills', cash), ('cabin', cabin), ('both', np.column_stack([cash, cabin]))]}


def contrast(x, a, b, c, q, shuffle, seed):
    differences = x[a] - x[b]
    if shuffle:
        differences = differences * np.random.default_rng(seed).choice([-1, 1], (len(a), 1))
    model = LogisticRegression(C=.3, fit_intercept=False, max_iter=300)
    model.fit(np.vstack([differences, -differences]), np.repeat([1, 0], len(a)))
    return model.decision_function(x[c]), model.decision_function(x[q])


def run():
    raw, _, groups, fold, parent, blocks, flags, wrong, masks = context()
    eligible = masks['full']
    assert eligible.sum() == 2448 and (eligible & wrong).sum() == 524
    fractions = np.column_stack([arr.mean(1) for arr in blocks.values()])
    opinion = np.column_stack([fractions, flags.astype(float)/2])
    base = np.column_stack([parent, opinion, fractions * parent[:, None]])
    fields = raw_fields(raw, parent)
    methods = ['opinions', 'bills', 'cabin', 'both', 'shuffled']
    prob = {n: np.full(len(raw), np.nan) for n in methods}
    eval_pairs = []
    for k in range(8):
        rows = np.flatnonzero(eligible & (fold == k))
        parts = [rows[v] for _,v in KFold(4,shuffle=True,random_state=9275+k).split(rows)]
        for j,q in enumerate(parts):
            # Each passenger's predicted risk averages three fits. No query order
            # is used for learning the paired direction or calibrating its risk.
            estimates = {n: [] for n in methods}
            for h,c in enumerate(parts):
                if h == j:
                    continue
                t = np.concatenate([v for m,v in enumerate(parts) if m not in (j,h)])
                assert not np.isin(groups[t], np.r_[groups[q],groups[c]]).any()
                assert not np.isin(groups[q], groups[c]).any()
                a,b = matched(t,parent,opinion,wrong)
                for n in methods:
                    xc, xq = base[c], base[q]
                    if n != 'opinions':
                        x = fields['both' if n == 'shuffled' else n]
                        sc, sq = contrast(x,a,b,c,q,n=='shuffled',219+k*16+j*4+h)
                        xc, xq = np.column_stack([xc,sc]), np.column_stack([xq,sq])
                    cal = LogisticRegression(C=.5,max_iter=250).fit(xc,wrong[c])
                    estimates[n].append(cal.predict_proba(xq)[:,1])
            for n in methods:
                prob[n][q] = np.mean(estimates[n],axis=0)
            eval_pairs.append(matched(q,parent,opinion,wrong))
    a,b = (np.concatenate([v[i] for v in eval_pairs]) for i in (0,1))
    assert (wrong[a] & ~wrong[b] & (parent[a]==parent[b])).all()
    same_flags = (flags[a] == flags[b]).all(1)
    report = {'out_of_fold_matched_pairs': len(a), 'matching_six_block_flags': int(same_flags.sum()),
              'views': {}, 'field_prevalence': {}}
    cabin = raw.Cabin.str.split('/', expand=True)
    observed = {'any_positive_bill': raw[FEES].gt(0).any(axis=1), 'spa_positive': raw.Spa.gt(0),
                'vr_positive': raw.VRDeck.gt(0), 'foodcourt_positive': raw.FoodCourt.gt(0),
                'deck_f': cabin[0].eq('F'), 'deck_g': cabin[0].eq('G'), 'starboard': cabin[2].eq('S')}
    for direction in (False,True):
        rows = same_flags & (parent[a] == direction)
        detail = {}
        for name, value in observed.items():
            field = value.to_numpy(dtype=int)
            difference = field[a] - field[b]
            detail[name] = dict(case_count=int(field[a[rows]].sum()), control_count=int(field[b[rows]].sum()),
                                fold_net_prevalence=[int(difference[rows & (fold[a]==k)].sum()) for k in range(8)])
        report['field_prevalence'][str(direction)] = dict(pairs=int(rows.sum()),observed=detail)
    for tag, scope in masks.items():
        use = scope & eligible
        paired = use[a] & use[b]
        summary = {}
        for n,p in prob.items():
            assert np.array_equal(np.isfinite(p),eligible)
            act = p > .5
            result = ticker(use,wrong,act)
            result['only_beyond_opinions'] = ticker(use,wrong,act & (prob['opinions']<=.5))
            result['auc'] = float(roc_auc_score(wrong[use],p[use]))
            gap = p[a[paired]] - p[b[paired]]
            result['paired_comparisons'] = int(len(gap))
            result['paired_accuracy'] = float(np.mean((gap>0) + .5*(gap==0)))
            exact = paired & same_flags
            exact_gap = p[a[exact]] - p[b[exact]]
            result['same_flags_pairs'] = len(exact_gap)
            result['same_flags_pair_accuracy'] = float(np.mean((exact_gap>0) + .5*(exact_gap==0))) if len(exact_gap) else None
            result['net_by_fold'] = [int((np.where(wrong,1,-1)*use*act*(fold==k)).sum()) for k in range(8)]
            summary[n] = result
        report['views'][tag] = summary
        print(tag,[(n,v['fixed'],v['harmed'],round(v['paired_accuracy'],3)) for n,v in summary.items()],flush=True)
    (RESULT.parent/'matched_remaining_solo_fields.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
