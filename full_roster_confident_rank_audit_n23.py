"""Audit expert ordering in confidently called passengers and select experts on disjoint bookings."""
import json

import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from broad_booking_policy_replay_n23 import interval
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_same_model_calibration_n23 import learn
from full_roster_unexpected_peers_n23 import reference_score


def centered_auc(pos, neg):
    # Agreement over case-control pairs; half a point for exact ties.
    ordered = np.sort(neg, axis=0)
    n = len(pos)*len(neg)
    if not n:
        return np.zeros(pos.shape[1]), 0
    wins = np.array([.5*(np.searchsorted(ordered[:,j],pos[:,j],side='left').sum()
                         + np.searchsorted(ordered[:,j],pos[:,j],side='right').sum())
                     for j in range(pos.shape[1])])
    return wins-.5*n, n


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    ref = latest_reference(y,group,fold,names,half,full)
    p = reference_score(full,ref,names.index('balanced_full'))
    earth = (raw.HomePlanet.eq('Earth') & raw.CryoSleep.eq(True)).to_numpy()
    route = raw.Destination.eq('TRAPPIST-1e').to_numpy()
    side = raw.Cabin.str.endswith(('/S','/P'),na=False).to_numpy()
    low, high = p <= .3, p >= .7
    focus = earth & route & side & low
    assert focus.sum() == 121 and y[focus].sum() == 50
    masks = {'focus':focus, 'other_low':low & ~focus, 'all_high':high}
    scores = np.column_stack([p, full])
    names = ['reference',*names]
    contrast = logit(scores.clip(.002,.998)) - logit(p.clip(.002,.998))[:,None]
    cohorts = np.where(earth & route,2,earth.astype(int))
    table = {}
    for label, mask in masks.items():
        table[label] = dict(n=int(mask.sum()), actual_true=int(y[mask].sum()),
                           auc=dict(zip(names,[float(roc_auc_score(y[mask],s[mask])) for s in scores.T],strict=True)))
    # The transfer policy is evaluated on all confident calls. Comparable donors
    # have the same reference direction; matching Earth/sleep and route refines locality.
    policies = {k: p.copy() for k in ['pooled_best','local_best','local_average']}
    selected = {k:{} for k in policies if k != 'local_average'}
    for f in np.unique(fold):
        outer = np.flatnonzero(fold == f)
        for a,b in GroupKFold(4,shuffle=True,random_state=462).split(outer,groups=group[outer]):
            donors, queries = outer[a], outer[b]
            assert not np.isin(group[donors],group[queries]).any()
            for mask in [low,high]:
                pool = donors[mask[donors]]
                q_all = queries[mask[queries]]
                wins, total = centered_auc(scores[pool[y[pool]]],scores[pool[~y[pool]]])
                if not total:
                    continue
                # Shrink local concordance by 36 on-scale positive/negative pairs.
                pooled = wins/total
                for category in [0,1,2]:
                    t = pool[cohorts[pool] == category]
                    q = q_all[cohorts[q_all] == category]
                    if not len(q):
                        continue
                    local, n = centered_auc(scores[t[y[t]]],scores[t[~y[t]]])
                    local = (local + 36*pooled)/(n+36)
                    for mode in policies:
                        evidence = pooled if mode == 'pooled_best' else local
                        if mode == 'local_average':
                            weights = np.maximum(evidence,0)
                            weights[0] = 0
                            if not weights.sum():
                                continue
                            weights /= weights.sum()
                        else:
                            best = int(np.argmax(evidence))
                            selected[mode][names[best]] = selected[mode].get(names[best],0)+len(q)
                            if best == 0:
                                continue
                            weights = np.eye(len(names))[best]
                        # Fit change relative to the incumbent using a probability contrast.
                        z = contrast @ weights
                        train = t if len(t)>=20 and np.unique(y[t]).size == 2 else pool
                        w = learn(z[train,None],y[train],logit(p[train].clip(.001,.999)),np.zeros(1))
                        policies[mode][q] = expit(logit(p[q].clip(.001,.999))+w[0]*z[q])
    results = {}
    for key, prob in policies.items():
        delta = ((prob>=.5)==y).astype(int) - (ref==y)
        results[key] = dict(accuracy=float(((prob>=.5)==y).mean()),fixed=int((delta==1).sum()),
            harmed=int((delta==-1).sum()), focus_auc=float(roc_auc_score(y[focus],prob[focus])),
            focus_fixed=int((delta[focus]==1).sum()), focus_harmed=int((delta[focus]==-1).sum()),
            all_common_fixed=int(((delta==1) & ((full>=.5)!=y[:,None]).all(1)).sum()),
            fold_net=[int(delta[fold==f].sum()) for f in np.unique(fold)], ci_pp=interval(delta,group))
    report = dict(descriptive=table, results=results, selected_passengers=selected)
    (ROOT/'reports/nov23_matched_learning_curve/confident_rank_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    for k, entry in table.items():
        print(k, entry['n'], entry['actual_true'], sorted(entry['auc'].items(),key=lambda x:-x[1])[:3])
    print(results)


if __name__ == '__main__':
    run()
