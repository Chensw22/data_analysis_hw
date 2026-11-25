"""Separate risk estimation from ranking similar passengers across bookings, without seeing pair outcomes."""
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from evaluate_nested_release_n26 import protocols
from frozen_queue_shared_errors_n26 import paired
from held_expert_portfolios_n26 import sources
from localized_opposite_order_pairs_n26 import coordinates
from manifest_sleep_contrast_n26 import design
from outside_panel_coverage_n26 import archives
from price_implied_deck_n26 import FEES
from target_mix_expert_routing_n26 import ROOT, ticker


def pair(rows, group, x, seed):
    # Random sides determine which pairs are possible; the optimal assignment minimizes
    # manifest distance. No target or model predictions enter matching.
    shuffled = np.random.default_rng(seed).permutation(rows)
    a, b = shuffled[::2], shuffled[1::2]
    dist = cdist(x[a],x[b],'sqeuclidean')
    same = group[a,None] == group[b]
    dist[same] = (dist.max()+1) * (len(rows)+1)
    i,j = linear_sum_assignment(dist)
    keep = ~same[i,j]
    return a[i[keep]], b[j[keep]]


def ordering(x, y, a, b):
    diff = x[a] - x[b]
    use = y[a] != y[b]
    if not use.any():
        return np.zeros(x.shape[1])
    dx, labels = diff[use], y[a[use]]
    model = LogisticRegression(C=.08,fit_intercept=False,max_iter=400)
    model.fit(np.vstack([dx,-dx]),np.r_[labels,~labels])
    return model.coef_[0]


def forecasts(raw, y, g, f):
    others, _ = archives(raw,y,g,f,True)
    scores = [others[n] for n in ('Mitra2_adapted','LimiX2','TabDPT')]
    for folder, pattern, field in [('nov26_original_tabstar','tuned','finetuned'),
                                   ('nov26_ordinal_factor_density','fold','conditional_bills')]:
        out = np.full(len(y),np.nan)
        for k in np.unique(f):
            with np.load(ROOT/'artifacts'/folder/f'{pattern}{k}.npz') as z:
                t,q = z['t'],z['q']
                np.testing.assert_array_equal(t,np.flatnonzero(f!=k))
                np.testing.assert_array_equal(q,np.flatnonzero(f==k))
                np.testing.assert_array_equal(z['train_y'],y[t])
                np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[q])
                assert not np.isin(g[t],g[q]).any()
                out[q] = z[field]
        scores.append(out)
    return logit(np.column_stack(scores).clip(.002,.998))


def run():
    raw, y, g, f, known, versions = paired()
    logits = forecasts(raw,y,g,f)
    match_fields = np.column_stack([design(raw,g),np.log1p(raw[FEES].to_numpy(float))])
    paid = raw[FEES].gt(0).any(axis=1).to_numpy()
    chunks = {'sleep_no_positive':raw.CryoSleep.eq(True).to_numpy() & ~paid,
              'awake_with_spend':raw.CryoSleep.eq(False).to_numpy() & paid}
    panels = sources()[-1]
    result = {}
    for cohort, eligible in chunks.items():
        baseline, recalibrated, seen = np.zeros(len(y)),np.zeros(len(y)),np.zeros(len(y),bool)
        forced = {n: [] for n in ('all', 'all_versions_usual_both')}
        for key, _, _, t, q in protocols(panels, g, include_full=True):
            if not key.startswith('full_'):
                continue
            t,q = t[eligible[t]],q[eligible[q]]
            assert not np.isin(g[t],g[q]).any() and np.unique(f[np.r_[t,q]]).size == 1
            geom = coordinates(match_fields,t)
            x = StandardScaler().fit(logits[t]).transform(logits).clip(-5,5)
            baseline[q] = LogisticRegression(C=.15,max_iter=400).fit(x[t],y[t]).predict_proba(x[q])[:,1]
            recalibrated[q] = baseline[q]
            # Supply multiple independently paired training comparisons, averaging
            # independently fitted coefficients so repeated people are not counted as extra training rows.
            beta = np.mean([ordering(x,y,*pair(t,g,geom,seed)) for seed in (1403,2159,3217)],axis=0)
            a,b = pair(q,g,geom,3917)
            pa,pb = baseline[a],baseline[b]
            both = pa*pb
            one = pa*(1-pb)+pb*(1-pa)
            rank = expit((x[a]-x[b])@beta)
            recalibrated[a],recalibrated[b] = both+one*rank,both+one*(1-rank)
            np.testing.assert_allclose(recalibrated[a]+recalibrated[b],pa+pb)
            # Label-restricted calculation below is a diagnostic, after estimating all deployable scores.
            opposite = y[a]!=y[b]
            usual = cohort == 'sleep_no_positive'
            both_usual = (versions[a] == usual).all(1) & (versions[b] == usual).all(1)
            for key,mask in [('all',opposite),('all_versions_usual_both',opposite & both_usual)]:
                for sa,sb,target in zip(pa[mask]-pb[mask],rank[mask]-.5,y[a[mask]],strict=True):
                    forced[key].append([float(.5 if v==0 else (v>0)==target) for v in (sa,sb)])
            assert not seen[q].any()
            seen[q] = True
        np.testing.assert_array_equal(seen,known & eligible)
        parent = versions[:,-1]
        common = seen & (versions == parent[:,None]).all(1)
        ordinary = common & (parent == (cohort=='sleep_no_positive'))
        new = ~np.isin(g,g[panels['historical']['ids']])
        left = (baseline>=.5)!=parent
        right = (recalibrated>=.5)!=parent
        info = {'ordering': {key: {'opposite_pairs': len(values),
                                   'correct': np.sum(values,axis=0).tolist(),
                                   'accuracy': np.mean(values,axis=0).tolist() if values else []}
                             for key,values in forced.items()}}
        for n,action in [('direct',left),('paired',right),('added_by_pairing',right & ~left)]:
            info[n] = {m:ticker(mask,y!=parent,action) for m,mask in
                       [('all',seen),('all_versions',common),('usual_direction',ordinary),('new_orders',ordinary & new)]}
        result[cohort] = info
        print(cohort,info['ordering'],info['paired']['usual_direction'],
              info['added_by_pairing']['usual_direction'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/matched_case_ordering.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
