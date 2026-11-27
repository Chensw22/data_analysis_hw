"""Cross-fit outcome likelihood ratios of recurrent integer charges at each of five facilities."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT, data
from price_implied_deck_n26 import FEES, build_features
from repeated_archive_misses_n26 import ART, cohort, count

OUT = ART / 'nov27_facility_integer_evidence'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/facility_integer_evidence.json'


def count_window(grid, center, radius):
    return np.searchsorted(grid, center + radius, side='right') - np.searchsorted(grid, center - radius)


def peak_evidence(bills, keys, y, groups, bank, ask):
    """Estimate local P(class | amount, facility, registry) at wide, narrow and exact prices.

    Smooth with wider windows and lower-order strata. The difference in log odds is the
    relative sharp class-conditional density, with broad price levels removed.
    """
    assert not np.isin(groups[bank], groups[ask]).any()
    out = np.zeros((len(ask), 2, len(FEES)))
    for j, amounts in enumerate(bills.T):
        col = amounts[ask]
        index = np.flatnonzero(col > 0)
        v = col[index]
        # Preserve the positive 32-unit price band; rotate the exact within-band price by 13.
        centers = np.column_stack([v, 1 + ((v-1)//32)*32 + (v-1+13)%32])
        bank_paid = bank[amounts[bank] > 0]
        prior = (1 + y[bank_paid].sum()) / (2 + len(bank_paid))
        wide = np.full(centers.shape, prior)
        sharp = np.full(centers.shape, prior)
        for level in keys:
            for key in np.unique(level[ask[index]]):
                select = level[ask[index]] == key
                use = bank_paid[level[bank_paid] == key]
                if not len(use):
                    continue
                point = centers[select]
                grid = np.sort(amounts[use]); positive = np.sort(amounts[use[y[use]]])
                w, p = count_window(grid, point, 24), count_window(positive, point, 24)
                background = (p + 14*wide[select]) / (w + 14)
                # Carry the parent peak odds ratio as a shrunk prior for this finer registry cell.
                prior_peak = expit(logit(background) + .65*(logit(sharp[select]) - logit(wide[select])))
                w, p = count_window(grid, point, 2), count_window(positive, point, 2)
                local = (p + 9*prior_peak) / (w + 9)
                w, p = count_window(grid, point, 0), count_window(positive, point, 0)
                sharp[select] = (p + 6*local) / (w + 6)
                wide[select] = background
        out[index, :, j] = logit(sharp) - logit(wide)
    return out


def fit_tree(x, y, q):
    model = lgb.LGBMClassifier(n_estimators=270, learning_rate=.035, num_leaves=21, min_child_samples=35,
                               reg_lambda=4, force_col_wise=True, n_jobs=1, verbosity=-1, random_state=517)
    model.fit(x, y)
    return model.predict_proba(q)[:, 1]


def train(x, bills, keys, y, groups, train, query):
    cross = np.zeros((len(train), 2, len(FEES)))
    inner = GroupKFold(4, shuffle=True, random_state=4571)
    for t, q in inner.split(train, groups=groups[train]):
        cross[q] = peak_evidence(bills, keys, y, groups, train[t], train[q])
    external = peak_evidence(bills, keys, y, groups, train, query)
    outputs = {'raw': fit_tree(x[train], y[train], x[query]), 'likelihood': external}
    for j, key in enumerate(['price', 'displaced']):
        a, b = cross[:, j], external[:, j]
        outputs[key] = fit_tree(np.column_stack([x[train], a, a.sum(1)]), y[train],
                                np.column_stack([x[query], b, b.sum(1)]))
    return outputs


def run():
    raw, y, g, random = data()
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=raw.columns)
    roster = pd.concat([raw, test], ignore_index=True)
    groups = roster.PassengerId.str.split('_').str[0].to_numpy()
    ids = roster.PassengerId.to_numpy(str)
    n = len(y)
    bills = roster[FEES].fillna(-1).to_numpy(int)
    paid = (bills>0).any(1)
    planet = roster.HomePlanet.fillna('?').to_numpy(str)
    deck = roster.Cabin.str[0].fillna('?').to_numpy(str)
    pdkey = np.char.add(np.char.add(planet, '/'), deck)
    pattern = np.where((bills>0).sum(1) == 1, '/one', '/many')
    keys = [np.repeat('*', len(ids)), planet, pdkey, np.char.add(pdkey, pattern)]
    x = pd.get_dummies(build_features(roster, groups)['fine'], dtype=float).to_numpy(dtype=float)
    _, blind, known, fresh = cohort(raw, y, random)
    with np.load(ART/'nov27_manifest_group_dro/aligned.npz') as z:
        np.testing.assert_array_equal(ids, z['passenger'])
        np.testing.assert_array_equal(random, z['fold'])
        ordered, anchor = z['ordered'], z['anchor']
    parent = pd.read_csv(ART/'proposal_n27_directional_foundation/submission.csv')
    np.testing.assert_array_equal(parent.PassengerId,ids[n:])
    np.testing.assert_array_equal(parent.Transported,anchor[n:])
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(ids,z['passenger'])
        risk = abs(logit(z['scores'][:,-3:].mean(1).clip(.001,.999)))
    native = np.where(anchor, risk, -risk)
    OUT.mkdir(exist_ok=True)
    report = {}; saved = {}
    for mode, fold in [('random', random), ('ordered', ordered)]:
        probs = {k: np.empty(len(ids)) for k in ['raw','price','displaced']}
        evidence = np.empty((len(ids),2,5))
        for j in range(9):
            t = np.flatnonzero(fold!=j) if j<8 else np.arange(n)
            q = np.flatnonzero(fold==j) if j<8 else np.arange(n,len(ids))
            path = OUT/(f'{mode}_{j}.npz' if j<8 else 'test.npz')
            if not path.exists():
                result = train(x,bills,keys,y,groups,t,q)
                temp = path.with_suffix('.tmp.npz')
                np.savez_compressed(temp,t=t,q=q,ids=ids[q],labels=y[t],**result)
                temp.replace(path)
            with np.load(path) as z:
                for k,v in [('q',q),('t',t),('ids',ids[q]),('labels',y[t])]:
                    np.testing.assert_array_equal(z[k],v)
                for k in probs: probs[k][q]=z[k]
                evidence[q]=z['likelihood']
            print(mode,j,flush=True)
        direct=probs['price']>=.5
        novelty=paid & (direct!=anchor) & ((probs['raw']>=.5)==anchor) & ((probs['displaced']>=.5)==anchor)
        lr=evidence.sum(2)
        delta=logit(probs['price'])-logit(probs['raw'])
        shifted=logit(probs['displaced'])-logit(probs['raw'])
        rules={
            'novelty_both_controls':novelty,
            'likelihood':paid & (((native+lr[:,0])>=0)!=anchor),
            'displaced_likelihood':paid & (((native+lr[:,1])>=0)!=anchor),
            'model_increment':paid & (((native+delta)>=0)!=anchor),
            'displaced_increment':paid & (((native+shifted)>=0)!=anchor)}
        # Primary policy: require both exact-price mechanisms to overturn the anchor, and the two displaced controls to abstain.
        rules['paired_confirmation']=rules['likelihood'] & rules['model_increment'] & ~rules['displaced_increment'] & ~rules['displaced_likelihood']
        scopes={'all':known, 'fresh':fresh&known, 'common':blind, 'false_positives':known&anchor[:n], 'false_negatives':known&~anchor[:n]}
        results={}
        for k,flag in rules.items():
            d=flag[:n]*np.where(anchor[:n]!=y,1,-1)
            results[k]={'test_changes':int(flag[n:].sum()),'folds':[int(d[known&(fold==b)].sum()) for b in range(8)],
                        'scopes':{s:count(flag[:n],m,anchor[:n]!=y) for s,m in scopes.items()}}
        report[mode]={'anchor_coverage':int(known.sum()),'models_correct_all':{k:int(np.sum((p[:n]>=.5)==y)) for k,p in probs.items()},'rules':results}
        diagnostics={}
        for j, facility in enumerate(FEES):
            mask=bills[:n,j]>0
            signed=np.where(y[mask],1.,-1.)
            coarse=logit(probs['raw'][:n][mask].clip(.001,.999))
            loss=lambda odds: float(np.logaddexp(0,-signed*odds).mean())
            err=mask & blind & (anchor[:n]!=y)
            diagnostics[facility]=dict(positive_train=int(mask.sum()),positive_test=int((bills[n:,j]>0).sum()),
                raw_heldout_logloss=loss(coarse),peak_heldout_logloss=loss(coarse+evidence[:n,0,j][mask]),
                displaced_heldout_logloss=loss(coarse+evidence[:n,1,j][mask]),common_errors=int(err.sum()),
                common_errors_correct_price_tree=int((err&(direct[:n]==y)).sum()),
                common_errors_correct_raw_tree=int((err&((probs['raw'][:n]>=.5)==y)).sum()))
        report[mode]['facilities']=diagnostics
        for k,v in rules.items():saved[f'{mode}_{k}']=v
        print(mode,results['paired_confirmation'],report[mode]['models_correct_all'],flush=True)
    for k in rules:np.testing.assert_array_equal(saved['random_'+k][n:],saved['ordered_'+k][n:])
    report['notes']=['Each of two price-model validation splits holds out whole orders; inner training embeddings exclude entire validation orders.',
                     'The same official parent replay and native margin, trained on the original randomized outer fold, are used in both comparisons. The alternative split independently refits the facility learner only.',
                     'The historical-outside subset is included in the full set. Sharp charge centers are compared to centers displaced by 13 units within their 32-unit band.']
    np.savez_compressed(OUT/'aligned.npz',ids=ids,anchor=anchor,random=random,ordered=ordered,**saved)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):run()
