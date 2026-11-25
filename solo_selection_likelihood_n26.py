"""Factor the outcome of an individual booking into outcome odds and a booking-size likelihood."""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART, DEST, stats
from exaone_validation_data import CATEGORICAL, compact_features
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from outstanding_error_families_n26 import observed

OUT = ART / 'nov26_solo_selection_likelihood'


def regimes():
    raw, y, groups, fold = data()
    _, _, _, rows, names, p, h, native, old_native, ref = panels_and_reference()
    with np.load(ART / 'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'], ref[0])
        ref[0] = z['prior']
    with np.load(ART / 'nov25_historical_nine/historical.npz') as z:
        hf, nine, parent = z['fold'], z['nine'], z['v131']
    each = np.arange(len(y))
    awake = raw.CryoSleep.eq(False).to_numpy()
    views = [
        ('historical', rows, hf, old_native, observed(rows, hf, h, names, old_native, parent, nine, nine, awake, 'historical')),
        ('full', each, fold, native, observed(each, fold, p, names, native, *ref, awake, 'full'))
    ]
    return raw, y, groups, views


def fit_model(x, target):
    return CatBoostClassifier(iterations=380, depth=5, learning_rate=.055, l2_leaf_reg=6,
                              random_seed=3027, thread_count=3, verbose=False,
                              allow_writing_files=False, cat_features=CATEGORICAL).fit(x, target)


def train_fold(x, solo, y, strata, train, query, seed):
    xt, xq = x.iloc[train], x.iloc[query]
    yt, st = y[train], solo[train]
    def predict(model, frame):
        return model.predict_proba(frame)[:, 1]
    blind = predict(fit_model(xt, yt), xq)
    direct = predict(fit_model(xt.assign(individual=st.astype(int)), yt),
                     xq.assign(individual=solo[query].astype(int)))
    sham = yt.copy()
    rng = np.random.default_rng(seed)
    for level in np.unique(strata[train]):
        loc = np.flatnonzero(strata[train] == level)
        sham[loc] = rng.permutation(sham[loc])
    out = dict(blind=blind, direct=direct)
    for name, status in [('true', yt), ('shuffled', sham)]:
        size_model = fit_model(xt.assign(hypothetical_outcome=status.astype(int)), st)
        for label in (0, 1):
            out[f'{name}_{label}'] = predict(size_model, xq.assign(hypothetical_outcome=label))
    return out


def run():
    raw, y, g, views = regimes()
    solo = ~pd.Series(g).duplicated(keep=False).to_numpy()
    x = compact_features(raw)
    # Do not include the member suffix, occupancy or group summary in a size predictor.
    paid = raw[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].gt(0).any(axis=1)
    strata = (x.HomePlanet + '/' + x.CryoSleep + '/' + paid.astype(str)).to_numpy()
    past = views[0][1]
    report = {}
    OUT.mkdir(exist_ok=True)
    for tag, ids, fold, native, view in views:
        pred = {}
        fresh = ~np.isin(g[ids], g[past])
        for k in np.unique(fold):
            mask = fold == k
            q = ids[mask]
            others = ids[~mask]
            train = others[~np.isin(g[others], g[q])]
            assert not np.isin(g[q], g[train]).any()
            if tag == 'historical':
                fresh[mask] = ~np.isin(g[q], g[others])
            path = OUT / f'{tag}_{k}.npz'
            if not path.exists():
                values = train_fold(x, solo, y, strata, train, q, 3027 + int(k))
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream, t=train, q=q, train_y=y[train], **values)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for name, value in [('t',train), ('q',q), ('train_y',y[train])]:
                    np.testing.assert_array_equal(z[name], value)
                for key in ['blind','direct','true_0','true_1','shuffled_0','shuffled_1']:
                    pred.setdefault(key, np.empty(len(ids)))[mask] = z[key]
        anchor, known = view['anchor'], view['known']
        truth = y[ids]
        single = solo[ids]
        consensus = view['eligible'] & (view['votes'] == anchor[:, None]).all(1)
        masks = {'solo':single & known, 'solo_common':single & consensus,
                 'fresh_solo':single & known & fresh, 'fresh_common':single & consensus & fresh}
        methods = {'blind': pred['blind'], 'direct': pred['direct']}
        metrics = {}
        corrections = {}
        for key in ('true','shuffled'):
            a, b = (pred[f'{key}_{j}'].clip(.002,.998) for j in (0,1))
            ratio = np.where(single, np.log(b/a), np.log((1-b)/(1-a)))
            metrics[key+'_size_nll_given_outcome'] = log_loss(single, np.where(truth,b,a))
            methods[key] = expit(logit(pred['blind']) + ratio)
            # Account for the correction already made by a size-aware model.
            corrections[key] = logit(methods[key]) - logit(pred['direct'])
        actions = {}
        for key, prob in methods.items():
            metrics[key+'_solo_outcome_nll'] = log_loss(truth[single], prob[single])
            actions[key] = (prob>=.5) != anchor
        for key, step in corrections.items():
            actions[key+'_induced'] = actions[key] & ~actions['direct']
            for w in (.5, 1.):
                updated = logit(native.clip(.002,.998)) + w*step
                actions[f'{key}_increment_{w}'] = (anchor==(native>=.5)) & ((updated>=0)!=anchor)
        table = {name:{cohort:stats(flip*np.where(truth!=anchor,1,-1), mask, fold, g[ids])
                       for cohort, mask in masks.items()} for name, flip in actions.items()}
        report[tag] = dict(rows={name:int(mask.sum()) for name,mask in masks.items()},
                           losses=metrics, actions=table)
        print(tag, metrics, [(n,v['solo']['fixed'],v['solo']['harmed']) for n,v in table.items()], flush=True)
    DEST.with_name('solo_selection_likelihood.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
