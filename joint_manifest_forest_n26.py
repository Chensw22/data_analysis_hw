"""Estimate P(transport, observed field pair | other fields) using disjoint bookings for tree splits and leaf frequencies."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART, DEST
from exaone_validation_data import compact_features
from family_identity_transfer_n26 import fit, panels_and_reference
from full_roster_booking_validation_n23 import data
from outstanding_error_families_n26 import observed

BILLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
OUT = ART / 'nov26_joint_manifest_forest'


def encode(frame):
    return pd.get_dummies(frame, dtype=float).fillna(-1.).to_numpy(np.float32)


def fields(raw):
    frame = compact_features(raw)
    bills = frame[BILLS].to_numpy(float)
    sorted_bills = np.sort(np.nan_to_num(bills, nan=-1.), axis=1)
    masked_bills = frame.drop(columns=BILLS).copy()
    observed = []
    for j, amount in enumerate(sorted_bills.T):
        masked_bills[f'ordered_bill_{j}'] = amount
    for amount in sorted_bills.T[-2:]:
        match = bills == amount[:, None]
        identity = np.where(match.sum(axis=1) == 1, match.argmax(axis=1), 5)
        observed.append(np.where(amount > 0, identity, 6))
    fields = {'service_pair': (encode(masked_bills), np.stack(observed, axis=1))}
    for name, cols in [('route', ['HomePlanet', 'Destination']), ('sleep_side', ['CryoSleep', 'Side']),
                       ('origin_berth', ['HomePlanet', 'Side', 'Deck'])]:
        fields[name] = encode(frame.drop(columns=cols)), frame[cols].to_numpy(str)
    return encode(frame), {name: (x, np.unique(key, axis=0, return_inverse=True)[1])
                           for name, (x, key) in fields.items()}


def predict(x, key, y, groups, t, q, seed):
    assert not np.isin(groups[t], groups[q]).any()
    width = int(key.max()) + 1
    totals = np.zeros((len(q), 3))
    splits = GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed)
    a, b = next(splits.split(t, groups=groups[t]))
    for split, (grow, count) in enumerate([(t[a], t[b]), (t[b], t[a])]):
        assert not np.isin(groups[grow], groups[count]).any()
        trees = ExtraTreesClassifier(n_estimators=64, min_samples_leaf=8, max_features=.8,
                                     n_jobs=2, random_state=seed+split)
        trees.fit(x[grow], 2*key[grow] + y[grow])
        nearest = trees.apply(x[count]).T
        lookup = trees.apply(x[q]).T
        for tree, l, held in zip(trees.estimators_, nearest, lookup):
            bins = tree.tree_.node_count
            n = np.bincount(l, minlength=bins)
            pos = np.bincount(l, weights=y[count], minlength=bins)
            nk = np.bincount(l*width + key[count], minlength=bins*width)
            pk = np.bincount(l*width + key[count], weights=y[count], minlength=bins*width)
            at = held*width + key[q]
            # Divide within each tree before pooling, so larger leaves have no extra vote.
            totals[:, 0] += (pos[held] + y[count].mean()) / (n[held] + 1)
            totals[:, 1] += nk[at] / n[held].clip(1)
            totals[:, 2] += pk[at] / n[held].clip(1)
    prior, mass, joint = (totals / (2*len(trees.estimators_))).T
    posterior = (joint + .01*prior) / (mass + .01)
    return np.column_stack([prior, posterior])


def generate():
    raw, y, groups, fold = data()
    frame, views = fields(raw)
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        old_rows, old_fold = z['rows'], z['fold']
        np.testing.assert_array_equal(y[old_rows], z['y'])
    OUT.mkdir(exist_ok=True)
    for series, rows, split in [('historical', old_rows, old_fold), ('full', np.arange(len(raw)), fold)]:
        for k in np.unique(split):
            q = rows[split==k]
            t = rows[(split!=k) & ~np.isin(groups[rows], groups[q])]
            seed = 9118 + int(k)
            path = OUT/f'{series}_{k}.npz'
            if not path.exists():
                models = {name: predict(x, key, y, groups, t, q, seed) for name, (x,key) in views.items()}
                direct = predict(frame, np.zeros(len(raw), int), y, groups, t, q, seed)[:, 0]
                with path.with_suffix('.tmp').open('wb') as buf:
                    np.savez_compressed(buf, **models, direct=direct, t=t, q=q, train_y=y[t], passenger=raw.PassengerId.iloc[q].to_numpy(str))
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for key, val in [('t',t), ('q',q), ('train_y',y[t]), ('passenger',raw.PassengerId.iloc[q])]:
                    np.testing.assert_array_equal(z[key], val)
            print(series, k, len(t), len(q), flush=True)


def run():
    generate()
    raw, y, groups, fold = data()
    _, _, _, rows, names, p, h, native, old_native, ref = panels_and_reference()
    awake = raw.CryoSleep.eq(False).to_numpy()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        np.testing.assert_array_equal(z['row'],np.arange(len(raw)))
        ref[0] = z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf, old, par = z['fold'], z['nine'], z['v131']
    history = observed(rows, hf, h, names, old_native, par, old, old, awake, 'historical')
    current = observed(np.arange(len(y)), fold, p, names, native, *ref, awake, 'full')
    results = {}
    for series, ids, kfold, base, anchor, clean in [
        ('full', np.arange(len(raw)), fold, native, current, ~np.isin(groups, groups[rows])),
        ('historical', rows, hf, old_native, history,
         np.array([not np.any((groups[rows]==g) & (hf!=k)) for g,k in zip(groups[rows],hf)]))]:
        methods = {}
        for k in np.unique(kfold):
            q = np.flatnonzero(kfold==k)
            with np.load(OUT/f'{series}_{k}.npz') as z:
                np.testing.assert_array_equal(z['q'], ids[q])
                names = ['service_pair','route','sleep_side','origin_berth']
                priors = np.stack([z[name][:,0] for name in names],axis=1)
                post = np.stack([z[name][:,1] for name in names],axis=1)
                direct = z['direct']
            odds = lambda values: logit(np.clip(values,.002,.998))
            raw_scores = {'direct':direct,'masked_mean':priors.mean(1),'restored_mean':post.mean(1),
                          **{name:post[:,j] for j,name in enumerate(names)}}
            sign = base[q]>=.5
            for name, scores in raw_scores.items():
                for weight in [1.,.25]:
                    hard = (1-weight)*base[q]+weight*scores >= .5
                    methods.setdefault(f'{name}_{weight}', np.zeros(len(ids),bool))[q] = (hard!=sign)
            offset = odds(base[q])
            initial = np.column_stack([np.ones(len(q)), offset, odds(direct)-offset])
            designs = dict(direct=initial, masked=np.column_stack([initial, odds(priors)-offset[:,None]]),
                           restored=np.column_stack([initial, odds(priors)-offset[:,None], odds(post)-odds(priors)]))
            for scheme in ['all','new']:
                if scheme == 'new' and series == 'historical':
                    continue
                batches = ([(np.flatnonzero(~clean[q]),np.flatnonzero(clean[q]))] if scheme=='new'
                           else GroupKFold(4,shuffle=True,random_state=6195).split(q,groups=groups[ids[q]]))
                for t, u in batches:
                    assert not np.isin(groups[ids[q[t]]], groups[ids[q[u]]]).any()
                    for name, x in designs.items():
                        coef = fit(x[t],offset[t],y[ids[q[t]]])
                        change = (offset[u]+x[u]@coef>=0)!=sign[u]
                        methods.setdefault(f'{scheme}_cal_{name}',np.zeros(len(ids),bool))[q[u]]=change
        anchor_label = anchor['anchor']
        known = anchor['known']
        match = anchor_label==(base>=.5)
        family = (anchor['votes']==anchor_label[:,None]).all(1)&anchor['eligible']
        scores = {}
        for name, flip in methods.items():
            changed = flip & match
            parts = {}
            for group, mask in [('all',known),('new_orders',known&clean),('all_families',family),
                                ('families_new',family&clean)]:
                if name.startswith('new_cal_') and group not in ('new_orders','families_new'):
                    continue
                edit = changed&mask
                positive = (anchor_label!=y[ids])
                parts[group] = dict(rows=int(mask.sum()),errors=int((positive&mask).sum()),
                                    fixed=int((edit&positive).sum()),harmed=int((edit&~positive).sum()),
                                    by_fold=[int((edit&positive&(kfold==k)).sum())-int((edit&~positive&(kfold==k)).sum()) for k in np.unique(kfold)])
            scores[name]=parts
        results[series]=scores
        print(series, [(name,list(s.get('all',s['new_orders']).values())[2:4]) for name,s in scores.items()],flush=True)
    DEST.with_name('joint_manifest_forest.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
