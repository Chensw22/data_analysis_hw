"""Test local cabin allocation gaps against a matched base and randomized deck-side context."""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from disjoint_tree_fields_n25 import ART, DEST, OUTER, RELEASE, stats
from exaone_validation_data import CATEGORICAL, compact_features
from full_roster_booking_validation_n23 import ROOT, data
from full_roster_six_failure_archive_n24 import prediction_sets


def roster_views(raw):
    base = compact_features(raw)
    group = raw.PassengerId.str.split('_').str[0].astype(int)
    cells = pd.DataFrame(dict(deck=base.Deck, side=base.Side, cabin=base.CabinNumber, order=group))
    cells = cells.dropna().groupby(['deck','side','cabin']).order.agg(['min','max','size','nunique']).reset_index()
    tracks = cells.groupby(['deck', 'side'])
    cols = []
    for distance in (1, 4):
        for shift in (-distance, distance):
            sign = 1 if shift > 0 else -1
            side = 'before' if shift > 0 else 'after'
            def add(name, val):
                key = f'{name}_{side}_{distance}'
                cols.append(key)
                cells[key] = val
            add('cabin_delta', sign * (cells.cabin-tracks.cabin.shift(shift)))
            edge = cells['min']-tracks['max'].shift(shift) if shift > 0 else tracks['min'].shift(shift)-cells['max']
            add('order_delta', edge)
            add('near_occupants', tracks['size'].shift(shift))
    cols += ['size','nunique']
    key = ['deck','side','cabin']
    lookup = pd.MultiIndex.from_arrays([base.Deck,base.Side,base.CabinNumber],names=key)
    true = cells.set_index(key)[cols]
    shuffled = cells.copy()
    rng = np.random.default_rng(444)
    for ix in tracks.indices.values():
        shuffled.loc[ix,cols] = cells.loc[rng.permutation(ix),cols].to_numpy()
    def extend(view):
        values = view.reindex(lookup).reset_index(drop=True)
        return pd.concat([base, values],axis=1)
    return {'base':base, 'gaps':extend(true), 'shuffled':extend(shuffled.set_index(key)[cols])}


def run():
    raw, y, groups, f = data()
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda k:k!='Name')
    views = roster_views(pd.concat([raw,test],ignore_index=True))
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows, hf = z['rows'], z['fold']
    _, panels = prediction_sets(raw,y,groups,f,rows,hf)
    full = np.column_stack(panels[0]) >= .5
    with np.load(ART/'nov25_production_anchor_cpu/oof.npz') as z:
        np.testing.assert_array_equal(z['fold'],f)
        en, ec, ep = z['certified_nine'].copy(),z['certified_v131_equals_v96'].copy(),z['v96']
    for k in OUTER:
        with np.load(RELEASE/f'release{k}.npz') as z:
            q = np.flatnonzero(f == k)
            np.testing.assert_array_equal(z['q'],q)
            en[q], ec[q] = z['nine'],z['v131']>=0
    all_agree = (full == en[:,None]).all(1)
    report, preds = {}, {}
    for name, frame in views.items():
        p = np.empty(len(y))
        for k in np.unique(f):
            t,q = np.flatnonzero(f!=k),np.flatnonzero(f==k)
            assert not np.isin(groups[t],groups[q]).any()
            model = CatBoostClassifier(iterations=490,depth=5,learning_rate=.05,l2_leaf_reg=5,
                thread_count=3,random_seed=735,verbose=False,allow_writing_files=False)
            model.fit(frame.iloc[t], y[t].astype(int), cat_features=CATEGORICAL)
            p[q] = model.predict_proba(frame.iloc[q])[:,1]
        preds[name] = p
    for name, p in preds.items():
        rules = {'direct':p >= .5, 'quarter':np.where((ep>=.5)==en,.75*ep+.25*p>=.5,en)}
        if name!='base':
            # Only consider an edit when adding neighborhood columns changed the base tree's decision.
            rules['context_dissent'] = np.where((p>=.5)!=(preds['base']>=.5),p>=.5,en)
        report[name] = {'all_accuracy':float(((p>=.5)==y).mean())}
        for rule, pred in rules.items():
            delta = (pred==y).astype(int)-(en==y)
            report[name][rule] = {scope:stats(delta,ec & mask,f,groups) for scope, mask in
                [('certified',np.ones(len(f),bool)),('consensus',all_agree),('three',np.isin(f,OUTER)),
                 ('other_five',~np.isin(f,OUTER)),('new_orders',~np.isin(groups,groups[rows]))]}
    for name in ('gaps','shuffled'):
        d = ((preds[name]>=.5)==y).astype(int)-((preds['base']>=.5)==y)
        report[name]['direct_vs_base_all_rows'] = stats(d,np.ones(len(f),bool),f,groups)
    DEST.with_name('independent_context_n25.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,r in report.items():
        print(name, r['all_accuracy'], [(k,x['certified']['fixed'],x['certified']['harmed'],
            x['consensus']['fixed'],x['consensus']['harmed']) for k,x in r.items() if isinstance(x,dict) and 'certified' in x])


if __name__=='__main__':
    run()
