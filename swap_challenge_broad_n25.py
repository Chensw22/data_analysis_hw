"""Estimate the class-specific bill/manifest dependence by detecting matched bill replacements."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets

ROOT = Path(__file__).resolve().parent
FEES = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']


def inputs(raw, groups):
    cabin = raw.Cabin.str.split('/', expand=True)
    cat = raw[['HomePlanet','CryoSleep','VIP','Destination']].astype('string').fillna('?')
    cat = cat.assign(deck=cabin[0].fillna('?'),side=cabin[2].fillna('?'))
    amount = raw[FEES].to_numpy(float)
    bills = np.column_stack([np.nan_to_num(np.log1p(amount),nan=-1),
                             np.log1p(np.nansum(amount,axis=1))])
    table = np.column_stack([pd.get_dummies(cat, dtype=float), raw.Age.fillna(-1),
                             pd.to_numeric(cabin[1]).fillna(-1),
                             pd.Series(groups).map(pd.Series(groups).value_counts()),
                             raw.PassengerId.str.split('_').str[1].astype(int), bills])
    # Labels never enter these age and sleep strata.
    strata = pd.factorize(cat.CryoSleep + '_' + pd.cut(raw.Age,[-1,12,18,80,200]).astype('string').fillna('?'))[0]
    return table, bills.shape[1], strata


def swapped(x, width, strata, g, y, train, conditional, seed):
    rng = np.random.default_rng(seed)
    donor = np.arange(len(train))
    strata = 2*strata[train] + (y[train] if conditional else 0)
    for cell in np.unique(strata):
        local = np.flatnonzero(strata == cell)
        for i in local:
            permitted = local[g[train[local]] != g[train[i]]]
            if len(permitted):
                donor[i] = rng.choice(permitted)
    keep = donor != np.arange(len(train))
    fake = x[train[keep]].copy()
    fake[:, -width:] = x[train[donor[keep]], -width:]
    assert not np.any(g[train[keep]] == g[train[donor[keep]]])
    return fake, y[train[keep]].astype(int)


def classifier():
    return CatBoostClassifier(iterations=240,depth=5,learning_rate=.065,random_seed=428,
                              thread_count=2,verbose=False,allow_writing_files=False)


def run():
    raw, y, groups, f8 = data()
    table, width, strata = inputs(raw, groups)
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        rows, folds, nine, v131 = [z[k] for k in ('rows','fold','nine','v131')]
        np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[rows])
        np.testing.assert_array_equal(z['y'],y[rows])
    with np.load(ROOT/'artifacts/nov25_mitra_fp32_head/oof.npz') as z:
        np.testing.assert_array_equal(z['row'],rows)
        anchor = z['native']
    _, panels = prediction_sets(raw,y,groups,f8,rows,folds)
    votes = np.column_stack(panels[1]) >= .5
    agree = (votes == nine[:,None]).all(1)
    free = ((anchor>=.5)==v131) & (v131==nine)
    names = ['binary','real_conditional','dependence_conditional','real_unconditional','dependence_unconditional','real_repeat','dependence_repeat']
    scores = np.full((len(rows),len(names)),np.nan)
    for fold in np.unique(folds):
        use = folds == fold
        q = rows[use]
        t = rows[~use & ~np.isin(groups[rows],groups[q])]
        assert not np.isin(groups[t],groups[q]).any()
        scores[use,0] = classifier().fit(table[t],y[t]).predict_proba(table[q])[:,1]
        for v,conditional in enumerate([True,False,True]):
            fake, side = swapped(table,width,strata,groups,y,t,conditional,seed=1561 if v==2 else 554)
            model = classifier().fit(np.vstack([table[t],fake]),np.r_[y[t].astype(int),2+side])
            p = model.predict_proba(table[q]).clip(1e-7,1)
            scores[use,1+2*v] = np.log(p[:,1]/p[:,0])
            scores[use,2+2*v] = np.log(p[:,1]/p[:,0])-np.log(p[:,3]/p[:,2])
        print('fold',fold,flush=True)
    assert np.isfinite(scores).all()
    native_odds = logit(anchor.clip(.001,.999))
    raw_odds = logit(scores[:,0].clip(.001,.999))
    decisions = {'binary':scores[:,0]>=.5}
    for j,kind in enumerate(('conditional','unconditional','repeat')):
        real, ratio = scores[:,1+2*j:3+2*j].T
        decisions['real_'+kind] = real >= 0
        decisions['raw_plus_dependency_'+kind] = raw_odds+ratio >= 0
        decisions['anchor_plus_dependency_'+kind] = native_odds+ratio >= 0
        decisions['quarter_real_'+kind] = .75*anchor+.25*expit(real)>=.5
    first = decisions['anchor_plus_dependency_conditional']
    second = decisions['anchor_plus_dependency_repeat']
    decisions['two_donors_confirm_dependency'] = np.where(first == second, first, nine)
    frozen = np.empty(len(rows))
    for fold in np.unique(folds):
        with np.load(ROOT / f'artifacts/nov25_mitra_four_outcomes/fold{fold}.npz') as z:
            np.testing.assert_array_equal(z['q'],rows[folds==fold])
            frozen[folds==fold] = z['scores'][:,0]
    truth=y[rows]
    result={}
    for name,pred in decisions.items():
        edit = free & (pred!=nine)
        delta = np.where(edit,np.where(nine==truth,-1,1),0)
        result[name]=dict(corrected=int((delta>0).sum()),harmed=int((delta<0).sum()),
                          common_fixed=int((delta[agree]>0).sum()),
                          common_fixed_also_frozen=int((agree & (delta>0) & ((frozen>=.5)==truth)).sum()),common_harmed=int((delta[agree]<0).sum()),
                          fold_net=[int(delta[folds==k].sum()) for k in np.unique(folds)])
    out = dict(rows=len(rows),consensus_rows=int(agree.sum()),common_errors=int((agree & (nine!=truth)).sum()),
               predictions=result)
    print(out,flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/swap_challenge_broad_n25.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
