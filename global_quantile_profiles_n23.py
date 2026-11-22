"""Cross-fit raw and conditional spending profiles across all development passengers."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss

import relational_error_corrector as ref
from exaone_validation_data import CATEGORICAL
from peer_bill_correction import BILLS
from sleeper_consensus_scope import ART, ROOT


def empirical_profiles(x, passenger_ids):
    costs = x[BILLS]
    favorable = costs[['FoodCourt', 'ShoppingMall']].sum(axis=1, min_count=2)
    service = costs[['Spa', 'VRDeck', 'RoomService']].sum(axis=1, min_count=3)
    total = favorable + service
    composition = (favorable + 1) / (total + 2)
    stats = costs.assign(total_spent=total, customer_share=composition, age=x.Age)
    groups = passenger_ids.str.split('_').str[0]
    group_size = groups.map(groups.value_counts())
    deck_side = [x.Deck, x.Side]
    roster_features = dict(party_size=group_size, missing_bills=costs.isna().sum(axis=1),
                           total_spent=total, customer_share=composition,
                           party_total_other=total.groupby(groups).transform('sum')-total,
                           party_spend_observed=total.groupby(groups).transform('count')-total.notna())
    for label, keys in [('route', [x.HomePlanet,x.Destination,x.CryoSleep]),
                        ('berth', [*deck_side, x.CabinNumber.floordiv(120).fillna(-1)])]:
        # Average rank respects ties at zero. Shrink small groups toward the global rank.
        grouped = stats.groupby(keys, sort=False)
        n = grouped.transform('count')
        global_rank = stats.rank(pct=True)
        rank = (grouped.rank() - .5 + 30*global_rank)/(n+30)
        deviation = np.log1p(stats)-np.log1p(grouped.transform('median'))
        for column in stats:
            roster_features[f'{label}_percentile_{column}'] = rank[column]
            roster_features[f'{label}_deviation_{column}'] = (deviation[column].fillna(0)*n[column]/(n[column]+30)).where(stats[column].notna())
    return pd.concat([x, pd.DataFrame(roster_features, index=x.index)], axis=1)


def run():
    inp,splits,fold,bases,_ = ref.overlap.load_authenticated()
    dev,y = inp['dev'], inp['y'].astype(bool)
    x = inp['features'].reset_index(drop=True)
    ids = pd.concat([pd.read_csv(ROOT / f'data/raw/{part}.csv', usecols=['PassengerId']).PassengerId
                     for part in ['train','test']], ignore_index=True)
    ref_rows = pd.read_csv(ART/'nov21_case_checks/v136_final_oof.csv')
    np.testing.assert_array_equal(dev, ref_rows.row)
    np.testing.assert_array_equal(ids.iloc[dev], ref_rows.PassengerId)
    np.testing.assert_array_equal(fold, ref_rows.fold)
    np.testing.assert_array_equal(y, ref_rows.true)
    hard, base = ref_rows.hard.to_numpy(), ref_rows.v131.to_numpy()
    scores, outputs = {}, {}
    for name,features in [('raw',x),('quantile',empirical_profiles(x,ids))]:
        p = np.full(len(y), np.nan)
        for train, val in splits:
            model = CatBoostClassifier(iterations=530, learning_rate=.04, depth=6, l2_leaf_reg=5,
                                       random_seed=2311, thread_count=2, verbose=False, allow_writing_files=False)
            model.fit(features.iloc[dev[train]], y[train], cat_features=CATEGORICAL)
            p[val] = model.predict_proba(features.iloc[dev[val]])[:,1]
        assert np.isfinite(p).all()
        ok = (p>=.5)==y
        fixed, harmed = ok & (base!=y), ~ok & (base==y)
        scores[name] = dict(accuracy=float(ok.mean()), logloss=float(log_loss(y,p)), fixed=int(fixed.sum()),
                            harmed=int(harmed.sum()), hard_fixed=int((fixed&hard).sum()), hard_harmed=int((harmed&hard).sum()),
                            fold_net=[int((fixed&(fold==i)).sum()-(harmed&(fold==i)).sum()) for i in range(3)])
        outputs[name] = p
        print(name,scores[name],flush=True)
    raw, rich = [(outputs[n]>=.5)==y for n in ('raw','quantile')]
    wrong = base!=y
    prior_wrong = np.logical_and.reduce([(v>=.5)!=y for v in bases.values()])
    comparison = {}
    for name,correct,control in [('quantile',rich,raw),('raw',raw,rich)]:
        exclusive=wrong & correct & ~control
        comparison[name]=dict(repairs_not_found_by_other=int(exclusive.sum()),
                              hard_repairs_not_found_by_other=int((exclusive & hard).sum()),
                              repairs_missed_by_other_and_authenticated_models=int((exclusive & prior_wrong).sum()))
    out=ART/'global_quantile_profiles_n23'
    out.mkdir(exist_ok=True)
    pd.DataFrame(dict(row=dev,fold=fold,y=y,p=outputs['quantile'],p_raw=outputs['raw'])).to_csv(out/'oof.csv',index=False)
    ref.write(ROOT/'reports/global_quantile_profiles_n23/results.json',
              dict(rows=len(y),protocol='Original three folds, shared unlabeled roster features, fixed 0.5 cutoff.',
                   historical_models=list(bases),scores=scores,exclusive=comparison))
    print(comparison)


if __name__=='__main__':
    run()
