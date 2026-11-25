"""Quantify how much each validation cohort supports the actual frozen test decisions."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from check_disagreement_fallback_n26 import FIRST, apply, fit, historical_training
from export_historical_veto_n26 import NAME as RESERVE
from forecast_frozen_queue_n26 import NAMES
from full_roster_booking_validation_n23 import ROOT, data
from held_expert_portfolios_n26 import sources
from six_way_against_scored_nine_n26 import test_decisions
from tabnet_attention_transfer_n26 import collect


def cells(groups):
    sizes = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
    return np.minimum(sizes,3)


def projection(local_class, target_class, signed, groups, target_n):
    details = []
    total = 0.
    missing = 0
    weak_total = 0.
    weak_rows = 0
    weights = np.zeros(len(local_class))
    for k in range(6):
        local = local_class==k
        target = int((target_class==k).sum())
        n = int(local.sum())
        orders = len(np.unique(groups[local]))
        net = int(signed[local].sum())
        details.append(dict(size=min(k//2+1,3),new_class=bool(k%2),passengers=n,bookings=orders,
                            test_changes=target,net=net))
        if n:
            total += target*net/n
            weights[local] = target/n
        else:
            missing += target
        if orders<10:
            weak_rows += target
        elif n:
            weak_total += target*net/n
    order_weights = np.bincount(np.unique(groups,return_inverse=True)[1], weights=weights)
    ess = float(order_weights.sum()**2 / np.square(order_weights).sum()) if weights.any() else 0.
    return dict(by_cell=details, unsupported_test_changes=missing, changes_with_under_ten_reference_bookings=weak_rows,
                effective_reference_bookings_for_supported_edits=ess,
                point_gain_pp=(100*total/target_n if not missing else None),
                gain_pp_range_if_missing_cells_arbitrary=[100*(total+s*missing)/target_n for s in (-1,1)],
                gain_pp_range_if_sparse_cells_arbitrary=[100*(weak_total+s*weak_rows)/target_n for s in (-1,1)])


def run():
    raw,y,g,fold = data()
    runs, truth,groups,fold2,versions = collect()
    for a,b in [(truth,y),(groups,g),(fold2,fold)]:
        np.testing.assert_array_equal(a,b)
    q = np.concatenate([r[0] for r in runs['full']])
    latest = np.concatenate([old^(same&act['actual_and_scores']) for _,same,old,act in runs['full']])
    assert len(np.unique(q))==len(q)==8532
    local = np.column_stack([versions[q],latest])
    historical = sources()[-1]['historical']
    historical_groups = g[historical['ids']]
    fresh = ~np.isin(g[q],historical_groups)
    idx = [list(NAMES).index(k) for k in [*FIRST,'review','fare','six']]
    veto = apply(local[:,idx],fit(*historical_training(y,g,historical)))
    local = np.column_stack([local,latest^veto])
    names = dict(NAMES, reserve=RESERVE)
    test_id = pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    test_group = test_id.str.split('_').str[0].to_numpy()
    assert not np.isin(test_group,g).any()
    target = np.column_stack([test_decisions(name,test_id)[0] for name in names.values()])
    assert local.shape[1]==target.shape[1]==len(names)
    size,test_size = cells(g)[q], cells(test_group)
    truth = y[q]
    scopes = {'all_eight_folds':np.ones(len(q),bool),'outside_historical_orders':fresh,
              'inside_historical_orders':~fresh}
    report = dict(test_passengers=len(test_id),test_bookings=len(set(test_group)),
                  test_size_1_2_3plus=np.bincount(test_size,minlength=4)[1:].tolist(),cohorts={})
    for scope, valid in scopes.items():
        item = dict(passengers=int(valid.sum()),bookings=len(set(g[q][valid])),strata={},candidates={})
        report['cohorts'][scope] = item
        for level in (1,2,3):
            sub = valid&(size==level)
            item['strata'][str(level)] = dict(rows=int(sub.sum()),orders=len(set(g[q][sub])),
                        champion_wrong=int((local[sub,0]!=truth[sub]).sum()),
                        tabnet_wrong=int((local[sub,-2]!=truth[sub]).sum()))
        for j,key in enumerate(list(names)[1:],1):
            eligible = valid&(local[:,j]!=local[:,0])
            changed = target[:,0]!=target[:,j]
            signed = np.where(local[eligible,j]==truth[eligible],1,-1)
            state = 2*(size[eligible]-1)+local[eligible,j]
            on_test = 2*(test_size[changed]-1)+target[changed,j]
            r = projection(state,on_test,signed,g[q][eligible],len(test_id))
            r.update(validation_net=int(signed.sum()), raw_gain_pp=float(100*signed.sum()/valid.sum()),
                     test_changes=int(changed.sum()))
            item['candidates'][key] = r
        item['raw_ranking'] = sorted(item['candidates'],key=lambda k:-item['candidates'][k]['validation_net'])
        item['projected_ranking_supported_only'] = sorted(
            [k for k,r in item['candidates'].items() if r['point_gain_pp'] is not None],
            key=lambda k:-item['candidates'][k]['point_gain_pp'])
    report['interpretation'] = ('All and fresh use exactly the same eight-fold frozen decisions. Fresh is a descriptive subset '
       'whose booking IDs were absent from the historical replay. Transfer assumes equal net gain per changed row of each '
       'booking size and prediction direction. Ranges allow arbitrary net effects in sparse cells while keeping other '
       'cell averages fixed; they are sensitivity analyses, not sampling confidence intervals.')
    assert report['cohorts']['all_eight_folds']['candidates']['reserve']['validation_net']==95
    (ROOT/'reports/nov23_matched_learning_curve/frozen_release_order_mix.json').write_text(json.dumps(report,indent=2)+'\n')
    for scope,r in report['cohorts'].items():
        print(scope, r['strata'], r['raw_ranking'], r['projected_ranking_supported_only'], flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
