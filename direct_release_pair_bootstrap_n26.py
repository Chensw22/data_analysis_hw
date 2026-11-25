"""Paired booking bootstrap directly on three frozen files' disagreements."""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from frozen_release_order_mix_n26 import (
    FIRST, NAMES, RESERVE, ROOT, apply, cells, collect, data, fit, historical_training,
    sources, test_decisions,
)


def bootstrap(y, group, sizes, local, target_sizes, test, name_a, name_b):
    a, b = local[name_a],local[name_b]
    changed = a!=b
    ta,tb = test[name_a],test[name_b]
    tc = ta!=tb
    _,groups = np.unique(group,return_inverse=True)
    ng = groups.max()+1
    active,ix = np.unique(groups[changed],return_inverse=True)
    strata = 2*(sizes[changed]-1)+b[changed]
    net = np.where(b[changed]==y[changed],1,-1)
    count = np.zeros((len(active),6),int)
    gain = np.zeros_like(count)
    np.add.at(count,(ix,strata),1)
    np.add.at(gain,(ix,strata),net)
    target = np.bincount(2*(target_sizes[tc]-1)+tb[tc],minlength=6)
    # Aggregate all orders with zero changes in a single category. Their bootstrap
    # contribution to each numerator and denominator is exactly zero.
    rng = np.random.default_rng(39246)
    w = rng.multinomial(ng,np.r_[np.ones(len(active)),ng-len(active)]/ng, size=20000)[:,:-1]
    num, den = w@gain,w@count
    missing = (den==0)@target
    estimate = np.divide(num,den,out=np.zeros_like(num,float),where=den!=0)@target
    def limits(array):
        return [float(k) for k in np.quantile(array,[.025,.975])]
    # If a stratum disappears, range its unobserved mean over [-1,+1]. Do
    # not condition on bootstrap draws which happen to have no empty stratum.
    target_n = len(ta)
    totals = count.sum(0)
    known_mean = np.divide(gain.sum(0),totals,out=np.zeros(6,float),where=totals>0)
    point = float((known_mean@target)*100/target_n) if np.all((totals!=0)|(target==0)) else None
    return dict(fixes=int((net>0).sum()),harms=int((net<0).sum()),bookings=len(active),test_changes=int(tc.sum()),
                conditional_pp=point,local_net_interval=limits(num.sum(1)),
                direct_pp_envelope=[limits(100*(estimate-missing)/target_n)[0],
                                    limits(100*(estimate+missing)/target_n)[1]],
                bootstrap_draws_with_missing=int((missing>0).sum()),draws=len(w),
                cells=[dict(size=k//2+1,new_class=bool(k%2),local_passengers=int(totals[k]),
                            local_bookings=int((count[:,k]>0).sum()),test_changes=int(target[k]),
                            local_net=int(gain[:,k].sum())) for k in range(6)])


def run():
    _,y,g,fold = data()
    runs, truth,group,other_fold,versions = collect()
    for a,b in [(y,truth),(g,group),(fold,other_fold)]:
        np.testing.assert_array_equal(a,b)
    q = np.concatenate([r[0] for r in runs['full']])
    latest = np.concatenate([p^(same&act['actual_and_scores']) for _,same,p,act in runs['full']])
    assert len(q)==len(set(q))==8532
    hist = sources()[-1]['historical']
    idx = [list(NAMES).index(k) for k in [*FIRST,'review','fare','six']]
    veto = apply(versions[q][:,idx],fit(*historical_training(y,g,hist)))
    local = dict(six=versions[q,-1],tabnet=latest,reserve=latest^veto)
    test_id = pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    test_g = test_id.str.split('_').str[0].to_numpy()
    names = dict(six=NAMES['six'],tabnet=NAMES['tabnet'],reserve=RESERVE)
    test = {}
    files = {}
    for k,name in names.items():
        test[k], digest = test_decisions(name,test_id)
        receipts = [json.loads(p.read_text()).get('submission_ref') for p in
                    (ROOT/'reports'/name/'online_attempts').glob('*.json')]
        files[k] = dict(digest=digest,official_receipts=receipts)
    report = dict(files=files,pairs={})
    for a,b in combinations(names,2):
        result = bootstrap(y[q],g[q],cells(g)[q],local,cells(test_g),test,a,b)
        report['pairs'][b+'_minus_'+a] = result
        print(b,a,result,flush=True)
    report['interpretation'] = ('Positive = second file outperforms first. Each pair compares predictions directly; '
        'shared successes and failures cancel. The transfer assumption is constant average correctness among '
        'changed passengers of the same size and direction. Percentile envelopes reflect whole-booking variation '
        'and worst/best values in empty resampled cells. They do not cover model reselection or domain shift.')
    (ROOT/'reports/nov23_matched_learning_curve/direct_release_pair_bootstrap.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
