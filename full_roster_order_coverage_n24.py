"""Measure how covered training orders compare with the unlabeled competition test roster."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_selected_tree_replay_n24 import original_prediction,prepared_vote


def profile(raw):
    g = raw.PassengerId.str.split('_').str[0]
    size = raw.groupby(g).PassengerId.transform('size').to_numpy()
    adult = raw.Age.ge(18)
    accompanied = adult.groupby(g).transform('sum').to_numpy()>adult.to_numpy(int)
    age = np.select([raw.Age.isna(),raw.Age.lt(13)],['unknown','child'],default='13plus')
    cells = [f'{min(n,4)}|{a}|{int(p)}' for n,a,p in zip(size,age,accompanied)]
    return np.asarray(cells),size,accompanied,age=='child'


def comparison(cells,test_cells,mask,margins):
    ids = cells[mask]
    counts = {}
    weights = np.zeros(len(ids))
    absent = 0
    comparisons = {}
    for k,n in zip(*np.unique(test_cells,return_counts=True)):
        group = ids==k
        counts[k] = dict(source=int(group.sum()),target=int(n))
        if group.any():
            weights[group] = n/len(test_cells)/group.sum()
        else:
            absent += n
    mass = weights.sum()
    for k,v in margins.items():
        d = v[mask]
        comparisons[k] = dict(unweighted_net=int(d.sum()),covered_weighted_pp=float(100*(weights@d)/mass),
                              weighted_contribution_pp=float(100*(weights@d)))
    return dict(rows=int(mask.sum()),test_rows_without_support=int(absent),
                covered_mass=float(mass),effective_rows=float(mass**2/(weights@weights)),
                largest_normalized_row_weight=float(weights.max()/mass),cells=counts,results=comparisons)


def run():
    raw,y,g,fold = data()
    _,_,old_groups,*_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    unseen = ~np.isin(g,old_groups)
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId','Age'])
    test_groups = test.PassengerId.str.split('_').str[0]
    assert len(test)==4277 and not np.isin(g,test_groups).any()
    cells,size,peer,child = profile(raw)
    target,n,adult,kid = profile(test)
    rows = np.arange(len(y))
    nine = full[:,[names.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]]
    challenger = original_prediction(rows,fold,g,y,'full')
    selected = prepared_vote(challenger,ref,nine)
    with np.load(ROOT/'artifacts/nov24_context_tree_calibration/oof.npz') as z:
        for k,v in [('y',y),('fold',fold),('passenger',raw.PassengerId)]:
            np.testing.assert_array_equal(z[k],v)
        p = z['full'][:,1]
    context = np.where((full[:,names.index('mitra96')]>=.5)==ref,p>=.5,ref)
    margins = {k:(pred==y).astype(int)-(ref==y).astype(int)
               for k,pred in [('queued_nine_vote',selected),('context_tree',context)]}
    assert margins['queued_nine_vote'].sum()==8 and margins['context_tree'].sum()==-19
    report = {}
    for name,subset in [('full',np.ones(len(y),bool)),('unseen',unseen)]:
        report[name] = comparison(cells,target,subset,margins)
        report[name].update(solo=int((subset&(size==1)).sum()),has_adult=int((subset&peer).sum()),
                            child_with_adult=int((subset&peer&child).sum()))
    report['test'] = dict(n=len(test),solo=int((n==1).sum()),has_adult=int(adult.sum()),child_with_adult=int((adult&kid).sum()))
    path = ROOT/'reports/nov23_matched_learning_curve/order_coverage.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    for k,v in report.items():
        print(k,{a:b for a,b in v.items() if a!='cells'})


if __name__=='__main__':
    with threadpool_limits(2):
        run()
