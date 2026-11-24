"""Reconstruct the 26-model unanimous cases and report prevalence with matching denominators."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from full_roster_conditional_pool_n23 import ROOT, BILLS, banks, data
from full_roster_kernel_repair_overlap_n24 import from_cache
from full_roster_reference_innovation_n24 import bank


def unanimous_cases(raw,y,group,fold,full):
    rows = np.arange(len(y))
    _, previous = from_cache('full',rows,fold,y,group)
    extra = bank(raw,y,group,rows,fold,'full')
    scores = np.column_stack([full,previous,extra])
    votes = scores >= .5
    same = votes.all(1) | (~votes).all(1)
    wrong = same & (votes[:,0]!=y)
    assert scores.shape[1]==26 and wrong.sum()==223
    return scores,same,wrong


def describe(raw,y,g,fold,scores,same,wrong):
    spending = raw[BILLS]
    paid = spending.gt(0)
    top = spending.fillna(0).idxmax(axis=1).where(paid.any(axis=1),'no_positive_bill')
    _,inv,count = np.unique(g,return_inverse=True,return_counts=True)
    solo = count[inv]==1
    profiles = {'solo':solo,'paid':paid.any(axis=1).to_numpy(),
                'awake':raw.CryoSleep.eq(False).to_numpy(),
                'any_missing_bill':spending.isna().any(axis=1).to_numpy(),
                'one_positive_bill':(paid.sum(axis=1)==1).to_numpy(),
                'two_or_more_positive_bills':(paid.sum(axis=1)>=2).to_numpy(),
                'complete_zero_bills':((~paid.any(axis=1)) & spending.notna().all(axis=1)).to_numpy()}
    profiles.update({'largest_'+str(k):(top==k).to_numpy() for k in top.unique()})
    result = {}
    for direction in (False,True):
        population = same & ((scores[:,0]>=.5)==direction)
        def count_rows(mask):
            ids = population & mask
            return dict(n=int(ids.sum()),exceptions=int((ids & wrong).sum()))
        result[str(direction)] = dict(all=count_rows(np.ones(len(y),bool)),
            profiles={k:count_rows(v) for k,v in profiles.items()},
            exceptions_by_fold=[int((wrong & population & (fold==k)).sum()) for k in np.unique(fold)])
    return result


def run():
    raw,y,g,fold = data()
    _,_,full = banks(raw,y,fold)
    p,same,wrong = unanimous_cases(raw,y,g,fold,full)
    out = ROOT/'artifacts/nov24_remaining_cases'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'audit.npz',passenger=raw.PassengerId.to_numpy(str),y=y,fold=fold,scores=p,
                        unanimous=same,missed=wrong)
    report = describe(raw,y,g,fold,p,same,wrong)
    path = ROOT/'reports/nov23_matched_learning_curve/remaining_cases.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    for side,v in report.items():
        print(side,v['all'],{k:v['profiles'][k] for k in ['paid','solo','complete_zero_bills','one_positive_bill','two_or_more_positive_bills','any_missing_bill']})


if __name__=='__main__':
    with threadpool_limits(2):
        run()
