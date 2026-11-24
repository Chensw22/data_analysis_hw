"""Compare measured passenger allocations with maximum-entropy bills holding both margins fixed."""
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_order_coverage_n24 import comparison,profile
from full_roster_permutation_selection_n24 import cat
from full_roster_two_view_reliability_n24 import design

MODES = ('personal','margins','measured','reconstructed','same_fit_reconstructed')


def matrices(raw,g,counter):
    reg,_ = design(raw,g)
    fee = raw[BILLS]
    positive = fee.gt(0)
    missing = fee.isna()
    amount = fee.fillna(0).to_numpy()
    personal = np.column_stack([reg,raw.CryoSleep.eq(True),raw.CryoSleep.isna(),positive,missing,
                                np.log1p(amount.sum(1))])
    common = np.column_stack([personal,np.log1p(fee.groupby(g).transform('sum')),
                              positive.groupby(g).transform('sum'),missing.groupby(g).transform('sum')])
    return (np.column_stack([personal,np.log1p(amount)]),common,
            np.column_stack([common,np.log1p(amount)]),np.column_stack([common,np.log1p(counter)]))


def predict(xs,y,g,t,q):
    assert not np.isin(g[t],g[q]).any()
    out = []
    for i,x in enumerate(xs):
        model = cat(497).fit(x[t],y[t])
        out.append(model.predict_proba(x[q],thread_count=3)[:,1])
        if i==2:
            counterfactual = model.predict_proba(xs[3][q],thread_count=3)[:,1]
    return np.column_stack([*out,counterfactual])


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    gate,_ = gate_probabilities(names,h,full,y,g,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    with np.load(ROOT/'artifacts/nov24_bill_allocation_cycles/train.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId)
        affected = z['variable'].any(1)
        xs = matrices(raw,g,z['counter'])
    a = protocol(xs,y,g,rows,of,'historical',predict,MODES,'fixed_bill_margins')
    b = protocol(xs,y,g,np.arange(len(y)),fold,'full',predict,MODES,'fixed_bill_margins')
    np.testing.assert_array_equal(b[~affected,2],b[~affected,4])
    np.testing.assert_array_equal(a[~affected[rows],2],a[~affected[rows],4])
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    cells = profile(raw)[0]
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId','Age'])
    target = profile(test)[0]
    unseen = ~np.isin(g,og)
    reports = {}
    def assess(name,p,past):
        fixed = hard & (p==y)
        delta = (p==y).astype(int)-(ref==y)
        weighted = comparison(cells,target,np.ones(len(y),bool),{'candidate':delta})
        r = dict(full=stats(p,ref,y,g,fold),historical=stats(past,champ,oy,og,of),
                 unseen=stats(p[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                 within_variable=stats(p[affected],ref[affected],y[affected],g[affected],fold[affected]),
                 fixed_among_147=int(fixed.sum()),repeated=int((fixed[rows] & (past==oy) & (champ!=oy)).sum()),
                 target_weighted_pp=weighted['results']['candidate']['covered_weighted_pp'])
        reports[name] = r
        print(name,r['full']['corrected'],r['full']['harmed'],r['historical']['net'],r['fixed_among_147'],flush=True)
    for j,name in enumerate(MODES):
        assess(name,b[:,j]>=.5,a[:,j]>=.5)
        past = np.where((old[:,0]>=.5)==champ,.75*old[:,0]+.25*a[:,j]>=.5,champ)
        assess(name+'_quarter',.75*gate+.25*b[:,j]>=.5,past)
    def apply(p,mat):
        return logit(p.clip(.001,.999))+logit(mat[:,2].clip(.001,.999))-logit(mat[:,4].clip(.001,.999))>=0
    p = np.where(affected,apply(gate,b),ref)
    past = np.where(affected[rows] & ((old[:,0]>=.5)==champ),apply(old[:,0],a),champ)
    assess('allocation_change_only',p,past)
    reports['same_fit_measured_vs_reconstructed'] = stats(b[:,2]>=.5,b[:,4]>=.5,y,g,fold)
    path = ROOT/'reports/nov23_matched_learning_curve/fixed_bill_margins.json'
    path.write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
