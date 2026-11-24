"""Measure which known bills can vary at fixed passenger totals, service totals and positive support."""
import json

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from threadpoolctl import threadpool_limits

from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference


def balanced(amount):
    positive = amount>0
    r,c = np.nonzero(positive)
    variable = np.zeros_like(positive)
    if len(r)<4 or len(np.unique(r))<2 or len(np.unique(c))<2:
        return amount.copy(),variable,0
    incidence = np.zeros((sum(amount.shape),len(r)))
    incidence[r,np.arange(len(r))] = 1
    incidence[len(amount)+c,np.arange(len(r))] = 1
    basis = null_space(incidence)
    if not basis.shape[1]:
        return amount.copy(),variable,0
    variable[r,c] = (basis**2).sum(1)>1e-10
    # Non-cycle edges are bridges with known amounts. Only balance the remaining submatrix.
    current = variable.astype(float)
    row = (amount*variable).sum(1)
    col = (amount*variable).sum(0)
    for step in range(30000):
        current *= (row/current.sum(1).clip(1.e-12))[:,None]
        current *= col/current.sum(0).clip(1.e-12)
        if step%10==0 and np.allclose(current.sum(1),row,rtol=1e-10,atol=1e-8):
            break
    else:
        raise RuntimeError('Positive bill table did not balance')
    out = np.where(variable,current,amount)
    np.testing.assert_allclose(out.sum(1),amount.sum(1),atol=1e-7,rtol=2e-10)
    np.testing.assert_allclose(out.sum(0),amount.sum(0),atol=1e-7,rtol=2e-10)
    np.testing.assert_array_equal(out>0,positive)
    return out,variable,basis.shape[1]


def partitions(raw):
    groups = raw.PassengerId.str.split('_').str[0]
    amounts = raw[BILLS].fillna(0).to_numpy(float)
    counter = amounts.copy()
    variable = np.zeros_like(amounts,bool)
    dims = np.zeros(len(raw),int)
    for ids in raw.groupby(groups,sort=False).indices.values():
        counter[ids],variable[ids],dims[ids] = balanced(amounts[ids])
    np.testing.assert_array_equal(counter[~variable],amounts[~variable])
    return counter,variable,dims


def run():
    raw,y,g,fold = data()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId',*BILLS])
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    report = {}
    out = ROOT/'artifacts/nov24_bill_allocation_cycles'
    out.mkdir(exist_ok=True)
    for name,part in [('train',raw),('test',test)]:
        counter,variable,dims = partitions(part)
        group = part.PassengerId.str.split('_').str[0].to_numpy()
        row = variable.any(1)
        changed = np.abs(counter-part[BILLS].fillna(0).to_numpy()).max(1)>.01
        report[name] = dict(rows=len(part),orders=len(np.unique(group)),variable_orders=len(np.unique(group[row])),
             members_of_variable_orders=int((dims>0).sum()),variable_passengers=int(row.sum()),
             reconstructed_differently=int(changed.sum()),variable_charges=int(variable.sum()),
             per_service={k:int(v) for k,v in zip(BILLS,variable.sum(0))})
        if name=='train':
            report[name].update(base_errors_in_variable_passengers=int((row & (ref!=y)).sum()),
                                common_errors_in_variable_passengers=int((row & hard).sum()),
                                common_errors_in_variable_orders=int(((dims>0)&hard).sum()))
        np.savez_compressed(out/f'{name}.npz',passenger=part.PassengerId.to_numpy(str),counter=counter,
                            variable=variable,dimensions=dims)
    path = ROOT/'reports/nov23_matched_learning_curve/bill_allocation_cycles.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    with threadpool_limits(2):
        run()
