"""Check conditional bill direction in matched held-out passengers, with nested risk controls."""
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from threadpoolctl import threadpool_limits

from full_roster_conditional_pool_n23 import ROOT,BILLS,data,banks,latest_reference
from full_roster_gate_probability_n23 import gate_probabilities


def run():
    raw,y,g,fold = data()
    names,h,full = banks(raw,y,fold)
    j = names.index('mitra96')
    p,half = full[:,j],h[:,j]
    ref = latest_reference(y,g,fold,names,h,full)
    gate,_ = gate_probabilities(names,h,full,y,g,fold)
    np.testing.assert_array_equal(ref,gate>=.5)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    bills = raw[BILLS].to_numpy(float)
    charge = np.log1p(np.nan_to_num(bills))
    age = raw.Age.fillna(30).to_numpy()/20
    groups = pd.MultiIndex.from_arrays([raw.HomePlanet.fillna('?'),raw.CryoSleep.astype(str),
              *np.where(bills>0,1,np.where(np.isnan(bills),-1,0)).T]).factorize()[0]
    matches = np.full((3,len(y),len(BILLS)),-1)
    effects = np.zeros((2,len(y),len(BILLS)))
    scope = np.zeros_like(bills,dtype=bool)
    rng = np.random.default_rng(924)
    for f in np.unique(fold):
        for key in np.unique(groups[fold==f]):
            queries = np.flatnonzero((fold==f)&(groups==key))
            donors = np.flatnonzero((fold!=f)&(groups==key))
            if not len(donors):
                continue
            assert not np.isin(g[queries],g[donors]).any()
            same_parity = fold[donors]%2==f%2
            for i in np.flatnonzero(bills[queries[0]]>0):
                other = np.arange(len(BILLS))!=i
                for q in queries:
                    dist = ((charge[donors][:,other]-charge[q,other])**2).sum(1)+(age[donors]-age[q])**2
                    far = np.abs(charge[donors,i]-charge[q,i])>=.3
                    risk = np.floor(10*half[donors])==np.floor(10*p[q])
                    for pos,mask in enumerate((far,far&same_parity,far&same_parity&risk)):
                        ids = np.flatnonzero(mask)
                        if len(ids):
                            matches[pos,q,i] = donors[ids[np.argmin(dist[ids])]]
                    idx = np.flatnonzero(same_parity)
                    if len(idx)<10:
                        continue
                    d = donors[idx[np.argsort(dist[idx],kind='stable')[:32]]]
                    centered = charge[d,i]-charge[d,i].mean()
                    residual = y[d]-half[d]
                    curvature = half[d]*(1-half[d])
                    scope[q,i] = True
                    for t,v in enumerate((centered,rng.permutation(centered))):
                        slope = (v@(residual-residual.mean()))/(6+v**2@curvature)
                        effects[t,q,i] = np.clip(slope*(charge[q,i]-charge[d,i].mean()),-1,1)
    report = {}
    for k,mode in enumerate(('all','same_parity','same_parity_and_risk')):
        report[mode] = {}
        for i,service in enumerate(BILLS):
            q = np.flatnonzero(matches[k,:,i]>=0)
            d = matches[k,q,i]
            diff = y[q]!=y[d]
            higher = (charge[q,i]>charge[d,i])==(y[q]>y[d])
            report[mode][service] = dict(matched=len(q),distinct_donors=len(np.unique(d)),
                discordant=int(diff.sum()),higher_transported=int((diff&higher).sum()),
                hard_discordant=int((diff&hard[q]).sum()),hard_higher=int((diff&hard[q]&higher).sum()),
                folds=[[int((diff&higher&(fold[q]==f)).sum()),int((diff&(fold[q]==f)).sum())] for f in np.unique(fold)])
    for k,mode in enumerate(('real','shuffled')):
        pred = logit(gate.clip(.001,.999))+effects[k].sum(1)/scope.sum(1).clip(1)>=0
        report[mode] = dict(covered=int(scope.any(1).sum()),covered_hard=int((hard & scope.any(1)).sum()),
             fixed=int(((ref!=y)&(pred==y)).sum()),harmed=int(((ref==y)&(pred!=y)).sum()),
             hard_fixed=int((hard&(pred==y)).sum()))
    path = ROOT/'reports/nov23_matched_learning_curve/matched_charge_audit.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(report['real'],report['shuffled'])
    for k in ('all','same_parity','same_parity_and_risk'):
        r = report[k]['FoodCourt']
        print(k,r['higher_transported'],r['discordant'])


if __name__=='__main__':
    with threadpool_limits(2):
        run()
