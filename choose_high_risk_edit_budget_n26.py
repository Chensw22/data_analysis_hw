"""Estimate a correction budget on separate calibration orders and test it in the consensus risk tail."""
import json

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from frozen_queue_shared_errors_n26 import paired
from target_mix_expert_routing_n26 import ROOT,batches,ticker
from transfer_to_high_risk_orders_n26 import challenge,expert_inputs,weights_to_target


def cutoff(cal_risk, correct_change, w, proposed):
    # Choose how far down the query ranking to act using only labeled calibration bookings.
    best,cut = 0.,np.inf
    for frac in (.05,.10,.25,.50):
        threshold = np.quantile(proposed,1-frac,method='higher')
        m = cal_risk >= threshold
        if not m.any():
            continue
        n = w[m].sum()**2/(w[m]@w[m])
        precision = (n*np.average(correct_change[m],weights=w[m])+2.)/(n+4.)
        gain = (proposed>=threshold).sum()*(2*precision-1)
        if gain>best:
            best,cut = gain,threshold
    return cut


def run():
    raw,y,g,f,known,v=paired()
    parent=v[:,-1]
    with np.load(ROOT/'artifacts/nov26_independently_calibrated_release_risk/held_risk.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[z['ids']])
        high=np.zeros(len(y),bool)
        high[z['ids']]=z['top20']
    eligible=known & high & (v==parent[:,None]).all(1)
    bundles=expert_inputs(raw,y,g,f)
    forecasts={}
    for b in batches(True):
        if b.tag!='full':
            continue
        k=int(f[b.ids[0]])
        a=np.flatnonzero(f==(k^1))
        features=bundles[k-k%2]
        assert not np.isin(g[a],g[b.ids]).any()
        for side in (False,True):
            q=b.ids[eligible[b.ids] & (parent[b.ids]==side)]
            if not len(q):
                continue
            decisions=[]
            ranks=[]
            for tr,cv in GroupKFold(2,shuffle=True,random_state=1843).split(a,groups=g[a]):
                t,c=a[tr],a[cv]
                assert not np.isin(g[t],g[c]).any()
                scale=StandardScaler().fit(features[t])
                xt,xc,xq=[scale.transform(features[s]).clip(-5,5) for s in (t,c,q)]
                outcome=challenge(xt,y[t],np.vstack([xc,xq]),np.ones(len(xt)))
                risk=1-outcome if side else outcome
                history,query=risk[:len(c)],risk[len(c):]
                w=weights_to_target(xc,xq)
                limit=cutoff(history,y[c]!=side,w,query)
                decisions.append(query>=limit)
                ranks.append(query)
            score=np.mean(ranks,axis=0)
            rules={'selected_any':np.logical_or.reduce(decisions),'selected_twice':np.logical_and.reduce(decisions)}
            for frac in (.05,.10,.25,.50):
                rules[f'fixed_{frac}']=score>=np.quantile(score,1-frac,method='higher')
            for name,d in rules.items():
                forecasts.setdefault(name,np.zeros(len(y),bool))[q]=d
    assert eligible.sum()==1268 and (eligible & (y!=parent)).sum()==525
    result={n:dict(all=ticker(eligible,y!=parent,a),fold_net=[
        int((a*np.where(y!=parent,1,-1)*(f==k)).sum()) for k in range(8)]) for n,a in forecasts.items()}
    print(result,flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/choose_high_risk_edit_budget.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
