"""Reweight fully held calibration bookings to the observed profiles of the highest-risk query bookings."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from frozen_queue_shared_errors_n26 import paired
from nested_density_arbitration_n26 import OUT,case_features,read_experts
from target_mix_expert_routing_n26 import ROOT,batches,ticker


def weights_to_target(x, z):
    features = np.vstack([x,z])
    role = np.arange(len(features))>=len(x)
    m = LogisticRegression(C=.03,max_iter=400).fit(features,role)
    return np.clip(np.exp(m.decision_function(x))*len(x)/len(z),.05,12.)


def challenge(x, y, z, weight):
    reg = LogisticRegression(C=.06,max_iter=500)
    reg.fit(x,y,sample_weight=weight/weight.mean())
    return reg.predict_proba(z)[:,1]


def expert_inputs(raw,y,g,f):
    raw_x = case_features(raw,g)
    bundles = {}
    for first in range(0,8,2):
        held = np.isin(f,[first,first+1])
        train,both = np.flatnonzero(~held),np.flatnonzero(held)
        folder = OUT/f'pair{first}_{first+1}'
        pr = read_experts(raw,y,g,train,both,folder/'density.npz',folder/f'tuned{first}.npz')
        scores = np.zeros((len(y),pr.shape[1]))
        scores[both] = logit(pr.clip(.005,.995))
        bundles[first] = np.column_stack([scores,raw_x])
    return bundles


def run():
    raw,y,g,f,known,versions = paired()
    with np.load(ROOT/'artifacts/nov26_independently_calibrated_release_risk/held_risk.npz') as z:
        np.testing.assert_array_equal(np.sort(z['ids']),np.flatnonzero(known))
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[z['ids']])
        high = np.zeros(len(y),bool)
        high[z['ids']] = z['top20']
    current = versions[:,-1]
    common = (versions==current[:,None]).all(1)
    assert (known & common & high).sum()==1268
    assert (known & common & high & (current!=y)).sum()==525
    decisions = {}
    ess = []
    bundles = expert_inputs(raw,y,g,f)
    for b in batches(True):
        if b.tag!='full':
            continue
        k = int(f[b.ids[0]])
        pair = k-(k%2)
        a = np.flatnonzero(f==(k^1))
        assert np.all(f[b.ids]==k) and not np.isin(g[a], g[b.ids]).any()
        assert not np.isin(g[b.train_ids],g[b.ids]).any()
        xx = bundles[pair]
        scale = StandardScaler().fit(xx[a])
        x = scale.transform(xx[a]).clip(-5,5)
        for side in (False,True):
            # Target membership is measured only within a single release calibration batch.
            # Every row used to train the common risk estimate excludes every target booking.
            q = b.ids[high[b.ids] & common[b.ids] & (current[b.ids]==side)]
            if not len(q):
                continue
            z = scale.transform(xx[q]).clip(-5,5)
            weight = weights_to_target(x,z)
            ess.append(float(weight.sum()**2/np.square(weight).sum()))
            base = (challenge(x,y[a],z,np.ones(len(a)))>=.5)!=side
            adapted = (challenge(x,y[a],z,weight)>=.5)!=side
            repeats = []
            for half,_ in GroupKFold(2,shuffle=True,random_state=8261).split(a,groups=g[a]):
                w = weights_to_target(x[half],z)
                repeats.append((challenge(x[half],y[a[half]],z,w)>=.5)!=side)
            prec = np.logical_and.reduce(repeats)
            rules = dict(ordinary=base, matched=adapted, only_after_weighting=adapted & ~base,
                         two_calibration_halves=adapted & prec, confirmed_new=adapted & prec & ~base)
            for name,act in rules.items():
                decisions.setdefault(name,np.zeros(len(y),bool))[q]=act
    target = known & high & common
    wrong = y!=current
    report = dict(eligible=int(target.sum()), errors=int((target & wrong).sum()),
                  effective_calibration_n_range=[min(ess),max(ess)], strategies={})
    for name,act in decisions.items():
        assert not (act & ~target).any()
        report['strategies'][name] = dict(all=ticker(target,wrong,act),predicted_no=ticker(target & ~current,wrong,act),
            predicted_yes=ticker(target & current,wrong,act),
            fold_net=[int((np.where(wrong,1,-1)*act*(f==k)).sum()) for k in range(8)])
        print(name,report['strategies'][name],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/transfer_to_high_risk_orders.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
