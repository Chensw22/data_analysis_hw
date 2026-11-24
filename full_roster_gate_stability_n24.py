"""Test whether override decisions replicate across separate calibrator training orders."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from broad_booking_policy_replay_n23 import interval
from full_roster_adaptive_bill_prior_n23 import EVIDENCE, proposal_design
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference


def fit_probability(x, y, indices, query):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1,max_iter=500))
    model.fit(x[indices], y[indices])
    return model.predict_proba(query)[:,1]


def run():
    raw,y,group,fold = data()
    names,half,full = banks(raw,y,fold)
    reference = latest_reference(y,group,fold,names,half,full)
    base, small_base = full[:,4]>=.5, half[:,4]>=.5
    ids = [names.index(k) for k in EVIDENCE]
    ht, hx = proposal_design(half[:,:5],half[:,ids])
    ft, fx = proposal_design(full[:,:5],full[:,ids])
    keys = ['original','separate_mean','separate_vote','separate_unanimous','same_scale',
            'two_scale_mean','two_scale_agree','pooled_both']
    scores = {k:np.zeros(len(y)) for k in keys}
    history = []
    for f in np.unique(fold):
        t = np.flatnonzero((fold%2==f%2) & (fold!=f) & ht)
        outer = np.flatnonzero(fold==f)
        assert not np.isin(group[t],group[outer]).any()
        for a,b in GroupKFold(4,shuffle=True,random_state=462).split(outer,groups=group[outer]):
            calib,q = outer[a],outer[b]
            calib,q = calib[ft[calib]],q[ft[q]]
            assert not np.isin(group[calib],group[q]).any()
            small = fit_probability(hx,y!=small_base,t,fx[q])
            single = np.array([fit_probability(hx,y!=small_base,t[fold[t]==h],fx[q]) for h in np.unique(fold[t])])
            same = fit_probability(fx,y!=base,calib,fx[q])
            pool = np.vstack([hx[t],fx[calib]])
            target = np.r_[y[t]!=small_base[t],y[calib]!=base[calib]]
            pooled = fit_probability(pool,target,np.arange(len(target)),fx[q])
            for k,v in [('original',small),('separate_mean',single.mean(0)),
                        ('separate_vote',(single>=.5).mean(0)),('separate_unanimous',single.min(0)),
                        ('same_scale',same),('two_scale_mean',(small+same)/2),
                        ('two_scale_agree',np.minimum(small,same)),('pooled_both',pooled)]:
                scores[k][q] = v
            history.append(dict(fold=int(f),small_n=len(t),matched_n=len(calib),query_n=len(q)))
    assert sum(r['query_n'] for r in history)==int(ft.sum())
    shared = ((full>=.5)!=y[:,None]).all(1)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev=np.sort(z['development'])
    output = {}
    for k,s in scores.items():
        candidate = base ^ (s>=.5)
        if k=='original':
            np.testing.assert_array_equal(candidate,reference)
        delta=(candidate==y).astype(int)-(reference==y)
        output[k]=dict(accuracy=float((candidate==y).mean()),fixed=int((delta==1).sum()),harmed=int((delta==-1).sum()),
                       even_net=int(delta[fold%2==0].sum()), odd_net=int(delta[fold%2==1].sum()),
                       even_overrides=int((s[fold%2==0]>=.5).sum()),odd_overrides=int((s[fold%2==1]>=.5).sum()),
                       common_fixed=int((delta[shared]==1).sum()),additional_net=int(delta.sum()-delta[dev].sum()),
                       fold_net=[int(delta[fold==f].sum()) for f in np.unique(fold)], ci_pp=interval(delta,group))
    (ROOT/'reports/nov23_matched_learning_curve/gate_stability.json').write_text(
        json.dumps(dict(results=output,calibration_counts=history),indent=2)+'\n')
    for k,v in output.items(): print(k,round(v['accuracy']*100,3),v['fixed'],v['harmed'],v['fold_net'],v['additional_net'])


if __name__=='__main__':
    run()
