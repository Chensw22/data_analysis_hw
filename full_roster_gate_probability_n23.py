"""Recover the learned override probability and test calibrated blends at equal decisions."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from broad_booking_policy_replay_n23 import interval
from full_roster_adaptive_bill_prior_n23 import EVIDENCE, proposal_design
from full_roster_cohort_prevalence_n23 import fraction
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_unexpected_peers_n23 import reference_score


def gate_probabilities(names, half, full, y, group, fold):
    indices = [names.index(k) for k in EVIDENCE]
    ht, hx = proposal_design(half[:,:5], half[:,indices])
    ft, fx = proposal_design(full[:,:5], full[:,indices])
    base = full[:,4] >= .5
    prob = full[:,4].copy()
    for f in np.unique(fold):
        t = (fold % 2 == f % 2) & (fold != f) & ht
        q = (fold == f) & ft
        assert not np.isin(group[t],group[q]).any()
        learner = make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500))
        learner.fit(hx[t],y[t] != (half[t,4] >= .5))
        flip = learner.predict_proba(fx[q])[:,1]
        prob[q] = np.where(base[q],1-flip,flip)
    return prob, ft


def run():
    raw,y,group,fold = data()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,group,fold,names,half,full)
    old = reference_score(full,ref,names.index('balanced_full'))
    gate, conflict = gate_probabilities(names,half,full,y,group,fold)
    np.testing.assert_array_equal(gate >= .5,ref)
    changed = ref != (full[:,4] >= .5)
    earth = (raw.HomePlanet.eq('Earth') & raw.CryoSleep.eq(True)).to_numpy()
    route = raw.Destination.eq('TRAPPIST-1e').to_numpy()
    side = raw.Cabin.str.endswith(('/P','/S'),na=False).to_numpy()
    focus = earth & route & side & (old <= .3)
    assert focus.sum()==121 and changed[focus].all()
    shared = ((full>=.5)!=y[:,None]).all(1)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    profiles, policies = {}, {}
    for source, p in [('export_score',old),('override_probability',gate)]:
        profiles[source] = {}
        for label,mask in [('all',np.ones(len(y),bool)),('conflict',conflict),('changed',changed),('focus',focus)]:
            error = (p[mask] - y[mask])**2
            profiles[source][label] = dict(n=int(mask.sum()),transported=int(y[mask].sum()),
                  expected=float(p[mask].sum()),brier=float(error.mean()), auc=float(roc_auc_score(y[mask],p[mask])),
                  at_most_30=int((p[mask]<=.3).sum()))
        for other, peer in [('ordinary',full[:,:4].mean(1)), ('restored',full[:,names.index('restored_full')])]:
            for mode in ['quarter','learn']:
                out = p.copy()
                for f in np.unique(fold):
                    outer = np.flatnonzero(fold == f)
                    for a,b in GroupKFold(4,shuffle=True,random_state=462).split(outer,groups=group[outer]):
                        donors,q = outer[a],outer[b]
                        # The two starting scores differ only in expert conflicts.
                        donors,q = donors[conflict[donors]],q[conflict[q]]
                        assert not np.isin(group[donors],group[q]).any()
                        w = .25 if mode == 'quarter' else fraction(p[donors],peer[donors],y[donors])
                        out[q] = (1-w)*p[q]+w*peer[q]
                delta = ((out>=.5)==y).astype(int) - (ref==y)
                policies[source+'_'+other+'_'+mode] = dict(accuracy=float(((out>=.5)==y).mean()),
                    fixed=int((delta==1).sum()),harmed=int((delta==-1).sum()),additional_net=int(delta.sum()-delta[dev].sum()),
                    common_fixed=int((delta[shared]==1).sum()), focus_fixed=int((delta[focus]==1).sum()),
                    focus_harmed=int((delta[focus]==-1).sum()), fold_net=[int(delta[fold==f].sum()) for f in np.unique(fold)],
                    ci_pp=interval(delta,group))
    report = dict(focus_by_fold=[int(focus[fold==f].sum()) for f in np.unique(fold)],
                  profiles=profiles,policies=policies)
    (ROOT/'reports/nov23_matched_learning_curve/gate_probability.json').write_text(json.dumps(report,indent=2)+'\n')
    print(profiles)
    for k,v in policies.items():
        print(k,round(v['accuracy']*100,3),v['fixed'],v['harmed'],v['fold_net'],v['additional_net'])


if __name__=='__main__':
    run()
