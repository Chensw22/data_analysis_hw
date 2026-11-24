"""Isolate independent amount evidence and cross-service dependence before applying it to the release anchor."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_six_failure_archive_n24 import prediction_sets
from full_roster_solo_coverage_n24 import cached


def calibrate(x,y,p,t,q):
    def obj(w):
        v = p[t]+x[t]@w
        return ((np.logaddexp(0,v)-y[t]*v).mean()+.012*(w@w),
                x[t].T@(expit(v)-y[t])/len(t)+.024*w)
    sol = minimize(obj,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
    assert sol.success
    return p[q]+x[q]@sol.x


def conditional_auc(scores,y,keys):
    total,weight = 0.,0
    for k in np.unique(keys):
        ids = keys==k
        pair = y[ids].sum()*(~y[ids]).sum()
        if pair:
            total += pair*roc_auc_score(y[ids],scores[ids])
            weight += pair
    return float(total/weight)


def run():
    raw,y,g,fold = data()
    _,_,_,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    _,panels = prediction_sets(raw,y,g,fold,rows,of)
    positive = raw[BILLS].gt(0).sum(axis=1).to_numpy()
    report = {}
    for (tag,ix,f,base,p),(six,archive) in zip([('full',np.arange(len(y)),fold,ref,full[:,names.index('mitra96')]),
           ('historical',rows,of,champ,old[:,0])],panels,strict=True):
        strict = ((np.column_stack([six,archive])>=.5)==base[:,None]).all(1)
        yq = y[ix]
        error = yq!=base
        v = cached('conditional_bill_flow',tag,ix,f,g,y)
        logits = logit(v.clip(1e-6,1-1e-6))
        indep = logits[:,1]-logits[:,0]
        dependence = logits[:,2]-logits[:,1]
        anchor = logit(p.clip(.003,.997))
        free = (p>=.5)==base
        indep = np.where(positive[ix]>0,indep,0)
        dependence = np.where(positive[ix]>=2,dependence,0)
        static = dict(amount_only=anchor+.25*indep,dependence_only=anchor+.25*dependence)
        designs = dict(base=np.column_stack([np.ones(len(ix)),anchor]),
            with_evidence=np.column_stack([np.ones(len(ix)),anchor,indep.clip(-8,8),dependence.clip(-8,8)]))
        calibrated = {k:anchor.copy() for k in designs}
        for level in np.unique(f):
            ids = np.flatnonzero(f==level)
            for a,b in GroupKFold(4,shuffle=True,random_state=394).split(ids,groups=g[ix][ids]):
                t,q = ids[a],ids[b]
                assert not np.isin(g[ix][t],g[ix][q]).any()
                t = t[free[t]]
                for k,x in designs.items():
                    calibrated[k][q] = calibrate(x,yq,anchor,t,q)
        result = {}
        agree_free = strict & free
        bands = np.digitize(np.abs(p-.5),[.1,.2,.3,.4,.47])+6*base.astype(int)
        for k,score in dict(unchanged=anchor,**static,**calibrated).items():
            pred = np.where(free,score>=0,base)
            info = stats(pred,base,yq,g[ix],f)
            info['strict_fixed'] = int((strict&error&(pred==yq)).sum())
            info['strict_harmed'] = int((strict&~error&(pred!=yq)).sum())
            risk = expit(np.where(base,-score,score))
            use = agree_free
            info.update(stratified_error_auc=conditional_auc(risk[use],error[use],bands[use]),
                        strict_brier=float(np.mean((risk[use]-error[use])**2)),strict_n=int(use.sum()))
            result[k] = info
            print(tag,k,info['net'],info['strict_fixed'],info['strict_harmed'],
                  round(info['stratified_error_auc'],3),round(info['strict_brier'],4),flush=True)
        report[tag] = result
    (ROOT/'reports/nov23_matched_learning_curve/bill_evidence_transfer.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
