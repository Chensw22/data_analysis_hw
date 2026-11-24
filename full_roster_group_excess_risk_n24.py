"""Fit a common correction by its excess loss relative to the anchor across passenger groups."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit,logsumexp
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,BILLS
from full_roster_direct_edit_utility_n24 import features
from full_roster_disagreement_calibration_n24 import stats
from full_roster_repartition_gate_n24 import panel_data
from full_roster_six_failure_archive_n24 import prediction_sets

MODES = ('ordinary','worst_absolute','worst_excess','outcome_excess','shuffled_outcome_excess')


def parameters(x,base,y,cell,mode):
    if mode=='ordinary':
        cell = np.zeros(len(y),int)
    if mode in ('outcome_excess','shuffled_outcome_excess'):
        cell = 2*cell+y
        if mode=='shuffled_outcome_excess':
            rng = np.random.default_rng(877)
            cell = cell.copy()
            for label in (0,1):
                use = np.flatnonzero(y==label)
                cell[use] = cell[rng.permutation(use)]
    _,inverse,count = np.unique(cell,return_inverse=True,return_counts=True)
    prior = np.log(count/len(y))
    anchor_loss = np.logaddexp(0,base)-y*base
    correction = 0 if mode=='worst_absolute' else anchor_loss
    temperature = .08
    def obj(w):
        v = base+x@w
        dif = np.logaddexp(0,v)-y*v-correction
        risk = np.bincount(inverse,weights=dif)/count
        vals = prior+risk/temperature
        weight = np.exp(vals-logsumexp(vals))[inverse]/count[inverse]
        return temperature*logsumexp(vals)+.01*(w@w),x.T@(weight*(expit(v)-y))+.02*w
    res = minimize(obj,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options=dict(maxiter=300))
    assert res.success
    return res.x


def predict(x,ask,p,y,c,t,q):
    mean,sd = x[t].mean(0),x[t].std(0).clip(.15)
    a,b = [np.column_stack([np.ones(len(v)),((v-mean)/sd).clip(-5,5)]) for v in (x[t],ask[q])]
    return np.column_stack([b@parameters(a,logit(p[t].clip(.003,.997)),y[t],c[t],mode) for mode in MODES])


def run():
    raw,y,g,fold,names,half,full,ref,_,cols,rot = panel_data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    js = [names.index(k) for k in ('mitra96','v44_without_name','tabicl')]
    large = full[:,js]
    _,panels = prediction_sets(raw,y,g,fold,rows,of)
    size = raw.groupby(g).PassengerId.transform('size').to_numpy()
    n = raw[BILLS].gt(0).sum(axis=1).clip(upper=2).to_numpy()
    region = 2*n+(size==1)
    def design(s,ix):
        fields = features(raw.iloc[ix],size[ix],s)
        # Allow disagreement between the three models to vary with service use.
        return np.column_stack([fields,fields[:,:2]*(n[ix,None]==0),fields[:,:2]*(n[ix,None]>=2)])
    idx = np.arange(len(y))
    query = design(large,idx)
    shifts = []
    for panel,s in [(fold%2,half[:,js]),((fold//2)%2,rot[:,[cols.index(j) for j in js]])]:
        x = design(s,idx)
        c = 6*(s[:,0]>=.5)+region
        out = np.full((len(y),len(MODES)),np.nan)
        for f in np.unique(fold):
            q = idx[fold==f]
            t = idx[(panel==panel[q[0]]) & (fold!=f)]
            assert not np.isin(g[t],g[q]).any()
            out[q] = predict(x,query,s[:,0],y,c,t,q)
        assert np.isfinite(out).all()
        shifts.append(out)
    x = design(old,rows)
    c = 6*(old[:,0]>=.5)+region[rows]
    hist = np.full((len(rows),len(MODES)),np.nan)
    for f in np.unique(of):
        ix = np.flatnonzero(of==f)
        for a,b in GroupKFold(4,shuffle=True,random_state=633).split(ix,groups=og[ix]):
            t,q = ix[a],ix[b]
            t = t[(old[t,0]>=.5)==champ[t]]
            assert not np.isin(og[t],og[q]).any()
            hist[q] = predict(x,x,old[:,0],oy,c,t,q)
    assert np.isfinite(hist).all()
    report = {}
    for tag,ix,f,base,p,shift,bank in [('original',idx,fold,ref,large[:,0],shifts[0],panels[0]),
                ('rotated',idx,fold,ref,large[:,0],shifts[1],panels[0]),
                ('mean',idx,fold,ref,large[:,0],np.mean(shifts,axis=0),panels[0]),
                ('historical',rows,of,champ,old[:,0],hist,panels[1])]:
        six,archive = bank
        missed = (base!=y[ix]) & ((six>=.5)==base[:,None]).all(1)
        strict = missed & ((archive>=.5)==base[:,None]).all(1)
        report[tag] = {}
        for j,mode in enumerate(MODES):
            pred = np.where((p>=.5)==base,logit(p.clip(.003,.997))+shift[:,j]>=0,base)
            r = stats(pred,base,y[ix],g[ix],f)
            r.update(six_fixed=int((missed&(pred==y[ix])).sum()),archive_fixed=int((strict&(pred==y[ix])).sum()))
            if tag!='historical':
                new = ~np.isin(g[ix],og)
                r['new_net'] = int((pred[new]==y[ix][new]).sum()-(base[new]==y[ix][new]).sum())
            report[tag][mode] = r
        print(tag,[(k,r['corrected'],r['harmed'],r['archive_fixed']) for k,r in report[tag].items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/group_excess_risk.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
