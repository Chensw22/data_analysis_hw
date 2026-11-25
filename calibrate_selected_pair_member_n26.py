"""Calibrate the probability that the selected passenger was wrong using untouched bookings."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from conflict_pair_trigger_n26 import inputs_for_trigger,match,train
from exhaustive_family_quorums_n26 import inputs
from full_roster_booking_validation_n23 import ROOT,data


def readout(mass,rank,personal,pairs,x,method):
    conditional = np.column_stack([rank,1-rank])
    marginal = mass[:,2,None]+mass[:,1,None]*conditional
    who = np.argmax(marginal if method=='joint' else personal,axis=1)
    i = np.arange(len(who))
    selected = pairs[i,who]
    agreement = (x[selected,:6]<0).mean(1)
    p = marginal[i,who] if method=='joint' else personal[i,who]
    feat = np.column_stack([logit(p.clip(.005,.995)),personal[i,who],marginal[i,who],
                      mass[:,0],abs(rank-.5),agreement,x[selected,-1]])
    return selected,p,feat


def stats(chosen,changed,wrong,fold):
    wins = changed&wrong[chosen]
    loss = changed&~wrong[chosen]
    deltas = wins.astype(int)-loss
    return dict(fixed=int(wins.sum()),harmed=int(loss.sum()),net=int(deltas.sum()),
                net_by_fold=np.bincount(fold[chosen],weights=deltas,minlength=8).astype(int).tolist())


def run():
    raw,y,g,f = data()
    _,cur,_ = inputs()
    side,wrong,x,a,b,d,gap = inputs_for_trigger(raw,y,g,f)
    report = {}
    for name,order in [('nearest',np.lexsort((b,a,d))),('confidence_gap',np.lexsort((b,a,d,-gap)))]:
        records = {'joint':[],'individual':[]}
        for k in np.unique(f):
            rows = np.flatnonzero(f==k)
            for tix,qix in GroupKFold(3,shuffle=True,random_state=153).split(rows,groups=g[rows]):
                t,q = rows[tix],rows[qix]
                qpair = match(a,b,order,q,len(y))
                two = [t[p] for _,p in GroupKFold(2,shuffle=True,random_state=972).split(t,groups=g[t])]
                output = {key:[] for key in records}
                for first,cal in (two,two[::-1]):
                    assert not np.isin(g[first],g[np.r_[cal,q]]).any() and not np.isin(g[cal],g[q]).any()
                    source = match(a,b,order,first,len(y))
                    calibration = match(a,b,order,cal,len(y))
                    joint = np.vstack([calibration,qpair])
                    m,r,p = train(x,wrong,first,source,joint)
                    for method,holder in output.items():
                        chosen,uncal,features = readout(m,r,p,joint,x,method)
                        n = len(calibration)
                        model = LogisticRegression(C=.1,max_iter=400).fit(features[:n],wrong[chosen[:n]])
                        forecast = model.predict_proba(features[n:])[:,1]
                        holder.append((chosen[n:],uncal[n:],forecast))
                for method,replicates in output.items():
                    c,p,r = [np.stack([h[j] for h in replicates]) for j in range(3)]
                    records[method].append((qpair,c[0],c[0]==c[1],p.mean(0),r.mean(0)))
        report[name] = {}
        for method,chunks in records.items():
            pairs,chosen,agree,p,r = [np.concatenate([c[j] for c in chunks]) for j in range(5)]
            assert pairs.size==len(np.unique(pairs))
            item = report[name][method] = {}
            for label,scope in [('all',cur['known']),('common',cur['common']),('new',cur['known']&cur['new'])]:
                valid = agree & (scope&(side==cur['parent']))[pairs].all(1)
                use = valid&(p>.5)
                stable = valid&(r>.5)
                budget = np.zeros(len(p),bool)
                for k in np.unique(f):
                    bank = np.flatnonzero(valid&(f[chosen]==k))
                    n = int(use[bank].sum())
                    budget[bank[np.argsort(-r[bank],kind='stable')[:n]]] = True
                assert use.sum()==budget.sum()
                item[label] = dict(pairs_checked=int(valid.sum()),uncalibrated=stats(chosen,use,wrong,f),
                   calibrated=stats(chosen,stable,wrong,f),same_budget=stats(chosen,budget,wrong,f),
                   raw_selected_success_rate=float(wrong[chosen[use]].mean()) if use.any() else None,
                   raw_selected_expected=float(p[use].mean()) if use.any() else None,
                   calibrated_probability_on_raw_selection=float(r[use].mean()) if use.any() else None)
            print(name,method,item['common'],flush=True)
    report['scope'] = ('Neither model fitting nor calibration uses a query booking. The calibrator is fitted to '
       'the observed error on the chosen passenger, after selection. Both halves must choose the same person. '
       'Raw and recalibrated versions use the same input pairs, matched relational training budget and eligibility.')
    (ROOT/'reports/nov23_matched_learning_curve/calibrate_selected_pair_member.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
