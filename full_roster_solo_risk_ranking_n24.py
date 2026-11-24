"""Test whether alternative opinions reorder solo passengers at comparable baseline risk."""
import json

import numpy as np
from scipy.optimize import brentq,minimize
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities


def fit_residual(x,offset,y,pair):
    design,anchor,target = x,offset,y
    if pair:
        rng = np.random.default_rng(631)
        risk = np.minimum((expit(offset)*10).astype(int),9)
        pos,neg = [],[]
        for value in np.unique(risk):
            a,b = [np.flatnonzero((risk==value)&(y==side)) for side in (True,False)]
            if not len(a) or not len(b):
                continue
            n = 4*(len(a)+len(b))
            pos.extend(rng.choice(a,n))
            neg.extend(rng.choice(b,n))
        if not pos:
            return np.zeros(x.shape[1]),0.
        design = x[pos]-x[neg]
        anchor = offset[pos]-offset[neg]
        target = np.ones(len(anchor))
    def objective(w):
        z = design@w+anchor
        loss = np.mean(np.logaddexp(0,z)-target*z)+.05*(w@w)
        return loss,design.T@(expit(z)-target)/len(z)+.1*w
    sol = minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
    assert sol.success
    score = offset+x@sol.x
    intercept = brentq(lambda a:expit(score+a).sum()-y.sum(),-20,20)
    return sol.x,intercept


def predict(views,p,y,g,fold,base):
    # Freeze existing protected decisions; fit residuals to the score actually available at inference.
    free = (p>=.5)==base
    offset = logit(p.clip(.002,.998))
    threshold = np.abs(p-.5)<=.2
    out = {f'{mode}_{scope}':base.copy() for mode in ('point','pair') for scope in ('all','uncertain')}
    for f in np.unique(fold):
        outer = np.flatnonzero(fold==f)
        for a,b in GroupKFold(4,shuffle=True,random_state=759).split(outer,groups=g[outer]):
            t,q = outer[a],outer[b]
            assert not np.isin(g[t],g[q]).any()
            for scope,use in [('all',free),('uncertain',free&threshold)]:
                fit,ask = t[use[t]],q[use[q]]
                center = views[fit].mean(0)
                z = ((views-center)/views[fit].std(0).clip(.2)).clip(-6,6)
                for name,pair in [('point',False),('pair',True)]:
                    w,c = fit_residual(z[fit],offset[fit],y[fit],pair)
                    out[f'{name}_{scope}'][ask] = offset[ask]+z[ask]@w+c>=0
    return out


def features(raw,p,archive,families):
    scores = logit(archive.clip(.002,.998))
    anchor = logit(p.clip(.002,.998))
    contrasts = [scores[:,families==f].mean(1)-anchor for f in np.unique(families)]
    fees = raw[BILLS]
    return np.column_stack([anchor,*contrasts,raw.HomePlanet.eq('Earth'),raw.CryoSleep.eq(True),
             raw.Age.fillna(30)/30,fees.gt(0).sum(axis=1),np.log1p(fees.fillna(0)),fees.isna().sum(axis=1)])


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    gate,_ = gate_probabilities(names,h,full,y,g,fold)
    solo = raw.groupby(g).PassengerId.transform('size').to_numpy()==1
    ix = np.flatnonzero(solo)
    hist = np.flatnonzero(solo[rows])
    p = full[ix,names.index('mitra96')]
    op = old[hist,0]
    art = ROOT/'artifacts/nov24_remaining_cases'
    with np.load(art/'archive_scores.npz') as z:
        for k,v in [('row',rows),('old_fold',of),('fold',fold),('y',y),('passenger',raw.PassengerId)]:
            np.testing.assert_array_equal(z[k],v)
        families = np.array([s.split('/')[0] for s in z['names']])
        x = features(raw.iloc[ix],p,z['full'][ix],families)
        ox = features(raw.iloc[rows[hist]],op,z['historical'][hist],families)
    with np.load(art/'after_corroboration.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId)
        hard = z['remaining'] & solo
    b = predict(x,p,y[ix],g[ix],fold[ix],ref[ix])
    a = predict(ox,op,oy[hist],og[hist],of[hist],champ[hist])
    uncertain = solo & (np.abs(gate-.5)<=.2)
    unseen = ~np.isin(g,og)
    report = {}
    for k,new in b.items():
        pred,past = ref.copy(),champ.copy()
        pred[ix],past[hist] = new,a[k]
        report[k] = dict(full=stats(pred,ref,y,g,fold),historical=stats(past,champ,oy,og,of),
             unseen=stats(pred[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
             uncertain=stats(pred[uncertain],ref[uncertain],y[uncertain],g[uncertain],fold[uncertain]),
             hard_fixed=int((hard & (pred==y)).sum()))
        print(k,report[k]['full']['corrected'],report[k]['full']['harmed'],report[k]['historical']['net'],
              report[k]['unseen']['net'],report[k]['hard_fixed'],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/solo_risk_ranking.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
