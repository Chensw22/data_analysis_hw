"""Learn bounded corrections to the release log odds using direct net accuracy on calibration orders."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_solo_coverage_n24 import cached

COST = .03


def features(raw,size,s):
    logits = logit(s.clip(.003,.997))
    fees = raw[BILLS]
    return np.column_stack([logits[:,1:]-logits[:,:1],logits[:,0],raw.CryoSleep.eq(True),raw.CryoSleep.isna(),
            raw.Age.fillna(30),raw.Age.isna(),raw.HomePlanet.eq('Earth'),raw.HomePlanet.eq('Europa'),
            raw.Destination.eq('TRAPPIST-1e'),raw.Cabin.str.endswith('/S',na=False),
            size,np.log1p(fees.fillna(0)),fees.gt(0).sum(axis=1),fees.isna().sum(axis=1)])


def gains(decision,base,y):
    return (decision==y).astype(float)-(base==y)-COST*(decision!=base)


def scan(prefix,axis,y,base,panels,robust):
    live = axis!=0
    if not live.any():
        return -np.inf,0.
    switch = -prefix[live]/axis[live]
    bounds = np.unique(np.r_[-1.5,switch[(switch>-1.5)&(switch<1.5)],1.5])
    # Query interiors to avoid making a policy depend on floating-point equality.
    candidates = (bounds[:-1]+bounds[1:])/2
    order = np.argsort(switch)
    delta = np.sign(axis[live])*((2*y[live].astype(float)-1)-COST*(1-2*base[live].astype(float)))
    starting = gains(np.where(live,axis<0,prefix>=0),base,y)
    gain = []
    for panel in ([panels,~panels] if robust else [np.ones(len(y),bool)]):
        idx = np.searchsorted(switch[order],candidates)
        totals = np.r_[0,(delta*panel[live])[order].cumsum()]
        gain.append((starting[panel].sum()+totals[idx])/panel.sum())
    reward = np.min(gain,axis=0)
    best = np.flatnonzero(np.isclose(reward,reward.max(),rtol=0,atol=1e-12))
    pick = best[np.argmin(abs(candidates[best]))]
    return reward[pick],candidates[pick]


def fit(a,offset,y,group,robust):
    panels = np.zeros(len(y),bool)
    left,right = next(GroupKFold(2,shuffle=True,random_state=633).split(a,groups=group))
    assert not np.isin(group[left],group[right]).any()
    panels[left] = True
    base = offset>=0
    theta = np.zeros(a.shape[1])
    value = 0.
    for _ in range(5):
        curr = offset+a@theta
        best = value
        choice = None
        for j in range(a.shape[1]):
            score,weight = scan(curr-a[:,j]*theta[j],a[:,j],y,base,panels,robust)
            if score>best+1e-10:
                best,choice = score,(j,weight)
        if choice is None:
            break
        j,weight = choice
        theta[j] = weight
        actual = gains(offset+a@theta>=0,base,y)
        value = min(actual[panels].mean(),actual[~panels].mean()) if robust else actual.mean()
        np.testing.assert_allclose(value,best,atol=1e-10)
    return theta


def predict(train,test,p,q,y,g):
    avg,sd = train.mean(0),train.std(0).clip(.15)
    a,b = [np.column_stack([np.ones(len(v)),((v-avg)/sd).clip(-5,5)]) for v in (train,test)]
    t,ask = [logit(v.clip(.003,.997)) for v in (p,q)]
    def logistic(w):
        v = t+a@w
        loss = (np.logaddexp(0,v)-y*v).mean()+.01*(w@w)
        return loss,a.T@(expit(v)-y)/len(y)+.02*w
    opt = minimize(logistic,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B')
    assert opt.success
    coefs = [opt.x,fit(a,t,y,g,False),fit(a,t,y,g,True)]
    return np.column_stack([ask+b@w>=0 for w in coefs])


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    js = [names.index(k) for k in ('mitra96','v44_without_name','tabicl')]
    half,bank = h[:,js],full[:,js]
    size = raw.groupby(g).PassengerId.transform('size').to_numpy()
    xa,xb = [features(raw,size,s) for s in (half,bank)]
    xo = features(raw.iloc[rows],size[rows],old)
    current = np.empty((len(y),3),bool)
    for f in np.unique(fold):
        t,q = (fold%2==f%2)&(fold!=f),fold==f
        assert not np.isin(g[t],g[q]).any()
        current[q] = predict(xa[t],xb[q],half[t,0],bank[q,0],y[t],g[t])
    previous = np.empty((len(oy),3),bool)
    for f in np.unique(of):
        ix = np.flatnonzero(of==f)
        for a,b in GroupKFold(4,shuffle=True,random_state=633).split(ix,groups=og[ix]):
            t,q = ix[a],ix[b]
            t = t[(old[t,0]>=.5)==champ[t]]
            assert not np.isin(og[t],og[q]).any()
            previous[q] = predict(xo[t],xo[q],old[t,0],old[q,0],oy[t],og[t])
    result = {}
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('y',y),('fold',fold),('passenger',raw.PassengerId)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    ix = np.arange(len(y))
    six = np.column_stack([cached(k,'full',ix,fold,g,y) for k in ('response_channels','broad_error_transport_projection')])
    missed = (ref!=y) & ((six>=.5)==ref[:,None]).all(1)
    for j,k in enumerate(('log_loss','net_edits','min_panel_net_edits')):
        pred = np.where((bank[:,0]>=.5)==ref,current[:,j],ref)
        past = np.where((old[:,0]>=.5)==champ,previous[:,j],champ)
        new = ~np.isin(g,og)
        result[k] = dict(full=stats(pred,ref,y,g,fold),historical=stats(past,champ,oy,og,of),
                        new_orders=stats(pred[new],ref[new],y[new],g[new],fold[new]),
                        hard_fixed=int((hard & (pred==y)).sum()),six_fixed=int((missed & (pred==y)).sum()))
        r = result[k]
        print(k,r['full']['corrected'],r['full']['harmed'],r['historical']['net'],r['new_orders']['net'],r['hard_fixed'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/direct_edit_utility.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
