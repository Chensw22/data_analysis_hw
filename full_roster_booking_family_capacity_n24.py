"""Bound the gain from broad booking-size routing and evaluate learned mixtures on other orders."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats


def blend_weights(p,y,prior):
    # Penalize deviations from a prior giving the baseline one half and equal mass to each family.
    def objective(theta):
        w = softmax(np.log(prior)+theta)
        z = (p@w).clip(1e-7,1-1e-7)
        loss = -np.where(y,np.log(z),np.log1p(-z)).sum()+8*(w@np.log(w/prior))
        grad = p.T@((z-y)/(z*(1-z)))+8*(1+np.log(w/prior))
        return loss,w*(grad-w@grad)
    opt = minimize(objective,np.zeros(p.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500})
    assert opt.success,opt.message
    return softmax(np.log(prior)+opt.x)


def evaluate(p,base,y,g,fold,solo,names):
    free = (p[:,0]>=.5)==base
    menu = np.column_stack([base,p[:,1:]>=.5,(.75*p[:,[0]]+.25*p[:,1:])>=.5])
    menu[~free] = base[~free,None]
    labels = ['baseline',*names,*['quarter_'+s for s in names]]
    correct = menu==y[:,None]
    gains = correct.astype(int)-(base==y)[:,None]
    families = np.array([s.split('/')[0] for s in names])
    unique = np.unique(families)
    prior = np.r_[.5,[.5/(len(unique)*(families==k).sum()) for k in families]]
    output = {}
    preds = {}
    for scope,stratum in [('pooled',np.zeros(len(y),int)),('booking_size',solo.astype(int))]:
        upper = np.zeros(len(y),bool)
        hindsight = []
        winner = base.copy()
        mixed = base.copy()
        for s in np.unique(stratum):
            rows = np.flatnonzero(stratum==s)
            best = int(gains[rows].sum(0).argmax())
            upper[rows] = menu[rows,best]
            hindsight.append(dict(population=('all' if scope=='pooled' else ('solo' if s else 'multi')),
                                   n=len(rows),choice=labels[best],net=int(gains[rows,best].sum())))
        for f in np.unique(fold):
            outer = np.flatnonzero(fold==f)
            for a,b in GroupKFold(4,shuffle=True,random_state=855).split(outer,groups=g[outer]):
                t,q = outer[a],outer[b]
                assert not np.isin(g[t],g[q]).any()
                for s in np.unique(stratum):
                    train = t[stratum[t]==s]
                    ask = q[stratum[q]==s]
                    best = int(gains[train].sum(0).argmax())
                    winner[ask] = menu[ask,best]
                    learn = train[free[train]]
                    use = ask[free[ask]]
                    w = blend_weights(p[learn],y[learn],prior)
                    mixed[use] = (p[use]@w)>=.5
        output[scope] = dict(restricted_menu_hindsight=hindsight, correct_upper=int((upper==y).sum()))
        preds[scope+'_winner'] = winner
        preds[scope+'_mixture'] = mixed
    return output,preds


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    solo = raw.groupby(g).PassengerId.transform('size').to_numpy()==1
    art = ROOT/'artifacts/nov24_remaining_cases'
    with np.load(art/'archive_scores.npz') as z:
        for k,v in [('row',rows),('old_fold',of),('fold',fold),('y',y),('passenger',raw.PassengerId)]:
            np.testing.assert_array_equal(z[k],v)
        methods = z['names'].tolist()
        past = np.column_stack([old[:,0],z['historical']])
        current = np.column_stack([full[:,names.index('mitra96')],z['full']])
    with np.load(art/'after_corroboration.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId)
        hard = z['remaining']
    a,ap = evaluate(past,champ,oy,og,of,solo[rows],methods)
    b,bp = evaluate(current,ref,y,g,fold,solo,methods)
    report = dict(historical_menu=a,full_menu=b,options=1+2*len(methods))
    for k,p in bp.items():
        report[k] = dict(full=stats(p,ref,y,g,fold),historical=stats(ap[k],champ,oy,og,of),
                         solo=stats(p[solo],ref[solo],y[solo],g[solo],fold[solo]),
                         multi=stats(p[~solo],ref[~solo],y[~solo],g[~solo],fold[~solo]),
                         hard_fixed=int((hard&(p==y)).sum()))
    path = ROOT/'reports/nov23_matched_learning_curve/booking_family_capacity.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(a,b,{k:(report[k]['full']['net'],report[k]['historical']['net']) for k in bp})


if __name__=='__main__':
    with threadpool_limits(2):
        run()
