"""Upweight baseline mistakes according to their measured influence on independent bookings."""
import json

import numpy as np
from scipy.linalg import solve
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from deployment_booking_mix_n23 import passenger_vectors
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats

MODES = ('uniform', 'all_mistakes', 'selected', 'shuffled', 'opposite')


def design(raw):
    x = passenger_vectors(raw)
    bills = raw[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].fillna(0).to_numpy()
    group = raw.PassengerId.str.split('_').str[0]
    return np.column_stack([x, bills / np.maximum(1,bills.sum(1,keepdims=True)),
                            np.log1p(bills.sum(1))/6, np.log1p(group.map(group.value_counts()))])


def basis(x, t):
    z = ((x-x[t].mean(0))/x[t].std(0).clip(.2)).clip(-6,6)
    rng = np.random.default_rng(418)
    direction = rng.normal(size=(z.shape[1],96))/np.sqrt(z.shape[1])
    hidden = np.tanh(z@direction+rng.uniform(-1,1,96))
    return np.column_stack([np.ones(len(x)), z, hidden])


def fitted(x, y, offset, w, penalty=32.):
    beta = np.zeros(x.shape[1])
    for _ in range(50):
        pred = expit(offset+x@beta)
        err = w*(pred-y)
        grad = x.T@err+penalty*beta
        curv = w*pred*(1-pred)
        hessian = (x.T*curv)@x + penalty*np.eye(x.shape[1])
        if np.max(np.abs(grad)) < 1e-7:
            return beta, hessian, pred
        step = solve(hessian, grad, assume_a='pos')
        current = np.sum(w*(np.logaddexp(0,offset+x@beta)-y*(offset+x@beta)))+.5*penalty*(beta@beta)
        for backoff in range(20):
            trial = beta-step*2.**(-backoff)
            v = offset+x@trial
            if np.sum(w*(np.logaddexp(0,v)-y*v))+.5*penalty*(trial@trial) <= current+1e-10:
                beta = trial
                break
        else:
            raise ArithmeticError('Newton step did not reduce the training objective')
    raise ArithmeticError('Newton iterations did not converge')


def influence(x, y, offset, group, seed):
    values = np.full(len(y),np.nan)
    for a,b in GroupKFold(2,shuffle=True,random_state=seed).split(x, groups=group):
        assert not np.isin(group[a],group[b]).any()
        beta,hessian,pred = fitted(x[a],y[a],offset[a],np.ones(len(a)))
        val_grad = x[b].T@(expit(offset[b]+x[b]@beta)-y[b])/len(b)
        adjoint = solve(hessian,val_grad,assume_a='pos')
        # Positive benefit is a first-order decrease in the average held-out log loss.
        values[a] = (pred-y[a])*(x[a]@adjoint)
    assert np.isfinite(values).all()
    return values


def weights(y, offset, value, seed):
    wrong = (offset>=0) != y
    rng = np.random.default_rng(seed)
    modes = {key:np.ones(len(y)) for key in MODES}
    for side in (False,True):
        ids = np.flatnonzero(wrong & ((offset>=0)==side))
        ordered = ids[np.argsort(value[ids],kind='stable')]
        # Identical extra training mass by class and predicted direction for all four policies.
        n = len(ids)//2
        modes['all_mistakes'][ids] += n/len(ids)
        modes['selected'][ordered[-n:]] += 1
        modes['shuffled'][rng.permutation(ids)[:n]] += 1
        modes['opposite'][ordered[:n]] += 1
        for key in MODES[1:]:
            np.testing.assert_allclose(modes[key][ids].sum(),len(ids)+n)
    return modes


def run():
    _, oy, og, rows, ofold, _, champ, _, _, _, _, old, _ = aligned()
    raw,y,g,fold = data()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    k = names.index('mitra96')
    teacher = logit(half[:,k].clip(.0001,.9999))
    base = logit(full[:,k].clip(.0001,.9999))
    oldbase = logit(old[:,0].clip(.0001,.9999))
    x = design(raw)
    adjust = {key:np.full(len(y),np.nan) for key in MODES}
    overlap = []
    for f in np.unique(fold):
        q = np.flatnonzero(fold == f)
        t = np.flatnonzero((fold != f) & (fold%2 == f%2))
        assert not np.isin(g[t],g[q]).any()
        xx = basis(x,t)
        first,second = [influence(xx[t],y[t],teacher[t],g[t],s) for s in (537,538)]
        w = weights(y[t],teacher[t],(first+second)/2,622)
        wrong = (teacher[t]>=0)!=y[t]
        chosen = [weights(y[t],teacher[t],d,622)['selected']>1 for d in (first,second)]
        overlap.append(dict(mistakes=int(wrong.sum()), selected=int(chosen[0].sum()),
              both_selected=int(np.sum(chosen[0] & chosen[1])), positively_aligned=int(((first>0)&(second>0)&wrong).sum())))
        for mode in MODES:
            beta,_,_ = fitted(xx[t],y[t],teacher[t],w[mode])
            adjust[mode][q] = xx[q]@beta
    common = ((full >= .5) != y[:,None]).all(1)
    hard = common[rows] & (champ != oy)
    unseen = ~np.isin(g,og)
    group_size = raw.PassengerId.str.split('_').str[0].map(raw.PassengerId.str.split('_').str[0].value_counts()).to_numpy()
    report = dict(training_overlap=overlap,results={})
    for mode,d in adjust.items():
        assert np.isfinite(d).all()
        report['results'][mode] = {}
        for weight in (1.,.25):
            pred = np.where((base>=0)==ref, base+weight*d >= 0, ref)
            past = np.where((oldbase>=0)==champ, oldbase+weight*d[rows] >= 0, champ)
            v = dict(full=stats(pred,ref,y,g,fold),historical=stats(past,champ,oy,og,ofold),
                     new_orders=stats(pred[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                     common_fixed=int((common & (pred==y)).sum()),persistent_fixed=int((hard & (past==oy) & (pred[rows]==oy)).sum()),
                     solo_misses_fixed=int((common & y & (group_size==1) & pred).sum()))
            report['results'][mode][str(weight)] = v
            print(mode,weight,[v[s]['net'] for s in ('historical','full','new_orders')],v['common_fixed'])
    path = ROOT/'reports/nov23_matched_learning_curve/mistake_influence.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
