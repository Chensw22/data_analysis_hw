"""Average calibration offsets over different partitions and depths of the passenger data."""
import json

import numpy as np
from scipy.special import expit, logit, logsumexp, softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats

GRID = np.linspace(-1.5,1.5,31)
LOG_PRIOR = np.log(softmax(-.5*(GRID/.55)**2))
MODES = ('constant','tree_average','permuted_bills','registration_only')


def inputs(raw,g):
    bills = raw[BILLS].to_numpy(float)
    size = raw.groupby(g).PassengerId.transform('size')
    cabin = raw.Cabin.str.split('/',expand=True)
    reg = np.column_stack([raw.CryoSleep.eq(True),raw.CryoSleep.isna(),raw.Age.lt(13),
                           raw.Age.ge(40),raw.VIP.eq(True),raw.HomePlanet.eq('Earth'),
                           raw.HomePlanet.eq('Europa'),raw.HomePlanet.isna(),raw.Destination.eq('55 Cancri e'),
                           cabin[0].eq('G'),cabin[2].eq('S'),size>1])
    return bills,reg


def binary(bills,reg,p,medians):
    positive = bills>0
    spend = np.column_stack([positive,bills>medians,np.isnan(bills),positive.sum(1)>1,
                             np.nansum(bills,axis=1)>1000,np.any(positive,axis=1)])
    base = np.column_stack([reg,p>=.5,np.abs(p-.5)>.25])
    return np.column_stack([base,spend]),base


def posterior_tree(train,query,contribution,possible,columns):
    depth = len(columns)
    leaves = 2**depth
    index = np.arange(depth-1,-1,-1)
    fit = train[:,columns].astype(int) @ (2**index)
    ask = query[:,columns].astype(int) @ (2**index)
    evidence = np.zeros((2*leaves,len(GRID)))
    np.add.at(evidence,leaves+fit,contribution)
    for node in reversed(range(1,leaves)):
        evidence[node] = evidence[2*node]+evidence[2*node+1]
    posterior = softmax(evidence+LOG_PRIOR,axis=1)
    log_mass = logsumexp(evidence+LOG_PRIOR,axis=1)
    stop = np.ones(2*leaves)
    for node in reversed(range(1,leaves)):
        child = log_mass[2*node]+log_mass[2*node+1]
        total = np.logaddexp(log_mass[node],child)
        stop[node] = np.exp(log_mass[node]-total)
        log_mass[node] = total-np.log(2)
    # Each query leaf can reach its own posterior or an ancestor's pruned model.
    mix = np.zeros((leaves,len(GRID)))
    for leaf in range(leaves):
        node = 1
        reach = 1.
        for level in range(depth+1):
            mix[leaf] += reach*stop[node]*posterior[node]
            reach *= 1-stop[node]
            if level<depth:
                node = 2*node+((leaf>>(depth-level-1))&1)
    assert np.allclose(mix.sum(1),1.)
    return log_mass[1],np.sum(mix[ask]*possible,axis=1)


def predict(bills,reg,y,t,q,pfit,pquery):
    x = logit(pfit.clip(1e-6,1-1e-6))
    z = x[:,None]+GRID
    contribution = y[t,None]*GRID-(np.logaddexp(0,z)-np.logaddexp(0,x)[:,None])
    query_prob = expit(logit(pquery.clip(1e-6,1-1e-6))[:,None]+GRID)
    constant = query_prob @ softmax(contribution.sum(0)+LOG_PRIOR)
    medians = np.array([np.median(c[c>0]) if (c>0).any() else np.inf for c in bills[t].T])
    train,train_reg = binary(bills[t],reg[t],pfit,medians)
    query,query_reg = binary(bills[q],reg[q],pquery,medians)
    # Shuffle the five bills together among donors with matching recorded sleep and paid-service count.
    rng = np.random.default_rng(832)
    source = np.arange(len(t))
    strata = 12*reg[t,0]+6*reg[t,1]+(bills[t]>0).sum(1)
    for v in np.unique(strata):
        idx = np.flatnonzero(strata==v)
        source[idx] = rng.permutation(idx)
    permuted,_ = binary(bills[t[source]],reg[t],pfit,medians)
    results = [constant]
    for a,b in ((train,query),(permuted,query),(train_reg,query_reg)):
        rng = np.random.default_rng(615)
        models = [posterior_tree(a,b,contribution,query_prob,rng.choice(a.shape[1],4,replace=False))
                  for _ in range(48)]
        mass,probs = zip(*models)
        results.append(softmax(mass) @ np.stack(probs))
    return np.column_stack(results)


def protocol(raw,y,g,rows,fold,source,target,same_fold):
    bills,reg = inputs(raw,g)
    result = np.full((len(rows),len(MODES)),np.nan)
    for f in np.unique(fold):
        outer = np.flatnonzero(fold==f)
        if same_fold:
            splits = [(outer[a],outer[b]) for a,b in GroupKFold(4,shuffle=True,random_state=528).split(outer,groups=g[rows[outer]])]
        else:
            splits = [(np.flatnonzero((fold%2==f%2)&(fold!=f)),outer)]
        for t,q in splits:
            assert not np.isin(g[rows[t]],g[rows[q]]).any()
            result[q] = predict(bills,reg,y,rows[t],rows[q],source[t],target[q])
    assert np.isfinite(result).all()
    return result


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    j = names.index('mitra96')
    a = protocol(raw,y,g,rows,of,old[:,0],old[:,0],True)
    b = protocol(raw,y,g,np.arange(len(y)),fold,h[:,j],full[:,j],False)
    c = protocol(raw,y,g,np.arange(len(y)),fold,full[:,j],full[:,j],True)
    out = ROOT/'artifacts/nov24_context_tree_calibration'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'oof.npz',row=rows,passenger=raw.PassengerId.to_numpy(str),y=y,
                        fold=fold,oldfold=of,historical=a,full=b,within_fold=c)
    unseen = ~np.isin(g,og)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    report = {}
    for i,name in enumerate(MODES):
        cast = lambda score,p,inc: np.where((p>=.5)==inc,score>=.5,inc)
        p = cast(b[:,i],full[:,j],ref)
        past = cast(a[:,i],old[:,0],champ)
        within = cast(c[:,i],full[:,j],ref)
        report[name] = dict(full=stats(p,ref,y,g,fold),within_fold=stats(within,ref,y,g,fold),
            historical=stats(past,champ,oy,og,of),unseen=stats(p[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
            fixed_among_147=int((hard & (p==y)).sum()),
            brier=float(np.mean((b[:,i]-y)**2)),original_brier=float(np.mean((full[:,j]-y)**2)))
        print(name,[report[name][k]['net'] for k in ('full','within_fold','historical','unseen')],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/context_tree_calibration.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
