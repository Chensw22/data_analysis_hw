"""Measure shared vote patterns and learn correlated expert errors conditional on billing."""
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel, predictions
from broad_bill_exception_n23 import load, strata
from broad_local_odds_comparator_n23 import REPORT
from broad_partition_evidence_n23 import EXTRA, SOURCES, SEEDS, write
from broad_release_miss_coverage_n23 import summarize
from released_correction_audit_n23 import development


def fit_class(votes):
    # One pseudo-observation in each cell keeps pair and singleton marginals consistent.
    n = len(votes)
    positive = votes.sum(0)
    ones = votes.T@votes
    table = np.stack([n-positive[:,None]-positive[None,:]+ones,positive[None,:]-ones,
                      positive[:,None]-ones,ones],axis=2)
    joint = (table+1)/(n+4)
    marginal = np.stack([n-positive+2,positive+2],axis=1)/(n+4)
    independent = (marginal[:,None,:,None]*marginal[None,:,None,:]).reshape(joint.shape)
    interaction = np.log(joint/independent)
    mi = (joint * interaction).sum(2)
    # Constant shifts leave the maximum spanning tree unchanged; nonzero edge
    # costs keep pairs with zero information available to connect components.
    parent,child = minimum_spanning_tree(-1-mi).nonzero()
    assert len(parent)==votes.shape[1]-1
    return np.log(marginal),parent,child,interaction[parent,child]


def likelihood(x, model):
    margin,parent,child,edge = model
    single = margin[np.arange(x.shape[1]),x].sum(1)
    code = 2*x[:,parent]+x[:,child]
    tree = single + edge[np.arange(len(parent)),code].sum(1)
    return np.column_stack([single,tree])


def evidence(x,y,bill,t,q):
    out = np.empty((len(q),2))
    for state in np.unique(bill[q]):
        subset = t[bill[t]==state]
        local = bill[q]==state
        yes,no = subset[y[subset]],subset[~y[subset]]
        out[local] = (likelihood(x[q[local]],fit_class(x[yes])) -
                      likelihood(x[q[local]],fit_class(x[no])) + np.log((len(yes)+1)/(len(no)+1)))
    return out


def collisions(x,y,bill,base,masks):
    # Grouping by vote code, billing state, and baseline decision gives an
    # optimistic ceiling measured retrospectively on the very same rows.
    _,group = np.unique(np.column_stack([bill,base,x]),axis=0,return_inverse=True)
    n = np.bincount(group)
    yes = np.bincount(group,weights=y).astype(int)
    no = n-yes
    against = np.where(y,no[group],yes[group])
    with_label = n[group]-against
    return dict(distinct=len(n),passengers_in_singletons=int((n[group]==1).sum()),
                empirical_min_errors=int(np.minimum(yes,no).sum()),
                minority_or_tied={k:int((mask & (against>=with_label)).sum()) for k,mask in masks.items()},
                shares_with_opposite={k:int((mask & (against>0)).sum()) for k,mask in masks.items()})


def run():
    row,y,base,old,_,_,_ = panel()
    ids,labels,frame,g,_ = load()
    np.testing.assert_array_equal(ids,row)
    np.testing.assert_array_equal(labels,y)
    labels,release,_ = development()
    np.testing.assert_array_equal(labels,y)
    miss = np.logical_and.reduce([p!=y for p in release.values()])
    bill = strata(frame)
    masks = dict(all_release=miss,paid=miss&y&(bill==2),zero=miss&~y&(bill==0))
    extra = {**EXTRA,**dict(list(SOURCES.items())[:4])}
    histories,structures={},{}
    for i,seed in enumerate(SEEDS):
        _,others,fold = predictions(row,y,seed,extra)
        votes = (np.column_stack([old[i],others])>=.5).astype(int)
        assert votes.shape==(len(y),50)
        structures[str(seed)] = collisions(votes,y,bill,base[i]>=.5,masks)
        outputs = {k:np.full(len(y),np.nan) for k in ("base","independent","tree","both")}
        for f in np.unique(fold):
            ix = np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
                t,q = ix[a],ix[b]
                summed = {k:np.zeros(len(q)) for k in outputs}
                for a1,a2 in GroupKFold(2,shuffle=True,random_state=719).split(t,groups=g[t]):
                    fit,cal = t[a1],t[a2]
                    assert not (np.intersect1d(g[fit],g[cal]).size or np.intersect1d(g[t],g[q]).size)
                    batch = np.r_[cal,q]
                    scores = evidence(votes,y,bill,fit,batch)
                    z = logit(base[i,batch].clip(.002,.998))[:,None]
                    for key,score in (("base",z),("independent",np.c_[z,scores[:,0]/50]),
                                      ("tree",np.c_[z,scores[:,1]]),("both",np.c_[z,scores[:,0]/50,scores[:,1]])):
                        fit_cal = LogisticRegression(C=1.,max_iter=500).fit(score[:len(cal)],y[cal])
                        summed[key] += .5*fit_cal.predict_proba(score[len(cal):])[:,1]
                for key,val in summed.items():
                    outputs[key][q] = val
        for key,val in outputs.items():
            assert np.isfinite(val).all()
            histories.setdefault(key,[]).append((val>=.5)==y)
    base_hit = (base>=.5)==y
    results = {k:summarize(np.stack(v),base_hit,masks) for k,v in histories.items()}
    write(REPORT/"vote_dependence.json",dict(codes=structures,heldout=results))
    print(structures,results)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
