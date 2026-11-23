"""Audit probability normalization, out-of-booking fitting and matched rare-error separation."""
from itertools import combinations, product

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_neighbor_decision_n23 import opinions
from broad_class_transport_n23 import encode
from broad_conditional_field_values_n23 import run_seed
from broad_dependence_anchors_n23 import likelihoods, predictions
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_tree_field_dividends_n23 import hard_cohorts


def normalization():
    widths = [2,3,2,2,3]
    grid = np.array(list(product(*[range(w) for w in widths])))
    rng = np.random.default_rng(910)
    train = grid[rng.choice(len(grid),89)]
    values = likelihoods(train,grid,widths,field_conditionals=True)
    gaps = {key:float(abs(np.exp(values[key]).sum(0)-1).max()) for key in ('single','double','independent')}
    pairs = list(combinations(range(len(widths)),2))
    conditional = np.exp(values['conditional_fields'])
    restricted_error, full_difference = 0., 0.
    for k in range(len(widths)):
        subset = [j for j,pair in enumerate(pairs) if k not in pair]
        complete = np.exp(values['double'][:,subset]).mean(1).reshape(widths)
        full = np.exp(values['double']).mean(1).reshape(widths)
        exact = complete/complete.sum(k,keepdims=True)
        full = full/full.sum(k,keepdims=True)
        restricted_error = max(restricted_error,abs(exact.ravel()-conditional[:,k]).max())
        full_difference = max(full_difference,abs(full.ravel()-conditional[:,k]).max())
    assert max(*gaps.values(), restricted_error) < 1e-12
    return dict(joint_errors=gaps,max_conditional_error=float(restricted_error),
                max_difference_from_all_anchor_pairs=float(full_difference),states=len(grid))


def label_mutation(x,y,g,fold,base,ref,expected):
    q = np.flatnonzero(fold==np.unique(fold)[0])
    t = np.flatnonzero((fold!=fold[q[0]])&~np.isin(g,g[q]))
    a,b = next(GroupKFold(3,shuffle=True,random_state=589).split(q,groups=g[q]))
    test = q[b]
    assert not np.intersect1d(g[test],np.r_[g[t],g[q[a]]]).size
    flipped=y.copy()
    flipped[test] = ~flipped[test]
    original=predictions(x,y,t,test)
    modified=predictions(x,flipped,t,test)
    assert all(np.array_equal(original[k],modified[k]) for k in modified)
    after,_ = run_seed(x,flipped,g,fold,base,ref)
    diff = {k:float(abs(p[test]-expected[k][test]).max()) for k,p in after.items()}
    assert max(diff.values())<=1e-12
    return dict(perturbed_passengers=len(test),perturbed_bookings=int(np.unique(g[test]).size),
                prediction_max_diff=diff,joint_identical=True)


def match(case, control, vote, g, confidence):
    i,j=np.flatnonzero(case),np.flatnonzero(control)
    diff=np.abs(confidence[:,i,None]-confidence[:,None,j])
    valid=(g[i,None]!=g[j]) & (vote[i,None]==vote[j]) & (diff.max(0)<=.025)
    cost=np.where(valid,diff.mean(0),2.e4)
    # Dummy assignments make maximization of the number of caliper matches lexicographically first.
    r,c=linear_sum_assignment(np.c_[cost,np.full((len(i),len(i)),100.)])
    use=c<len(j)
    case,control=i[r[use]],j[c[use]]
    assert len(set(control))==len(control) and valid[r[use],c[use]].all()
    error=abs(confidence[:,case]-confidence[:,control])
    return case,control,dict(n=len(case),max_gap=float(error.max()),mean_gap=float(error.mean()))


def intervals(case,control,g):
    _, gi=np.unique(g,return_inverse=True)
    graph=coo_matrix((np.ones(len(case)),(gi[case],gi[control])),shape=(gi.max()+1,gi.max()+1)).tocsr()
    _, cc=connected_components(graph,directed=False)
    assert (cc[gi[case]]==cc[gi[control]]).all()
    _,idx=np.unique(cc[gi[case]],return_inverse=True)
    rng=np.random.default_rng(712)
    draws=rng.integers(0,idx.max()+1,size=(4000,idx.max()+1))
    den=np.bincount(idx)[draws].sum(1)
    def interval(value):
        sums=np.bincount(idx,weights=value)
        return np.quantile(sums[draws].sum(1)/den,[.025,.975]).tolist()
    return interval


def main():
    row,y,frame,g,_=load()
    _,_,hard,correct=hard_cohorts(row,y)
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(row,z['row'])
        vote=z['votes'][0,:,0]
    np.testing.assert_array_equal(vote[hard],~y[hard])
    np.testing.assert_array_equal(vote[correct],y[correct])
    info,old,conf,signals={},[],[],{}
    training_support={}
    x=encode(frame)
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as source:
            fold=source['fold']
            np.testing.assert_array_equal(row,source['row'])
        unsupported=[]
        for f in np.unique(fold):
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[fold==f]))
            assert not np.intersect1d(g[t],g[fold==f]).size
            unseen=sum(len(np.setdiff1d(x[:,j],x[t,j])) for j in range(x.shape[1]))
            unsupported.append(int(unseen))
        training_support[seed]=unsupported
        if seed=='v131': continue
        fold,base,ref,bill,near=opinions(seed,row,y,frame.BookingSize.to_numpy()>1)
        confidence=np.where(vote,base,1-base)
        conf.append(confidence)
        def read(folder,keys):
            with np.load(ART/f'nov23_{folder}/{seed}.npz') as z:
                for k,v in [('row',row),('y',y),('fold',fold),('base',base),('ref',ref)]:
                    np.testing.assert_array_equal(z[k],v)
                return {k:z[k] for k in keys}
        fields=read('abstaining_fields',['personal_first','bill_first','averaged_order'])
        old.append(np.c_[bill,near,*fields.values()]>=.5)
        old[-1]=(old[-1]!=ref[:,None]) & (ref==(base>=.5))[:,None]
        arms=read('dependence_anchors',['independent','single','double','shuffled_double'])
        fitted=read('conditional_field_values',['anchor','marginal','conditional','dependence_only','both'])
        if seed==str(SEEDS[0]):
            info=label_mutation(x,y,g,fold,base,ref,fitted)
        for k,p in {'incumbent':base,**arms,**fitted}.items():
            signals.setdefault(k,[]).append(np.where(vote,1-p,p))
    old=np.stack(old).all(0).any(1)
    assert (old&hard).sum()==41
    a,b,meta=match(hard,correct,vote,g,np.array(conf))
    ci=intervals(a,b,g)
    summary,comparisons,concs={}, {}, {}
    for name,runs in signals.items():
        scores=np.stack(runs)
        stable=(scores>.5).all(0)
        # Fitting never uses this cohort. Its historical votes are unanimous, so the risk direction is common.
        delta=scores[:,a]-scores[:,b]
        conc=(delta>0).astype(float)+.5*(delta==0)
        concs[name]=conc.mean(0)
        summary[name]=dict(matched_concordance=float(conc.mean()),concordance_by_split=conc.mean(1).tolist(),
                           concordance_ci=ci(conc.mean(0)), hard_stable=int((stable&hard).sum()),
                           hard_novel=int((stable&hard&~old).sum()), similar_correct_reversed=int((stable&correct).sum()),
                           new_similar_correct_reversed=int((stable&correct&~old).sum()))
    for source,base in [('double','independent'),('double','single'),('double','shuffled_double'),
                        ('conditional','marginal'),('both','marginal'),('conditional','incumbent')]:
        difference=concs[source]-concs[base]
        comparisons[source+'-minus-'+base]=dict(difference=float(difference.mean()),paired_ci=ci(difference))
    union=np.any([np.all(np.stack(signals[k])>.5,0) for k in ('independent','single','double')],axis=0)
    coverage=dict(stable=int((union&hard).sum()),new=int((union&hard&~old).sum()),
                  stable_harms=int((union&correct).sum()),new_harms=int((union&correct&~old).sum()))
    write(ROOT/'reports/nov23_dependence_score_audit/matched.json',dict(normalization=normalization(),
          unobserved_field_levels_per_training_fold=training_support,three_real_models_union=coverage,
          label_invariance=info,matched=meta,population=dict(hard=int(hard.sum()),similar_correct=int(correct.sum()),prior=41),
          scores=summary,paired_comparison=comparisons))
    print(meta,comparisons,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
