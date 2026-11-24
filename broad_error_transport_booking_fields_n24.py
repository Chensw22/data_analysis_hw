"""Marginalize bijective within-booking assignments of sleep or registration fields."""
import itertools
import json
from functools import lru_cache

import numpy as np
from scipy.special import expit, logit, softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_error_transport_pair_surprise_n24 import fields
from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_permutation_selection_n24 import cat
from full_roster_solo_coverage_n24 import cached

FIELD_GROUPS = {'sleep':['CryoSleep'], 'registration':['Age','VIP','HomePlanet','Destination']}
MODES = ['measured', *[f'{block}_{method}' for block in FIELD_GROUPS for method in ('prior','conditional')]]
TAG = 'broad_error_transport_booking_fields'


@lru_cache(maxsize=8)
def permutations(size):
    return np.asarray(list(itertools.permutations(range(size))), dtype=int)


def assignment_margins(log_weight):
    n = len(log_weight)
    perms = permutations(n)
    probability = softmax(log_weight[np.arange(n),perms].sum(axis=1))
    marginal = np.stack([np.bincount(perms[:,i],weights=probability,minlength=n) for i in range(n)])
    np.testing.assert_allclose(marginal.sum(0),1,atol=1e-10)
    np.testing.assert_allclose(marginal.sum(1),1,atol=1e-10)
    return marginal


def assignments(raw,g):
    blocks,labels,group = fields(raw)
    np.testing.assert_array_equal(group,g)
    keys = list(blocks)
    start = np.cumsum([0,*[blocks[k].shape[1] for k in keys]])
    base = np.column_stack(list(blocks.values()))
    members = raw.groupby(g).indices
    matrices,owners,weights = {},{},{}
    diagnostics = {}
    for block,targets in FIELD_GROUPS.items():
        # Never use any member of the hypothesized block to predict another member of that block.
        visible = np.column_stack([v for k,v in blocks.items() if k not in targets])
        scores = {k:np.empty((len(g),int(labels[k].max()+1))) for k in targets}
        for t,q in GroupKFold(4,shuffle=True,random_state=177).split(base,groups=g):
            assert not np.isin(g[q],g[t]).any()
            for key in targets:
                k = scores[key].shape[1]
                clf = ExtraTreesClassifier(n_estimators=50,min_samples_leaf=18,max_features=.8,
                                           random_state=26,n_jobs=2)
                clf.fit(visible[t],labels[key][t])
                prior = np.bincount(labels[key][t],minlength=k)+1.
                p = np.tile(.03*prior/prior.sum(),(len(q),1))
                p[:,clf.classes_] += .97*clf.predict_proba(visible[q])
                scores[key][q] = np.log(p)
        own,other,prior_w,cond_w = [],[],[],[]
        n_support = 0
        avg_identity = []
        col = np.concatenate([np.arange(start[keys.index(key)],start[keys.index(key)+1]) for key in targets])
        for ids in members.values():
            n = len(ids)
            src = np.repeat(ids,n)
            dest = np.tile(ids,n)
            bias = np.eye(n)*np.log(4)
            if n==1:
                prior = post = np.ones((1,1))
            else:
                ll = sum(scores[k][ids][:,labels[k][ids]] for k in targets)
                prior,post = assignment_margins(bias),assignment_margins(ll+bias)
                if np.any(base[ids][:,col] != base[ids[0],col]):
                    n_support += n
                    avg_identity.extend(post.diagonal())
            own.append(src)
            other.append(dest)
            prior_w.append(prior.ravel())
            cond_w.append(post.ravel())
        src,dest = np.concatenate(own),np.concatenate(other)
        counter = base[src].copy()
        counter[:,col] = base[dest][:,col]
        owners[block],matrices[block] = src,counter
        weights[block] = [np.concatenate(prior_w),np.concatenate(cond_w)]
        diagnostics[block] = dict(people_in_bookings_with_any_field_variation=n_support,
                                  mean_identity_probability_among_these=float(np.mean(avg_identity)))
    return base,matrices,owners,weights,diagnostics


def run():
    np.testing.assert_allclose(assignment_margins(np.log([[4.,1.],[1.,4.]])),np.array([[16,1],[1,16]])/17)
    raw,y,g,fold = data()
    base,mat,owner,weight,diagnostics = assignments(raw,g)
    solo = ~raw.PassengerId.str.split('_').str[0].duplicated(keep=False).to_numpy()
    def predict(_,target,groups,t,q):
        assert not np.isin(groups[t],groups[q]).any()
        model = cat(815).fit(base[t],target[t])
        measured = model.predict_proba(base[q])[:,1]
        columns = [measured]
        for block in FIELD_GROUPS:
            relevant = np.isin(owner[block],q)
            scores = model.predict_proba(mat[block][relevant])[:,1]
            for w in weight[block]:
                mean = np.bincount(owner[block][relevant],weights=w[relevant]*scores,minlength=len(y))[q]
                np.testing.assert_allclose(measured[solo[q]],mean[solo[q]],atol=1e-15)
                columns.append(mean)
        return np.column_stack(columns)
    names,h,b = banks(raw,y,fold)
    reference = latest_reference(y,g,fold,names,h,b)
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    idx = np.arange(len(y))
    full = protocol(base,y,g,idx,fold,'full',predict,MODES,TAG)
    historical = protocol(base,y,g,rows,of,'historical',predict,MODES,TAG)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for key,v in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[key],v)
        hard = z['remaining']
    six = np.column_stack([cached(k,'full',idx,fold,g,y) for k in ('response_channels','broad_error_transport_projection')])
    missed = (reference!=y) & ((six>=.5)!=y[:,None]).all(1)
    assert missed.sum()==848 and hard.sum()==147
    new = ~np.isin(g,og)
    anchor = b[:,names.index('mitra96')]
    report = dict(diagnostics=diagnostics, eligible_multi=int((~solo).sum()), remaining_multi_errors=int((~solo & (reference!=y)).sum()),
                  hard_multi=int((~solo&hard).sum()), six_model_multi=int((~solo&missed).sum()),models={})
    def rules(p,ref,outputs,j):
        shift = logit(outputs[:,j].clip(.001,.999))-logit(outputs[:,0].clip(.001,.999))
        keep = (p>=.5)!=ref
        return {'direct':outputs[:,j]>=.5,
                'change_only':np.where(keep,ref,logit(p.clip(.001,.999))+shift>=0),
                'quarter_change':np.where(keep,ref,logit(p.clip(.001,.999))+.25*shift>=0)}
    for j,key in enumerate(MODES):
        for mode,pred in rules(anchor,reference,full,j).items():
            hist = rules(old[:,0],champ,historical,j)[mode]
            if mode != 'direct':
                np.testing.assert_array_equal(pred[solo],reference[solo])
            report['models'][key+'_'+mode] = dict(full=stats(pred,reference,y,g,fold),
                historical=stats(hist,champ,oy,og,of),new_orders=stats(pred[new],reference[new],y[new],g[new],fold[new]),
                versus_same_measured_model=stats(pred,full[:,0]>=.5,y,g,fold) if mode=='direct' else None,
                shared_hard_fixed=int((hard & (reference!=y) & (pred==y)).sum()),
                six_model_fixed=int((missed & (pred==y)).sum()))
    path = ROOT/'reports/nov23_matched_learning_curve'/f'{TAG}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(diagnostics)
    for k,r in report['models'].items():
        print(k, r['full']['corrected'],r['full']['harmed'],r['historical']['net'],r['new_orders']['net'],r['six_model_fixed'],r['shared_hard_fixed'])


if __name__=='__main__':
    with threadpool_limits(2):
        run()
