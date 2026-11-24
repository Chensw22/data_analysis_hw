"""Exhaustively optimize two-expert decision tables, then test them on held-out orders."""
import itertools
import json

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_six_failure_archive_n24 import prediction_sets
from full_roster_solo_coverage_n24 import cached


def optimum(v,delta):
    a = v.astype(float)
    both = (a.T*delta)@a
    one = delta@a
    cells = np.stack([delta.sum()-one[:,None]-one[None,:]+both,one[None,:]-both,one[:,None]-both,both])
    gains = cells.clip(min=0).sum(0)
    i,j = np.unravel_index(np.argmax(gains),gains.shape)
    return int(i),int(j),cells[:,i,j]>0,int(gains[i,j])


def apply(v,policy):
    i,j,act,_ = policy
    return act[2*v[:,i].astype(int)+v[:,j]]


def check():
    v = np.random.default_rng(86).integers(2,size=(21,6)).astype(bool)
    d = np.random.default_rng(77).choice([-1,1],21)
    brute = max(int(d@np.array(t)[2*v[:,i].astype(int)+v[:,j]]) for i,j in
                itertools.combinations_with_replacement(range(6),2) for t in itertools.product((0,1),repeat=4))
    result = optimum(v,d)
    assert result[-1]==brute==d@apply(v,result)


def study(v,error,g,fold,region,labels):
    delta = np.where(error,1,-1)
    ids = np.arange(len(g))
    hindsight = np.zeros(len(g),bool)
    details = []
    for k in np.unique(region):
        here = ids[region==k]
        rule = optimum(v[here],delta[here])
        hindsight[here] = apply(v[here],rule)
        details.append(dict(region=int(k),rows=len(here),experts=[labels[rule[0]],labels[rule[1]]],
                            flip=rule[2].tolist(),net=rule[-1]))
    selected = np.zeros(len(g),bool)
    confirmed = selected.copy()
    for f in np.unique(fold):
        group_rows = ids[fold==f]
        parts = [group_rows[b] for _,b in GroupKFold(4,shuffle=True,random_state=795).split(group_rows,groups=g[group_rows])]
        for k in np.unique(region):
            blocks = [part[region[part]==k] for part in parts]
            for i,q in enumerate(blocks):
                train = np.concatenate([blocks[(i+1)%4],blocks[(i+2)%4],blocks[(i+3)%4]])
                if not len(q) or not len(train):
                    continue
                assert not np.isin(g[q],g[train]).any()
                rule = optimum(v[train],delta[train])
                selected[q] = apply(v[q],rule)
                t = np.r_[blocks[(i+1)%4],blocks[(i+2)%4]]
                val = blocks[(i+3)%4]
                if not len(t) or not len(val):
                    continue
                rule = optimum(v[t],delta[t])
                if delta[val]@apply(v[val],rule)>0:
                    confirmed[q] = apply(v[q],rule)
    return dict(hindsight=hindsight,held_out=selected,confirmed=confirmed),details


def run():
    check()
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    columns,panels = prediction_sets(raw,y,g,fold,rows,of)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,val in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[k],val)
        hard = z['remaining']
    extra = dict(dependency_tree=('independent','common','class','average'),
                 latent_profiles=('shared','class','random'),broad_error_transport_pair_surprise=('raw','marginal','dependence','all'))
    sources = columns+['six/'+str(j) for j in range(6)]+[k+'/'+n for k,ns in extra.items() for n in ns]
    labels = ['direct/'+n for n in sources]+['quarter/'+n for n in sources]
    result = {}
    for (tag,ix,f,base,p),(six,archive) in zip([
        ('full',np.arange(len(y)),fold,ref,full[:,names.index('mitra96')]),
        ('historical',rows,of,champ,old[:,0])],panels,strict=True):
        e = np.column_stack([archive,six,*[cached(k,tag,ix,f,g,y) for k in extra]])
        v = np.column_stack([(e>=.5)!=base[:,None],(((.25*e+.75*p[:,None])>=.5)!=base[:,None]) &
                             (((p>=.5)==base)[:,None])])
        assert v.shape==(len(ix),len(labels))
        conf = np.where(base,p,1-p)
        risk = 3*base.astype(int)+np.digitize(conf,[.65,.8])
        agree = ((six>=.5)==base[:,None]).all(1)
        result[tag] = {}
        for name,reg in [('one_region',np.zeros(len(ix),int)),('risk',risk),('risk_and_agreement',2*risk+agree)]:
            policies,detail = study(v,base!=y[ix],g[ix],f,reg,labels)
            metrics = {}
            for method,flip in policies.items():
                pred = base^flip
                score = stats(pred,base,y[ix],g[ix],f)
                score['hard_fixed'] = int((flip & (base!=y[ix]) & hard[ix]).sum())
                score['six_fixed'] = int((flip & (base!=y[ix]) & agree).sum())
                metrics[method] = score
            assert metrics['hindsight']['net']==sum(x['net'] for x in detail)
            result[tag][name]=dict(hindsight_choices=detail,methods=metrics)
            print(tag,name,[(k,v['net'],v['hard_fixed']) for k,v in metrics.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/pair_truth_tables.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
