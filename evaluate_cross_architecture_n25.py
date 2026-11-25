"""Compare direct corrections, mutually supported corrections, and conditional pooling across two transformers."""
import argparse
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold

from awake_context_ranks_n25 import OUT as IC_AWAKE
from cross_architecture_state_n25 import OUT
from disjoint_tree_fields_n25 import ART,DEST,stats
from evaluate_awake_demonstrations_n25 import deployed
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_teacher_n25 import OUT as REF
from passenger_state_contexts_n25 import OUT as IC_GLOBAL


def offset_fit(features,base,y,t,q):
    # Ridge prior toward retaining the native log odds.
    x=features[t]
    def objective(w):
        logits=base[t]+x@w
        loss=np.logaddexp(0,logits).sum()-y[t]@logits+5*w@w
        return loss,x.T@(expit(logits)-y[t])+10*w
    fit=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
    assert fit.success
    return expit(base[q]+features[q]@fit.x)


def split_pool(prob,base,awake,g,y,k):
    anchors=logit(base.clip(.01,.99))
    d=logit(prob.clip(.01,.99))-anchors[:,None]
    fields=np.column_stack([np.ones(len(base)),anchors,awake])
    rng=np.random.default_rng(561+k)
    shuffled=d.copy()
    bands=np.digitize(base,np.arange(.1,1,.1))+10*awake
    for b in np.unique(bands):
        ix=np.flatnonzero(bands==b)
        shuffled[ix]=d[rng.permutation(ix)]
    designs={'native_calibration':fields,'conditional_pool':np.column_stack([fields,d]),
             'shuffled_pool':np.column_stack([fields,shuffled])}
    result={name:np.empty(len(y)) for name in designs}
    for t,q in GroupKFold(4,shuffle=True,random_state=369+k).split(base,groups=g):
        assert not np.isin(g[t],g[q]).any()
        for name,x in designs.items():
            result[name][q]=offset_fit(x,anchors,y,t,q)
    return result


def read(path,ix,key):
    with np.load(path) as z:
        np.testing.assert_array_equal(z['q'],ix)
        return z[key]


def run(folds):
    raw,y,g,fold=data()
    awake=raw.CryoSleep.eq(False).to_numpy()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,fold,rows,hf)
    votes=(np.column_stack(panels[0])>=.5).sum(1)
    majority=votes>34
    common=(votes==0)|(votes==68)
    near=(votes<=3)|(votes>=65)
    masks=[]
    effects={}
    for k in folds:
        ix=np.flatnonzero(fold==k)
        a=awake[ix]
        with np.load(REF/f'release{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],ix)
            native,parent,nine,low,high=(z[key] for key in ('native','v131','nine','nine_on_low_branch','nine_on_high_branch'))
        base=native>=.5
        known=parent>=0
        masks.append((ix,known))
        dpt=read(OUT/f'{k}_global.npz',ix,'prob')
        ic=read(IC_GLOBAL/f'{k}_global.npz',ix,'p')
        dpt_state=dpt.copy()
        ic_state=ic.copy()
        dpt_state[a]=read(OUT/f'{k}_awake.npz',ix[a],'prob')
        ic_state[a]=read(IC_AWAKE/f'{k}_personal.npz',ix[a],'p')
        queued=np.where(a,ic_state>=.5,base)
        pool=split_pool(np.column_stack([dpt,ic,dpt_state,ic_state]),native,a,g[ix],y[ix],k)
        methods={'dpt_global':dpt,'dpt_awake':np.where(a,dpt_state,native),'dpt_routed':dpt_state,
            'routed_mean':(ic_state+dpt_state)/2,'routed_quarter':.75*native+.125*(ic_state+dpt_state),
            'both_awake':np.where(a&((ic_state>=.5)==(dpt_state>=.5)),ic_state,native),
            'either_awake':np.where(a&((ic_state>=.5)!=base),ic_state,np.where(a,dpt_state,native)),**pool}
        for name,probs in methods.items():
            d=probs>=.5
            branches=[]
            for sign,ref in ((False,low),(True,high)):
                prior=np.where(known,parent,sign)
                hit=deployed(d,prior,ref,base)==y[ix]
                branches.extend([hit.astype(int)-(ref==y[ix]),hit.astype(int)-(deployed(queued,prior,ref,base)==y[ix])])
            direct=np.where(a,d,majority[ix]) if name in ('dpt_awake','both_awake','either_awake') else d
            delta=(direct==y[ix]).astype(int)-(majority[ix]==y[ix])
            effects.setdefault(name,[]).append((*branches,delta))
    ix=np.concatenate([i for i,_ in masks])
    known=np.concatenate([v for _,v in masks])
    results={}
    for name,blocks in effects.items():
        l,ql,h,qh,delta=(np.concatenate([v[j] for v in blocks]) for j in range(5))
        np.testing.assert_array_equal(l[known],h[known])
        np.testing.assert_array_equal(ql[known],qh[known])
        def bounds(a,b):return [int(np.minimum(a,b).sum()),int(np.maximum(a,b).sum())]
        def summary(a,mask):return stats(a,mask,fold[ix],g[ix])
        results[name]=dict(nine_bounds=bounds(l,h),queued_bounds=bounds(ql,qh),native_release=summary(l,known),
            queued=summary(ql,known),new_orders=summary(l,known&~np.isin(g[ix],g[rows])),
            consensus=summary(delta,common[ix]),near=summary(delta,near[ix]))
        r=results[name]
        print(name,r['nine_bounds'],r['queued_bounds'],r['consensus']['fixed'],r['consensus']['harmed'],flush=True)
    path=DEST.with_name('cross_architecture_state.json') if len(folds)==8 else OUT/'interim.json'
    path.write_text(json.dumps(dict(folds=folds,rows=len(ix),known=int(known.sum()),results=results),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',nargs='+',type=int,default=list(range(8)),choices=range(8))
    run(parser.parse_args().folds)
