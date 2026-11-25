"""Select expert mixtures for the least successful training booking group and score unseen orders."""
import json

import numpy as np
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from evaluate_nested_release_n26 import DECK, OUT, panel_predictions, protocols, sources, stages
from family_identity_transfer_n26 import panels_and_reference
from frozen_queue_shared_errors_n26 import paired
from joint_candidate_coverage_n26 import branch_decisions
from tabnet_attention_transfer_n26 import collect, RESULT
from target_mix_expert_routing_n26 import ROOT,batches,ticker

REPORT=RESULT.parent/'worst_booking_group_ensemble.json'


def first_stages(raw,panels,versions):
    _,_,_,rows,names,_,s,_,native,_=panels_and_reference()
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        old,prior,f=z['nine'],z['v131'],z['fold']
    np.testing.assert_array_equal(panels['historical']['ids'],rows)
    branches=branch_decisions(rows,f,s,names,native,prior,old,old,raw.CryoSleep.eq(False).to_numpy(),'historical',891)
    history=np.column_stack([branches[key][0] for key in ('nine','fields','awake','fields_awake','band','all_three')])
    np.testing.assert_array_equal(history[panels['historical']['known'],-1],panels['historical']['anchor'][panels['historical']['known']])
    return {'historical':history,'full':versions[:,:7]}


def select(gains):
    # Column zero retains the current release. A feasible optimum therefore has nonnegative worst-group gain.
    n=gains.shape[1]
    res=linprog(np.r_[np.zeros(n),-1.],A_ub=np.column_stack([-gains,np.ones(len(gains))]),
                b_ub=np.zeros(len(gains)),A_eq=np.r_[np.ones(n),0][None,:],b_eq=[1.],
                bounds=[(0,1)]*n+[(None,None)],method='highs')
    if not res.success:
        raise RuntimeError(res.message)
    w=res.x[:-1]
    assert abs(w.sum()-1)<1.e-6 and abs(np.min(gains@w)-res.x[-1])<1.e-6
    if res.x[-1]<1.e-7:
        w=np.eye(1,n)[0]
    return w


def portfolios(xt,xq,ct,cq,train_scope,wrong,groups):
    outputs={k:np.zeros(len(xq),bool) for k in ('pooled','worst_quarter','each_half_selects')}
    labels=partition(groups,'robust-expert-group:',4)
    used=[]
    for cell in np.unique(cq):
        t=train_scope&(ct==cell)
        q=cq==cell
        parts=[t&(labels==i) for i in range(4)]
        if min(np.count_nonzero(m) for m in parts)==0:
            continue
        signed=xt*np.where(wrong,1.,-1.)[:,None]
        gains=np.stack([signed[m].mean(0) for m in parts])
        average=signed[t].mean(0)
        best=int(average.argmax())
        outputs['pooled'][q]=xq[q,best]
        robust=select(gains)
        outputs['worst_quarter'][q]=xq[q]@robust>.5000001
        pairs=[select(gains[:2]),select(gains[2:])]
        outputs['each_half_selects'][q]=np.all(np.column_stack([xq[q]@w>.5000001 for w in pairs]),axis=1)
        used.append(int((robust[1:]>1.e-6).sum()))
    return outputs,used


def run():
    raw,_,_,_,_,_=paired()
    prior,y,g,fold,versions=collect()
    panels=sources()[-1]
    first=first_stages(raw,panels,versions)
    pred={n:panel_predictions(v,n) for n,v in panels.items()}
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
    iters={tag:iter(rows) for tag,rows in prior.items()}
    result={}
    for details,b in zip(protocols(panels,g,True),batches(True),strict=True):
        key,src,_,it,_=details
        origin='full' if b.tag=='full' else 'historical'
        q,same,now,previous=next(iters[b.tag])
        np.testing.assert_array_equal(b.ids,q)
        np.testing.assert_array_equal(now,b.query_cell%2)
        np.testing.assert_array_equal(src['ids'][it],b.train_ids)
        assert not np.isin(b.groups,g[q]).any()
        with np.load(OUT/f'{key}.npz') as z:
            np.testing.assert_array_equal(z['outer'],q)
            np.testing.assert_array_equal(z['ids'],b.train_ids)
            anchor,review,fare,old=stages(src,pred[origin],it,enabled[b.train_ids],z['risk'])
        np.testing.assert_array_equal(anchor,first[origin][it,-1])
        np.testing.assert_array_equal(old,b.train_cell%2)
        train_scope=(first[origin][it]==old[:,None]).all(1)&(old==review)&(old==fare)
        # Historical filtered-awake changes are contained in the awake branch.
        xt,xq=b.train_actions[:,:100],b.query_actions[:,:100]
        assert not xt[:,0].any() and not xq[:,0].any()
        policies,used=portfolios(xt,xq,b.train_cell%4,b.query_cell%4,train_scope,b.train_error,b.groups)
        scope=same&~previous['actual_and_scores']
        result.setdefault(b.tag,[]).append((q,scope,b.query_error,policies,used))
    report={}
    for tag,rows in result.items():
        ids,scope,wrong=(np.concatenate([row[k] for row in rows]) for k in range(3))
        assert len(ids)==len(np.unique(ids))
        metrics={}
        for name in rows[0][3]:
            act=np.concatenate([row[3][name] for row in rows])
            r=ticker(scope,wrong,act)
            r['net_by_fold']=[int((np.where(wrong,1,-1)*scope*act*(fold[ids]==k)).sum()) for k in range(8)]
            metrics[name]=r
        counts=np.concatenate([row[4] for row in rows])
        report[tag]=dict(methods=metrics,cell_fits=len(counts),nonempty_robust_fits=int((counts>0).sum()),
                         multi_expert_fits=int((counts>1).sum()))
        print(tag,report[tag],flush=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
