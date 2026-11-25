"""Learn booking splits by expected net corrections, then check the leaf choices on separate orders."""
import json

import numpy as np
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from grouped_expert_choice_ceiling_n26 import best
from tabnet_attention_transfer_n26 import RESULT
from target_mix_expert_routing_n26 import ticker
from worst_booking_group_ensemble_n26 import order_splits

REPORT=RESULT.parent/'booking_policy_tree.json'


def split_features(fields,experts):
    # Keep the 18 existing family representatives; the action bank includes all 99 applicable experts.
    return np.column_stack([fields,experts[:,1:19]])


def boundaries(train,query,discovery):
    feature,cuts=[],[]
    for j,col in enumerate(train.T):
        u=np.unique(col[discovery])
        if len(u)<2:
            continue
        candidate=[(u[0]+u[-1])/2] if len(u)==2 else np.unique(np.quantile(col[discovery],[.25,.5,.75]))
        feature.extend([j]*len(candidate))
        cuts.extend(candidate)
    return train[:,feature]>cuts,query[:,feature]>cuts


def route(cases,draw,confirm,held,actions,query_actions,wrong):
    tx,qx=boundaries(*cases,draw)
    gain=actions*np.where(wrong,1.,-1.)[:,None]
    output=np.zeros((len(held),3),bool)
    def grow(rows,verify,q,depth):
        score=gain[rows].sum(0)
        choice=best(actions[rows],wrong[rows])
        if depth<3 and len(rows)>=48 and tx.shape[1]:
            tests=tx[rows]
            n=tests.sum(0)
            left=tests.T.astype(float)@gain[rows]
            right=score-left
            quality=left.max(1)+right.max(1)
            quality[(n<24)|(len(rows)-n<24)]=-np.inf
            idx=int(quality.argmax())
            if quality[idx]>score[choice]:
                for side in (False,True):
                    grow(rows[tx[rows,idx]==side],verify[tx[verify,idx]==side],q[qx[q,idx]==side],depth+1)
                return
        output[q,0]=query_actions[q,choice]
        if len(verify):
            observed=gain[verify].sum(0)
            if observed[choice]>0:
                output[q,1]=query_actions[q,choice]
            other=best(actions[verify],wrong[verify])
            output[q,2]=query_actions[q,other]
    grow(np.flatnonzero(draw),np.flatnonzero(confirm),np.flatnonzero(held),0)
    return output


def policies(b,t,q,at,aq):
    x=(split_features(b.train_fields,at),split_features(b.query_fields,aq))
    half=partition(b.groups,'policy-discovery:',2)
    both=route(x,t,np.zeros_like(t),q,at,aq,b.train_error)[:,0]
    views=np.stack([route(x,t&(half==i),t&(half!=i),q,at,aq,b.train_error) for i in range(2)])
    return dict(same_sample_tree=both,each_half_discovers=views[:,:,0].all(0),
                confirmed_choice_on_both=views[:,:,1].all(0),
                reestimated_leaves_agree=views[:,:,2].all(0),one_confirmed=views[:,:,1].any(0))


def run():
    parts={}
    for b,_,_,_,t,q,f,at,aq in order_splits():
        flags=policies(b,t,q,at,aq)
        parts.setdefault(b.tag,[]).append((b.ids,q,b.query_error,f,flags))
    report={}
    for tag,items in parts.items():
        ids,scope,wrong,fold=[np.concatenate([r[k] for r in items]) for k in range(4)]
        assert len(np.unique(ids))==len(ids)
        report[tag]={}
        for name in items[0][4]:
            act=np.concatenate([r[4][name] for r in items])
            assert not (act&~scope).any()
            s=ticker(scope,wrong,act)
            s['net_by_fold']=[int((act*np.where(wrong,1,-1)*(fold==k)).sum()) for k in range(8)]
            report[tag][name]=s
        print(tag,{name:(s['fixed'],s['harmed']) for name,s in report[tag].items()},flush=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
