"""Estimate the absolute chance of a wrong vote at matched continuous confidence and deck/service state."""
import json
import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import data
from held_expert_portfolios_n26 import outcome_scores, panel_predictions, sources
from tabnet_attention_transfer_n26 import RESULT
from target_mix_expert_routing_n26 import ticker
from worst_booking_group_ensemble_n26 import order_splits


def local_rate(train_score,query_score,error,train_cell,query_cell):
    # Use a common bandwidth on log odds and shrink the conditional rate toward the
    # broad confidence-matched estimate when a deck/service cell has few observations.
    wt=np.exp(-.5*((query_score[:,None]-train_score[None,:])/.5)**2)
    prior=(wt@error+1)/(wt.sum(1)+2)
    conditional=wt*(query_cell[:,None]==train_cell[None,:])
    mass=conditional.sum(1)
    return prior,(conditional@error+10*prior)/(mass+10)


def shuffle_cells(cell,score,seed):
    band=np.floor(score*2)
    rng=np.random.default_rng(seed)
    new=cell.copy()
    for v in np.unique(band):
        idx=np.flatnonzero(band==v)
        new[idx]=rng.permutation(cell[idx])
    return new


def run():
    raw,_,g,_=data()
    solo=raw.groupby(g).PassengerId.transform('size').eq(1).to_numpy()
    cell=raw.Cabin.str.startswith('G/',na=False).to_numpy().astype(int)*2+~raw.FoodCourt.gt(0).to_numpy()
    views=sources()[-1]
    conf={}
    for tag,v in views.items():
        panel=outcome_scores(v,panel_predictions(v,tag))
        conf[tag]=logit(panel[:,:18].mean(1).clip(.002,.998))
    runs={}
    for step,(b,origin,it,iq,ts,qs,fold,xt,xq) in enumerate(order_splits()):
        t,q=b.train_ids,b.ids
        np.testing.assert_array_equal(t,views[origin]['ids'][it])
        np.testing.assert_array_equal(q,views['full']['ids'][iq])
        assert not np.isin(g[t],g[q]).any()
        pt,pq=b.train_cell%2,b.query_cell%2
        target=qs&solo[q]&xq[:,1:].any(1)
        preds=np.zeros((len(q),3))
        for side in (0,1):
            at=ts&solo[t]&(pt==side)&xt[:,1:].any(1)
            aq=target&(pq==side)
            if not aq.any():
                continue
            s,z=conf[origin][it[at]],conf['full'][iq[aq]]
            truth=b.train_error[at]
            vals,other=cell[t[at]],cell[q[aq]]
            before,after=local_rate(s,z,truth,vals,other)
            _,placebo=local_rate(s,z,truth,shuffle_cells(vals,s,5420+step),other)
            preds[aq]=np.column_stack([before,after,placebo])
        runs.setdefault(b.tag,[]).append((q,target,b.query_error,pq,fold,preds))
    report={}
    for tag,items in runs.items():
        ids,scope,wrong,ref,f,p=(np.concatenate([a[j] for a in items]) for j in range(6))
        assert len(np.unique(ids))==len(ids)
        a,b,c=(p[:,i]>.5 for i in range(3))
        rules=dict(confidence=a,real=b,sham=c,added_by_cell=b&~a,added_beyond_both=b&~a&~c)
        report[tag]={}
        for key,mask in [('all_solo',scope),('positive',scope&(ref==1)),('positive_g_no_food',scope&(ref==1)&(cell[ids]==3))]:
            out={}
            for name,act in rules.items():
                item=ticker(mask,wrong,act)
                item['net_by_fold']=[int((mask*act*np.where(wrong,1,-1)*(f==k)).sum()) for k in range(8)]
                out[name]=item
            report[tag][key]=out
        print(tag,{k:(v['fixed'],v['harmed']) for k,v in report[tag]['all_solo'].items()},flush=True)
    (RESULT.parent/'local_risk_by_booking_evidence.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):run()
