"""Test NODE's incremental evidence versus an ordinary tree and TabNet on new bookings."""
import json
import numpy as np
from scipy.special import logit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import fit
from attentive_tabnet_n26 import OUT as TABNET,MODELS
from booking_mix_confidence_transfer_n26 import partition
from dense_oblivious_node_n26 import OUT
from held_expert_portfolios_n26 import sources
from full_roster_booking_validation_n23 import data
from tabnet_attention_transfer_n26 import RESULT
from target_mix_expert_routing_n26 import ticker
from worst_booking_group_ensemble_n26 import order_splits


def scores():
    raw,y,g,f=data()
    panels=sources()[-1]
    output={}
    for tag,v in panels.items():
        folder=OUT if tag=='full' else OUT/'history'
        with np.load(folder/'combined.npz') as z:
            np.testing.assert_array_equal(z['ids'],v['ids'])
            np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[v['ids']])
            np.testing.assert_array_equal(z['fold'],v['fold'])
            bank=np.full((len(y),3),np.nan)
            bank[v['ids'],:2]=np.column_stack([z['NODE'],z['ordinary_tree']])
        folder=TABNET if tag=='full' else TABNET/'history'
        for k in np.unique(v['fold']):
            q=v['ids'][v['fold']==k]
            with np.load(folder/f'fold{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                np.testing.assert_array_equal(z['train_y'],y[z['t']])
                assert not np.isin(g[z['t']],g[q]).any()
                bank[q,2]=np.column_stack([z[n] for n in MODELS]).mean(1)
        assert np.isfinite(bank[v['ids']]).all()
        output[tag]=bank
    return output


def design(x,p,parent):
    votes=logit(p.clip(.002,.998))*np.where(parent,-1,1)[:,None]
    controls=np.column_stack([x,votes[:,1:],votes[:,1]*votes[:,2]])
    return controls,votes[:,0]


def rules(train,query,target,wt,node_t,node_q):
    scale=StandardScaler().fit(train)
    xt,xq=scale.transform(train).clip(-5,5),scale.transform(query).clip(-5,5)
    relation=Ridge(alpha=12.).fit(xt,node_t)
    delta_t=node_t-relation.predict(xt)
    delta_q=node_q-relation.predict(xq)
    base=fit(xt,target,xq,wt)>.5
    learnt=fit(np.column_stack([xt,delta_t,delta_t*node_t]),
               target,np.column_stack([xq,delta_q,delta_q*node_q]),wt)>.5
    return base,learnt


def run():
    bank=scores()
    results={}
    for b,origin,_,_,source,scope,fold,_,_ in order_splits():
        t=b.train_ids
        q=b.ids
        old,now=b.train_cell%2!=0,b.query_cell%2!=0
        xt,nt=design(b.train_fields,bank[origin][t],old)
        xq,nq=design(b.query_fields,bank['full'][q],now)
        challenge=(bank['full'][q,0]>=.5)!=now
        ct=(bank[origin][t,0]>=.5)!=old
        weights=np.where(ct,1.,.2)
        base,new=rules(xt[source],xq,b.train_error[source],weights[source],nt[source],nq)
        groups=partition(b.groups,'node-calibration-confirmation:',2)
        halves=[]
        for k in range(2):
            subset=source&(groups==k)
            _,h=rules(xt[subset],xq,b.train_error[subset],weights[subset],nt[subset],nq)
            halves.append(h)
        edits=dict(raw=challenge,base=challenge&base,augmented=challenge&new,
                   newly_supported=challenge&new&~base,confirmed=challenge&new&halves[0]&halves[1])
        results.setdefault(b.tag,[]).append((scope,b.query_error,fold,edits))
    report={}
    for tag,items in results.items():
        scope,wrong,fold=(np.concatenate([v[k] for v in items]) for k in range(3))
        report[tag]={}
        for name in items[0][3]:
            action=np.concatenate([v[3][name] for v in items])
            r=ticker(scope,wrong,action)
            r['net_by_fold']=[int((np.where(wrong,1,-1)*action*scope*(fold==k)).sum()) for k in range(8)]
            report[tag][name]=r
        print(tag,{n:(r['fixed'],r['harmed']) for n,r in report[tag].items()},flush=True)
    (RESULT.parent/'node_countervote_transfer.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
