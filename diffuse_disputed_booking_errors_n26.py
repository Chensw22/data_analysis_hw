"""Propagate error evidence between nearby opinions without querying any held booking outcome."""
import json
import numpy as np
from scipy.sparse import csr_matrix,eye
from scipy.sparse.linalg import spsolve
from scipy.special import logit
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import fit
from held_expert_portfolios_n26 import sources,panel_predictions,outcome_scores
from tabnet_attention_transfer_n26 import RESULT
from target_mix_expert_routing_n26 import ticker
from worst_booking_group_ensemble_n26 import order_splits

REPORT=RESULT.parent/'diffuse_disputed_booking_errors.json'


def standardized(a,b):
    scale=StandardScaler().fit(a)
    return np.vstack([scale.transform(a),scale.transform(b)]).clip(-5,5)/np.sqrt(a.shape[1])


def neighbors(x,t_count,groups):
    dist=euclidean_distances(x[t_count:],x,squared=True)
    dist[groups[t_count:,None]==groups[None,:]]=np.inf
    def kernel(block):
        k=min(16,block.shape[1])
        near=np.argpartition(block,k-1,axis=1)[:,:k]
        d=np.take_along_axis(block,near,axis=1)
        assert np.isfinite(d).all()
        w=np.exp(-d/d.max(1,keepdims=True).clip(1.e-6))
        w/=w.sum(1,keepdims=True)
        return csr_matrix((w.ravel(),(np.repeat(np.arange(len(d)),k),near.ravel())),shape=block.shape)
    return kernel(dist[:,:t_count]),kernel(dist)


def spread(one,all_edges,target,prior,residual):
    n=len(target)
    support,unlabeled=all_edges[:,:n],all_edges[:,n:]
    strength=.85
    system=eye(len(prior),format='csc')-strength*unlabeled.tocsc()
    result=spsolve(system,(1-strength)*prior+strength*(support@target))
    correction=spsolve(system,strength*(support@residual))
    assert np.max(np.abs(result-((1-strength)*prior+strength*(support@target+unlabeled@result))))<1.e-7
    return dict(one_hop=(1-strength)*prior+strength*(one@target),multi_hop=result,
                residual=prior+correction)


def run():
    _,_,groups,panels=sources()
    model={tag:outcome_scores(v,panel_predictions(v,tag)) for tag,v in panels.items()}
    reports={}
    for b,origin,it,iq,ts,qs,f,at,aq in order_splits():
        t=ts&at[:,1:].any(1)
        q=qs&aq[:,1:].any(1)
        names=['base','one_hop','multi_hop','residual','shuffled_residual','only_real']
        output={n:np.zeros(len(q),bool) for n in names}
        if q.any():
            truth=b.train_error[t].astype(float)
            legs=b.groups[t];query_groups=groups[b.ids[q]]
            assert not np.isin(legs,query_groups).any()
            xt,xq=b.train_fields[t],b.query_fields[q]
            prior=fit(xt,truth,xq,np.ones(len(truth)))
            oof=np.empty(len(xt))
            for tr,ho in GroupKFold(3,shuffle=True,random_state=8827).split(xt,groups=legs):
                oof[ho]=fit(xt[tr],truth[tr],xt[ho],np.ones(len(tr)))
            family=logit(model[origin][it[t],:18].clip(.002,.998))*np.where(b.train_cell[t]%2,-1,1)[:,None]
            test=logit(model['full'][iq[q],:18].clip(.002,.998))*np.where(b.query_cell[q]%2,-1,1)[:,None]
            x=np.column_stack([standardized(xt,xq),standardized(family,test)])
            one,graph=neighbors(x,len(xt),np.r_[legs,query_groups])
            predictions=spread(one,graph,truth,prior,truth-oof)
            perm=np.arange(len(truth))
            rng=np.random.default_rng(6281)
            cell=b.train_cell[t]%4
            for level in np.unique(cell):
                rows=np.flatnonzero(cell==level)
                perm[rows]=rng.permutation(rows)
            placebo=spread(one,graph,truth[perm],prior,(truth-oof)[perm])['residual']
            for n,v in dict(base=prior,**predictions,shuffled_residual=placebo).items():
                output[n][q]=v>.5
            output['only_real']=output['residual']&~output['base']&~output['shuffled_residual']
        reports.setdefault(b.tag,[]).append((q,b.query_error,f,output))
    result={}
    for tag,rows in reports.items():
        q,wrong,fold=(np.concatenate([v[j] for v in rows]) for j in range(3))
        result[tag]={}
        for name in rows[0][-1]:
            a=np.concatenate([v[3][name] for v in rows])
            r=ticker(q,wrong,a)
            r['net_by_fold']=[int((a*np.where(wrong,1,-1)*(fold==k)).sum()) for k in range(8)]
            result[tag][name]=r
        print(tag,{n:(v['fixed'],v['harmed']) for n,v in result[tag].items()},flush=True)
    REPORT.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
