"""Learn which dissenting votes provide additional evidence after accounting for other models' votes."""
import json

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import fit
from evaluate_nested_release_n26 import DECK, REVIEW, protocols, sources, panel_predictions, outcome_scores, stages
from expanded_joint_error_witnesses_n26 import FAMILIES, load_scores
from frozen_queue_shared_errors_n26 import paired
from held_expert_portfolios_n26 import NEW
from tabnet_attention_reliability_n26 import signals
from tabnet_attention_transfer_n26 import historical_outputs
from target_mix_expert_routing_n26 import ROOT, batches, ticker

RESULT=ROOT/'reports/nov23_matched_learning_curve/dependence_of_challenger_votes.json'


def likelihood_tables(x):
    # Binary marginal and joint probabilities with consistent symmetric pseudocounts.
    n,p=x.shape
    v=x.astype(float)
    both=v.T@v
    count=v.sum(0)
    tables=np.stack([n-count[:,None]-count[None,:]+both,count[None,:]-both,count[:,None]-both,both])
    joint=(tables+1)/(n+4)
    marginal=(count+2)/(n+4)
    independent=np.stack([np.outer(1-marginal,1-marginal),np.outer(1-marginal,marginal),
                          np.outer(marginal,1-marginal),np.outer(marginal,marginal)])
    factors=np.log(joint/independent)
    info=(joint*factors).sum(0)
    assert info.shape==(p,p)
    return marginal,info,factors


def odds(train,query,wrong):
    prior=(wrong.sum()+1)/(len(wrong)+2)
    params=[likelihood_tables(train[wrong==label]) for label in (False,True)]
    info=(1-prior)*params[0][1]+prior*params[1][1]
    # Normalize before constructing a sparse graph so near-zero finite edges survive conversion.
    weights=1+info
    np.fill_diagonal(weights,0)
    a,b=minimum_spanning_tree(-weights).nonzero()
    assert len(a)==train.shape[1]-1
    out=np.zeros((len(query),2))
    for sign,(m,_,correction) in zip((-1,1),params,strict=True):
        individual=np.where(query,np.log(m),np.log1p(-m)).sum(1)
        pairs=correction[2*query[:,a]+query[:,b],a,b].sum(1)
        out[:,0]+=sign*individual
        out[:,1]+=sign*(individual+pairs)
    return out+np.log(prior/(1-prior))


def conditional(train,query,old,now,wrong):
    scores=np.empty((len(query),2))
    for side in np.unique(now):
        t=old==side
        scores[now==side]=odds(train[t],query[now==side],wrong[t])
    return scores


def run():
    raw,y,g,f,known,versions=paired()
    _,_,_,panels=sources()
    ext=load_scores(raw,y,g,panels)
    net={'full':signals(raw,y,g,f)[0],'historical':historical_outputs(raw,y,g,panels['historical'])[0]}
    experts={}
    p={}
    names=[*panels['full']['names'][:18],*NEW[:-2],*FAMILIES,'TabNet_mean']
    for tag,v in panels.items():
        p[tag]=panel_predictions(v,tag)
        s=outcome_scores(v,p[tag])
        ix=v['ids']
        experts[tag]=np.column_stack([s[:,:18],s[:,-len(NEW):-2],ext[tag][ix],net[tag][ix].mean(1)])>=.5
        assert experts[tag].shape==(len(ix),len(names))
    with np.load(DECK/'deck_estimates.npz') as z:
        eligible=z['eligible']
    items={}
    for detail,b in zip(protocols(panels,g,True),batches(True),strict=True):
        key,src,dst,t,q=detail
        origin='full' if b.tag=='full' else 'historical'
        ids=b.ids
        np.testing.assert_array_equal(src['ids'][t],b.train_ids)
        np.testing.assert_array_equal(dst['ids'][q],ids)
        assert known[ids].all() and not np.isin(g[b.train_ids],g[ids]).any()
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],ids)
            risk=np.column_stack([z['identities_error'],z['identities_outcome']])
        anchor,review,fare,now=stages(dst,p['full'],q,eligible[ids],risk)
        np.testing.assert_array_equal(anchor,versions[ids,6])
        np.testing.assert_array_equal(now,b.query_cell%2)
        common=(versions[ids,:7]==now[:,None]).all(1)&(review==now)&(fare==now)
        old=b.train_cell%2!=0
        train=experts[origin][t]!=old[:,None]
        query=experts['full'][q]!=now[:,None]
        loo=np.empty((len(t),2))
        for tr,qh in GroupKFold(3,shuffle=True,random_state=9182).split(t,groups=b.groups):
            assert not np.isin(b.groups[tr],b.groups[qh]).any()
            loo[qh]=conditional(train[tr],train[qh],old[tr],old[qh],b.train_error[tr])
        test=conditional(train,query,old,now,b.train_error)
        w=np.ones(len(t))
        base=fit(b.train_fields,b.train_error,b.query_fields,w)>.5
        actions=dict(base=base,independent=test[:,0]>0,tree=test[:,1]>0)
        for j,name in enumerate(('independent','tree')):
            xt=np.column_stack([b.train_fields,loo[:,j]])
            xq=np.column_stack([b.query_fields,test[:,j]])
            actions['calibrated_'+name]=fit(xt,b.train_error,xq,w)>.5
        items.setdefault(b.tag,[]).append((ids,common,b.query_error,query.sum(1),actions))
    report=dict(models=names,method_source='https://research.ibm.com/publications/approximating-discrete-probability-distributions-with-dependence-trees',evaluations={})
    for tag,block in items.items():
        ids,common,wrong,votes=[np.concatenate([s[i] for s in block]) for i in range(4)]
        assert len(ids)==len(np.unique(ids))
        levels={'zero':votes==0,'one':votes==1,'two_to_four':(votes>=2)&(votes<=4),
                'five_to_half':(votes>=5)&(votes<=len(names)//2),'majority':votes>len(names)//2}
        r=dict(rows=len(ids),common=ticker(common,wrong,np.zeros(len(ids),bool)),
               by_number_dissenting={k:dict(rows=int((common&m).sum()),wrong=int((common&m&wrong).sum())) for k,m in levels.items()},methods={})
        for n in block[0][-1]:
            act=np.concatenate([s[-1][n] for s in block])
            stats=ticker(common,wrong,act)
            stats['net_by_fold']=[int((act*common*np.where(wrong,1,-1)*(f[ids]==k)).sum()) for k in range(8)]
            stats['all_changes']=ticker(np.ones(len(ids),bool),wrong,act)
            r['methods'][n]=stats
        report['evaluations'][tag]=r
        print(tag,r['by_number_dissenting'],{n:(s['fixed'],s['harmed']) for n,s in r['methods'].items()},flush=True)
    RESULT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
