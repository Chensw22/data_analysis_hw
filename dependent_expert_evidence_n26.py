"""Model how distinct expert opinions co-occur under each outcome, conditional on the native risk."""
import json
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from held_expert_portfolios_n26 import DECK, ROOT, outcome_scores, panel_predictions, replay, sources, splits, ticker


def opinions(v, p):
    native=v['off']*np.where(v['anchor'],-1,1)
    prob=outcome_scores(v,p)
    odds=logit(prob.clip(.003,.997))
    center=np.median(odds,axis=1)
    # Quantify disagreement with the typical expert independently of its absolute confidence.
    relative=odds-center[:,None]
    features=np.column_stack([native,center,center-native,np.std(odds,axis=1),v['common']])
    return features,relative


def pair_counts(x):
    n,k=x.shape
    a=(x[:,:,None]==np.arange(3)).reshape(n,k*3).astype(float)
    return (a.T@a).reshape(k,3,k,3).transpose(0,2,1,3)


def evidence(xt,xq,base_train,base_query,y):
    cuts=np.quantile(xt,[1/3,2/3],axis=0)
    encode=lambda x:(x>cuts[0]).astype(int)+(x>cuts[1]).astype(int)
    t,q=encode(xt),encode(xq)
    _,i=np.unique(t.T,axis=0,return_index=True)
    unique=np.sort(i)
    t,q=t[:,unique],q[:,unique]
    counts=pair_counts(t)
    mass=(counts+.5)/(len(t)+4.5)
    prod=mass.sum(3,keepdims=True)*mass.sum(2,keepdims=True)
    info=(mass*np.log(mass/prod)).sum((2,3))
    np.fill_diagonal(info,0)
    # Every non-diagonal edge has a positive selection weight, even when evidence is tied.
    graph=minimum_spanning_tree(-info-1.e-6)
    _,parents=breadth_first_order(graph+graph.T,0,directed=False)
    nodes=np.arange(t.shape[1])
    child=nodes[parents>=0]
    parent=parents[child]
    assert len(child)==len(nodes)-1
    split=lambda b:np.digitize(b,[-1.4,0,1.4])
    bt,bq=split(base_train),split(base_query)
    scores=np.zeros((len(q),2))
    for label in (0,1):
        subset=y==label
        global_pairs=pair_counts(t[subset])+.5
        global_pairs/=global_pairs.sum((2,3),keepdims=True)
        for band in range(4):
            selected=subset&(bt==band)
            held=np.flatnonzero(bq==band)
            if not len(held):
                continue
            pair=pair_counts(t[selected])+15*global_pairs
            pair/=pair.sum((2,3),keepdims=True)
            # Off-diagonal joint tables have proper smoothed marginal distributions.
            neighbor=(nodes+1)%len(nodes)
            marg=pair[nodes,neighbor].sum(2)
            field=q[held]
            lp=np.log(marg[nodes,field]).sum(1)
            conditional=pair[child,parent,field[:,child],field[:,parent]]/pair[child,parent].sum(1)[np.arange(len(child)),field[:,parent]]
            tree=np.log(marg[0,field[:,0]])+np.log(conditional).sum(1)
            scores[held]+= (1 if label else -1)*np.column_stack([lp,tree])
    return np.clip(scores,-120,120)


def run():
    _,y,g,panels=sources()
    predictions={tag:panel_predictions(v,tag) for tag,v in panels.items()}
    x={tag:opinions(v,predictions[tag]) for tag,v in panels.items()}
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
        agrees=(z['label']>=0)&(z['fine'].argmax(1)==z['label'])
    output={}
    for key,src,dst,t,q in splits(panels,g):
        if not (key.startswith('bookingmix') or key=='historical_to_new'):
            continue
        a,b=src['ids'][t],dst['ids'][q]
        assert not np.isin(g[a],g[b]).any()
        f,s=x['historical'][0][t],x['historical'][1][t]
        fq,sq=x['full'][0][q],x['full'][1][q]
        fitted=np.empty((len(t),2))
        for train,held in GroupKFold(3,shuffle=True,random_state=6936).split(s,groups=g[a]):
            assert not np.isin(g[a][train],g[a][held]).any()
            fitted[held]=evidence(s[train],s[held],f[train,0],f[held,0],y[a][train])
        transferred=evidence(s,sq,f[:,0],fq[:,0],y[a])
        prior=replay(key,q,dst,predictions['full'],enabled)
        models={}
        for name,k in [('base',None),('independent',0),('tree',1)]:
            train_features=f if k is None else np.column_stack([f,fitted[:,k]/8])
            query_features=fq if k is None else np.column_stack([fq,transferred[:,k]/8])
            fit=LogisticRegression(C=.5,max_iter=500).fit(train_features,y[a])
            models[name]=fit.predict_proba(query_features)[:,1]
        tag='bookingmix' if key.startswith('bookingmix') else key
        output.setdefault(tag,[]).append((b,prior,dst['common'][q],models))
    report={}
    for tag,entries in output.items():
        ids,base,common=[np.concatenate([e[k] for e in entries]) for k in range(3)]
        assert len(np.unique(ids))==len(ids)
        scores={k:np.concatenate([e[3][k] for e in entries]) for k in entries[0][3]}
        actions={k:(pr>=.5)!=base for k,pr in scores.items()}
        actions.update(tree_increments=actions['tree']&~actions['base'],
                       tree_only=actions['tree']&~actions['independent']&~actions['base'])
        masks={'all':np.ones(len(ids),bool),'previous_unanimity':common,
               'paid_agree_unanimity':common&enabled[ids]&agrees[ids]}
        report[tag]={k:{n:ticker(m,base!=y[ids],action) for n,m in masks.items()} for k,action in actions.items()}
        report[tag]['log_loss']={k:float(-np.log(np.where(y[ids],p,1-p).clip(1e-9)).mean()) for k,p in scores.items()}
        print(tag,report[tag]['tree_increments'],report[tag]['log_loss'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/dependent_expert_evidence.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
