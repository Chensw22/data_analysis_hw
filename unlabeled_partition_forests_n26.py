"""Evaluate label-independent finite-life Mondrian trees and multiscale random Voronoi cells."""
import json
import numpy as np
from scipy.special import logit
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits

from exhaustive_family_quorums_n26 import inputs,summary
from full_roster_booking_validation_n23 import ROOT,OUT,data
from relaxed_passenger_transport_n26 import coordinates


def mondrian(x,y,t,q,seed,trees=40,horizon=2.0):
    total=np.zeros(len(y))
    prior=y[t].mean()
    def predict_node(bank,query,weight,parent,remaining,rng):
        if not len(query):
            return
        prob=(y[bank].sum()+6*parent)/(len(bank)+6)
        train=x[bank]
        low=train.min(0); high=train.max(0)
        widths=high-low
        rate=widths.sum()
        # Only geometry, elapsed time and training occupancy decide partitions.
        # Splitting/termination remains identical under every permutation of outcomes.
        elapsed=rng.exponential(1/rate) if rate>1e-12 and len(bank)>12 else np.inf
        distance=(np.maximum(low-x[query],0)+np.maximum(x[query]-high,0)).sum(1)
        retained=np.exp(-min(remaining,elapsed)*distance)
        total[query]+=weight*(1-retained)*parent
        weight=weight*retained
        if elapsed>=remaining:
            total[query]+=weight*prob
            return
        col=np.searchsorted(np.cumsum(widths),rng.random()*rate,side='right')
        cut=rng.uniform(low[col],high[col])
        left=train[:,col]<=cut
        direction=x[query,col]<=cut
        seeds=rng.integers(2**32,size=2)
        for use,ask,childseed in [(left,direction,seeds[0]),(~left,~direction,seeds[1])]:
            predict_node(bank[use],query[ask],weight[ask],prob,remaining-elapsed,np.random.default_rng(childseed))
    for child in np.random.default_rng(seed).integers(2**32,size=trees):
        predict_node(t,q,np.ones(len(q)),prior,horizon,np.random.default_rng(child))
    return total[q]/trees


def voronoi(x,y,t,q,seed):
    # Repeated randomly sampled training references in the original full metric.
    # At each scale every cell is assigned its labeled training-member class average.
    rng=np.random.default_rng(seed)
    sizes=(32,128,384)
    sums=np.zeros((len(q),len(sizes)))
    xt=np.asarray(x[t],np.float32); xq=np.asarray(x[q],np.float32)
    for _ in range(24):
        choice=rng.choice(len(t),sizes[-1],replace=False)
        bank=xt[choice]
        norm=np.einsum('ij,ij->i',bank,bank)
        train=norm-2*(xt@bank.T)
        query=norm-2*(xq@bank.T)
        for j,k in enumerate(sizes):
            index=train[:,:k].argmin(1)
            counts=np.bincount(index,minlength=k)
            rates=(np.bincount(index,weights=y[t],minlength=k)+6*y[t].mean())/(counts+6)
            sums[:,j]+=rates[query[:,:k].argmin(1)]
    return sums/24


def run():
    raw,y,g,f=data()
    names=['mondrian','voronoi','voronoi32','voronoi128','voronoi384','extra_trees']
    prob={key:np.full(len(y),np.nan) for key in names}
    for k in np.unique(f):
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        assert not np.isin(g[t],g[q]).any()
        x=coordinates(raw,g,t)
        prob['mondrian'][q]=mondrian(x,y,t,q,1540+int(k))
        values=voronoi(x,y,t,q,1620+int(k))
        prob['voronoi'][q]=values.mean(1)
        for j,tag in enumerate(names[2:5]):
            prob[tag][q]=values[:,j]
        forest=ExtraTreesClassifier(n_estimators=80,min_samples_leaf=8,max_features=.8,n_jobs=2,random_state=2180+int(k))
        prob['extra_trees'][q]=forest.fit(x[t],y[t]).predict_proba(x[q])[:,1]
        print('held fold',int(k),flush=True)
    for p in prob.values():
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    out=ROOT/'artifacts/nov26_unlabeled_partition_forests'
    out.mkdir(exist_ok=True)
    tmp=out/'tmp.npz'
    np.savez_compressed(tmp,passenger=raw.PassengerId.to_numpy(str),fold=f,y=y,row=np.arange(len(y)),**prob)
    archive=out/'eight_fold_oof.npz'
    tmp.replace(archive)
    with np.load(OUT/'mitra96.npz') as z,np.load(archive) as stored:
        for key in ('y','passenger','fold'):
            np.testing.assert_array_equal(z[key],stored[key])
        mitra=z['p'].clip(.003,.997)
    _,cur,_=inputs()
    parent=cur['parent']
    blend={n:(.75*logit(mitra)+.25*logit(p.clip(.003,.997)))>=0 for n,p in prob.items()}
    actions={}
    for n,p in prob.items():
        actions[n]=(p>=.5)!=parent
        actions['blend_'+n]=blend[n]!=parent
        if n!='extra_trees':
            actions['unique_'+n]=(blend[n]!=parent)&(blend['extra_trees']==parent)&((mitra>=.5)==parent)
    actions['both_unlabelled']=actions['unique_mondrian']&actions['unique_voronoi']
    report=dict(accuracy={n:float(((p>=.5)==y).mean()) for n,p in prob.items()},
                blend_accuracy={n:float((v==y).mean()) for n,v in blend.items()},scopes={})
    for name,m in [('all',cur['known']),('common',cur['common']),('new',cur['known']&cur['new'])]:
        record={n:summary(v,parent!=y,m,f) for n,v in actions.items()}
        for n,p in blend.items():
            record[n+'_vs_mitra']=summary(p!=(mitra>=.5),(mitra>=.5)!=y,m,f)
        report['scopes'][name]=record
    report['specification']=('Mondrian fixed lifetime 2, exact bounding-box split rate, minimum 12 training '
       'passengers before further splitting, 40 random trees, labels used only in hierarchical smoothed class '
       'probabilities, distance-dependent extrapolation beyond training boxes. Random Voronoi has 24 independent '
       'training-only site samples and 32/128/384 sites, full covariate Euclidean metric. Supervised ExtraTrees '
       'uses exactly the same fold-preprocessed anonymous fields. Blend 25 percent on the log odds scale. '
       'Previous archive broad_mondrian_partition_n23.py used label-dependent purity stopping without finite lifetime.')
    (ROOT/'reports/nov23_matched_learning_curve/unlabeled_partition_forests.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['accuracy'],report['blend_accuracy'],flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
