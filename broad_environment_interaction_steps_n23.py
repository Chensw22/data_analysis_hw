"""Estimate environment-specific deviations in each boosting update with empirical pooling."""
import numpy as np
from catboost import CatBoostClassifier
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_cross_environment_leaf_steps_n23 import CATEGORICAL,DEPTH,OUT,RATE,REPORT,TREE_COUNT,assign,centered
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired,read_opinions
from broad_partition_evidence_n23 import ART,SEEDS,write


def member_environments(t,q,g,region,random):
    result=[]
    for kind,seed in [('order',211),('cabin',813)]:
        a=assign(t,g,region,'random' if random else kind,4,seed)
        if random:
            ids,inv=np.unique(g[q],return_inverse=True)
            b=np.random.default_rng(seed).integers(4,size=len(ids))[inv]
        elif kind=='order':
            cut=[g[t[a==j]].max() for j in range(3)]
            np.testing.assert_array_equal(np.searchsorted(cut,g[t]),a)
            b=np.searchsorted(cut,g[q])
        else:
            mapping=dict(zip(region[t],a))
            np.testing.assert_array_equal([mapping[x] for x in region[t]],a)
            b=np.array([mapping.get(x,-1) for x in region[q]])
        result.append(np.r_[a,b])
    return result


def local_step(gradient,hessian,leaf,env,global_step,mode):
    people=len(gradient)
    nleaf=2**DEPTH
    joint=env[:people]*nleaf+leaf[:people]
    hg=np.bincount(joint,weights=hessian,minlength=4*nleaf).reshape(4,nleaf)
    dg=np.bincount(joint,weights=gradient,minlength=4*nleaf).reshape(4,nleaf)
    leftover=dg-hg*global_step
    intercept=leftover.sum(1)/(hg.sum(1)+30.)
    residual=leftover-hg*intercept[:,None]
    # Demean within each leaf so no global leaf intercept is counted again.
    residual-=hg*(residual.sum(0)/hg.sum(0).clip(.001))
    good=hg>2
    mass=hg*good
    centered_score=np.where(good,residual,0.)
    centered_score-=mass*(centered_score.sum(0)/mass.sum(0).clip(.001))
    q=(centered_score**2/mass.clip(.001)).sum()
    df=np.maximum(good.sum(0)-1,0).sum()
    scale=(mass*(1-mass/mass.sum(0).clip(.001))).sum()
    tau2=max(0.,(q-df)/max(scale,.001))
    if mode=='fixed':
        tau2=.04
    if mode=='intercept':
        tau2=0.
    offset=residual*tau2/(1+hg*tau2)
    assert np.isfinite(offset).all()
    effect=np.zeros(len(leaf))
    known=env>=0
    effect[known]=intercept[env[known]]+offset[env[known],leaf[known]]
    return effect,tau2


def run():
    row,y,frame,g,region=load()
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(row,z['row'])
        np.testing.assert_array_equal(y,z['y'])
        wrong=(z['votes']!=y[None,:,None]).all((0,2))
        right=(z['votes']==y[None,:,None]).all((0,2))
    assert wrong.sum()==983 and right.sum()==5343
    stats,hist,tau={},{},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,bank,_=read_opinions(seed,row,y)
        with np.load(OUT/f'{seed}.npz') as z:
            for k,v in [('row',row),('y',y),('fold',fold),('ref',ref)]:np.testing.assert_array_equal(z[k],v)
            pooled=z['pooled']
        pred={}; tau[seed]=[]
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f);t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            tree=CatBoostClassifier(iterations=TREE_COUNT,learning_rate=RATE,depth=DEPTH,verbose=False,
                                    thread_count=2,allow_writing_files=False,random_seed=811)
            tree.fit(frame.iloc[t],y[t],cat_features=CATEGORICAL)
            leaves=tree.calc_leaf_indexes(frame.iloc[np.r_[t,q]]).astype(int)
            real=member_environments(t,q,g,region,False)
            sham=member_environments(t,q,g,region,True)
            info={}
            for name,env,mode in [('control',[],'intercept'),('main_only',real,'intercept'),('fixed',real,'fixed'),
                                  ('empirical',real,'empirical'),('random',sham,'empirical')]:
                eta=np.full(len(leaves),logit(y[t].mean()))
                run_tau=[]
                for leaf in leaves.T:
                    p=expit(eta[:len(t)]); grad=y[t]-p; h=p*(1-p)
                    count=2**DEPTH
                    root,across=centered(np.bincount(leaf[:len(t)],weights=grad,minlength=count),
                                         np.bincount(leaf[:len(t)],weights=h,minlength=count))
                    step=root+across
                    total=step[leaf]
                    for e in env:
                        offset,variance=local_step(grad,h,leaf,e,step,mode)
                        total+=offset/len(env)
                        run_tau.append(variance)
                    eta+=RATE*total
                prob=expit(eta[len(t):])
                if name=='control':
                    np.testing.assert_allclose(prob,pooled[q],atol=1e-10,rtol=0)
                if run_tau:
                    info[name]=dict(nonzero=float(np.mean(np.array(run_tau)>0)),mean=float(np.mean(run_tau)))
                pred.setdefault(name,np.full(len(y),np.nan))[q]=prob
            tau[seed].append(info)
        votes={k:v>=.5 for k,v in pred.items()}
        for k in ['main_only','fixed','empirical','random']:
            add=logit(pred[k])-logit(pooled)
            votes['transfer_'+k]=np.where(ref==(bank[:,0]>=.5),logit(bank[:,0].clip(.002,.998))+add>=0,ref)
        votes['reference']=ref
        hits={k:v==y for k,v in votes.items()}
        stats[seed]={}
        for k,v in hits.items():
            compare=hits['reference' if k.startswith('transfer_') else 'control']
            stats[seed][k]=dict(matched=paired(v,compare),vs_reference=paired(v,hits['reference']))
            if seed!='v131':hist.setdefault(k,[]).append(v)
        print(seed,stats[seed]['empirical'],stats[seed]['transfer_empirical'],flush=True)
        np.savez_compressed(OUT/f'interaction_{seed}.npz',row=row,y=y,fold=fold,base=bank[:,0],ref=ref,**pred)
    results={}
    panel={k:np.stack(v) for k,v in hist.items()}
    for k,h in panel.items():
        comp=panel['reference' if k.startswith('transfer_') else 'control']
        delta=(h.astype(float)-comp).mean(0)
        results[k]=dict(gain=float(100*delta.mean()),interval_pp=interval(delta,g),fixed983=int(h[:,wrong].all(0).sum()),
                        hurt5343=int((~h[:,right]).all(0).sum()))
    write(REPORT/'interaction.json',dict(plans=stats,grouped=results,variances=tau))


if __name__=='__main__':
    with threadpool_limits(2):run()
