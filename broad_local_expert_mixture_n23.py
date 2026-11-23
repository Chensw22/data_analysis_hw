"""Estimate which forecast to use from the query's reconstruction by both outcome classes."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import REPORT, cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write


def mixture(train_x,query_x,train_p,query_p,y):
    prior=np.log([.64,.12,.12,.12])
    x=np.c_[np.ones(len(train_x)),train_x]
    q=np.c_[np.ones(len(query_x)),query_x]
    observed=np.where(y[:,None],train_p,1-train_p)
    def objective(flat):
        w=flat.reshape(x.shape[1],3)
        logits=prior+np.c_[np.zeros(len(x)),x@w]
        fractions=softmax(logits,axis=1)
        likelihood=(fractions*observed).sum(1)
        gradient=fractions*(1-observed/likelihood[:,None])
        return -np.log(likelihood).sum()+4*(flat@flat),(x.T@gradient[:,1:]).ravel()+8*flat
    fit=minimize(objective,np.zeros(x.shape[1]*3),method='L-BFGS-B',jac=True)
    if not fit.success: raise RuntimeError(fit.message)
    weights=softmax(prior+np.c_[np.zeros(len(q)),q@fit.x.reshape(x.shape[1],3)],axis=1)
    return (weights*query_p).sum(1),weights


def main():
    row,y,_,g,_=load()
    wrong,right,hard=cohorts(row,y)
    plans,hist={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_local_profile_fit/{seed}.npz') as a, np.load(ART/f'nov23_class_hulls/{seed}.npz') as b:
            for k,v in [('row',row),('y',y)]:np.testing.assert_array_equal(a[k],v)
            for k in ['row','y','fold','ref','base']:np.testing.assert_array_equal(a[k],b[k])
            fold,ref,base=a['fold'],a['ref'],a['base']
            scores=np.column_stack([base,a['ordinary'],a['balanced'],a['local_slopes']]).clip(.001,.999)
            # Distances depend solely on the reference library, external to this entire fold.
            raw=b['raw'][:,:,[0,1,2,3,5,6]]
        support=np.c_[raw.min(1),raw[:,0]-raw[:,1]]
        direction=np.where(ref,1.,-1.)
        votes=(scores[:,1:]>=.5)==ref[:,None]
        apparent=np.c_[np.abs(base-.5),ref,(scores[:,1:]-base[:,None])*direction[:,None],votes.sum(1)]
        arms={k:ref.copy() for k in ('constant','opinions','profiles','shuffled')}
        avg={k:[] for k in arms}
        for f in np.unique(fold):
            ids=np.flatnonzero((fold==f)&(ref==(base>=.5)))
            for a,b in GroupKFold(3,shuffle=True,random_state=536).split(ids,groups=g[ids]):
                t,q=ids[a],ids[b]
                assert not np.intersect1d(g[t],g[q]).size
                # Preserve the marginal geometry and the ballot direction of each passenger.
                # Shuffle whole profiles independently on the calibration and query sides.
                sham=support.copy()
                rng=np.random.default_rng(284)
                for idx in [t,q]:
                    for v in (False,True):
                        sub=idx[ref[idx]==v]
                        sham[sub]=support[rng.permutation(sub)]
                for k,design in [('constant',np.empty((len(y),0))),('opinions',apparent),
                                 ('profiles',np.c_[apparent,support]),('shuffled',np.c_[apparent,sham])]:
                    if design.shape[1]:
                        scale=StandardScaler().fit(design[t])
                        tr=scale.transform(design[t]).clip(-6,6)
                        va=scale.transform(design[q]).clip(-6,6)
                    else: tr,va=design[t],design[q]
                    p,w=mixture(tr,va,scores[t],scores[q],y[t])
                    arms[k][q]=p>=.5
                    avg[k].extend(w[:,0].tolist())
        hits={k:(v==y) for k,v in {'reference':ref,**arms}.items()}
        metrics={k:paired(v,ref==y) for k,v in hits.items()}
        plans[seed]=dict(changes=metrics,mean_base_weight={k:float(np.mean(v)) for k,v in avg.items()})
        if seed!='v131':
            for k,v in hits.items():hist.setdefault(k,[]).append(v)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in metrics.items()},flush=True)
    r=np.stack(hist['reference'])
    stats={k:dict(common_correct=int((np.all(v,axis=0)&wrong).sum()),hard_stable=int((np.all(v,axis=0)&hard).sum()),
                  hard_ever=int((np.any(v,axis=0)&hard).sum()),stable_harms=int((~np.any(v,axis=0)&right).sum()),
                  gain95_pp=interval((np.stack(v).astype(float)-r).mean(0),g)) for k,v in hist.items()}
    write(REPORT/'mixture.json',dict(plans=plans,stable=stats))


if __name__=='__main__':
    with threadpool_limits(1):main()
