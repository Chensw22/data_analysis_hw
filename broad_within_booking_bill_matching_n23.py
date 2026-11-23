"""Conditional likelihood for the observed assignment of two bills to two actual companions."""
from itertools import combinations
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from broad_booking_canonical_n23 import load_roster
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_link_authenticity_n23 import REPORT
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired,read_opinions
from broad_partition_evidence_n23 import ART,SEEDS,write


def pair_index(groups,idx):
    table=pd.Series(idx).groupby(groups[idx]).agg(list)
    pairs=[(i,j,1/(len(m)-1)) for m in table for i,j in combinations(m,2)]
    a,b,w=np.array(pairs).T
    return a.astype(int),b.astype(int),w


def encode(frame):
    identity=pd.get_dummies(frame[['HomePlanet','CryoSleep','Destination','Deck','Side','VIP']],dtype=float).to_numpy()
    age=frame.Age.fillna(27).to_numpy()
    identity=np.c_[np.ones(len(age)),identity,age/45,(age<13).astype(float),frame.Age.isna()]
    amounts=frame[BILLS].to_numpy(float)
    bills=np.c_[np.log1p(np.nan_to_num(amounts))/6, amounts>0,np.isnan(amounts),
                np.log1p(np.nansum(amounts,axis=1))/6,(amounts>0).sum(1)/3]
    return identity,bills


def terms(person,bills,i,j,yi,yj):
    # The difference of the two total compatibilities for the real vs swapped allocation.
    delta=bills[i]-bills[j]
    fixed=person[i]-person[j]
    specific=yi[:,None]*person[i]-yj[:,None]*person[j]
    covariates=np.column_stack([fixed,specific])
    return (covariates[:,:,None]*delta[:,None,:]).reshape(len(i),-1)


def fit_score(x,w):
    def loss(beta):
        v=x@beta
        return (w@np.logaddexp(0,-v)+6*(beta@beta),-(x.T@(w*expit(-v)))+12*beta)
    ans=minimize(loss,np.zeros(x.shape[1]),method='L-BFGS-B',jac=True,options={'maxiter':350})
    if not ans.success: raise RuntimeError(ans.message)
    return ans.x


def fold_fit(frame,g,size,person,bills,targets,t,q):
    # All members' visible attributes are available, including those without development labels.
    all_q=np.flatnonzero(np.isin(g,g[q]))
    i,j,w=pair_index(g,t)
    a,b,_=pair_index(g,all_q)
    allx=np.c_[person,bills,np.log(size)]
    model=CatBoostClassifier(iterations=300,depth=5,learning_rate=.06,l2_leaf_reg=4,thread_count=2,
                             verbose=False,allow_writing_files=False,random_seed=806).fit(allx[t],targets[t])
    def forecast(first,second):
        feat=np.c_[person[first],bills[second],np.log(size[first])]
        return model.predict_proba(feat)[:,1]
    pa=(forecast(a,a)+forecast(a,b))/2
    pb=(forecast(b,a)+forecast(b,b))/2
    states=np.array([[0,0],[0,1],[1,0],[1,1]])
    prior=np.stack([np.where(ha,pa,1-pa)*np.where(hb,pb,1-pb) for ha,hb in states],axis=1)
    def collapse(left,right):
        total=np.bincount(np.r_[a,b],weights=np.r_[left,right],minlength=len(frame))
        n=np.bincount(np.r_[a,b],minlength=len(frame))
        return total[q]/n[q]
    real=targets.copy()
    assert np.isin(real[t],[0,1]).all()
    # Preserve the joint spending/target and sleep/target distribution of labeled training people.
    shuffle=real.copy()
    risk=frame[['Spa','RoomService','VRDeck']].gt(0).any(axis=1).to_numpy()
    rng=np.random.default_rng(537)
    for ids in pd.DataFrame(dict(risk=risk[t],sleep=frame.CryoSleep.to_numpy()[t])).groupby(['risk','sleep']).indices.values():
        s=t[ids]
        shuffle[s]=rng.permutation(real[s])
    result={'direct':forecast(q,q),'symmetrized':collapse(pa,pb)}
    sanity={}
    for name,labels in [('owner',real),('shuffled',shuffle)]:
        x=terms(person,bills,i,j,labels[i],labels[j])
        beta=fit_score(x,w)
        prob=[]
        for s1,s2 in states:
            prob.append(expit(terms(person,bills,a,b,np.full(len(a),s1),np.full(len(b),s2))@beta))
        likelihood=np.column_stack(prob)
        post=prior*likelihood
        post/=post.sum(1)[:,None]
        np.testing.assert_allclose(post.sum(1),1)
        result[name]=collapse(post[:,2]+post[:,3],post[:,1]+post[:,3])
        # Held-out labels only evaluate the ownership model after probabilities are obtained.
        scored=np.isin(a,q)&np.isin(b,q)&np.any(bills[a]!=bills[b],axis=1)
        z=likelihood[scored,2*real[a[scored]]+real[b[scored]]]
        sanity[name]=dict(pairs=len(z),mean_log_loss=float(-np.log(z.clip(1e-10)).mean()),
                          accuracy=float(np.mean((z>.5)+.5*(z==.5))))
    return result,sanity


def main():
    row,y,_,g,_=load()
    wrong,right,hard=cohorts(row,y)
    frame,groups,size,_=load_roster()
    np.testing.assert_array_equal(g,groups[row])
    target=np.full(len(frame),-1,int)
    target[row]=y
    person,bills=encode(frame)
    many=size[row]>1
    out=ART/'nov23_within_booking_bill_matching'
    out.mkdir(exist_ok=True)
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        base=op[:,0]
        anchor=logit(base.clip(.002,.998))
        scores,diagnostics={},[]
        for f in np.unique(fold):
            q=np.flatnonzero((fold==f)&many)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[fold==f]))
            assert not np.intersect1d(g[t],g[q]).size
            p,metric=fold_fit(frame,groups,size,person,bills,target,row[t],row[q])
            for k,v in p.items():scores.setdefault(k,base.copy())[q]=v
            diagnostics.append(metric)
        hits={'reference':ref==y}
        active=many&(ref==(base>=.5))
        for k,p in scores.items():
            hits[k]=np.where(many,p>=.5,ref)==y
            hits['quarter_'+k]=np.where(active,.75*base+.25*p>=.5,ref)==y
            if k in ('owner','shuffled'):
                increment=logit(p.clip(.002,.998))-logit(scores['symmetrized'].clip(.002,.998))
                hits['shift_'+k]=np.where(active,anchor+increment>=0,ref)==y
        metrics={k:paired(v,ref==y) for k,v in hits.items()}
        plans[seed]=dict(changes=metrics,owner_recognition=diagnostics)
        if seed!='v131':
            for k,v in hits.items():history.setdefault(k,[]).append(v)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in metrics.items()},flush=True)
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,base=base,ref=ref,**scores)
    r=np.stack(history['reference'])
    stats={k:dict(common_correct=int((np.all(v,axis=0)&wrong).sum()),hard_stable=int((np.all(v,axis=0)&hard).sum()),
                  hard_ever=int((np.any(v,axis=0)&hard).sum()),stable_harms=int((~np.any(v,axis=0)&right).sum()),
                  gain95_pp=interval((np.stack(v).astype(float)-r).mean(0),g)) for k,v in history.items()}
    write(REPORT/'bill_owners.json',dict(plans=plans,stable=stats))


if __name__=='__main__':
    with threadpool_limits(1): main()
