"""Marginalize unobserved outcomes in a model of the count transported per complete booking."""
from itertools import product
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load,strata
from broad_error_vote_tables_n23 import bank
from broad_geographic_transfer_n23 import BILLS
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, evaluate, write
from exaone_validation_data import CATEGORICAL, compact_features
from released_correction_audit_n23 import development

REPORT = ROOT/"reports/nov23_booking_annotation"


def contexts(frame, groups, row, y, t, q):
    """Cross-fit passenger probabilities for every roster member in the training bookings."""
    observed=np.full(len(frame),-1,int)
    observed[row[t]]=y[t]
    fit_ids=np.flatnonzero(np.isin(groups,groups[row[t]]))
    query_ids=np.flatnonzero(np.isin(groups,groups[row[q]]))
    assert not np.intersect1d(groups[fit_ids],groups[query_ids]).size
    assert (observed[query_ids]==-1).all()
    train_scores=np.full(len(frame),np.nan)
    queries=[]
    for a,b in GroupKFold(3,shuffle=True,random_state=512).split(t,groups=groups[row[t]]):
        left=row[t[a]]
        right=np.flatnonzero(np.isin(groups,groups[row[t[b]]]))
        assert not np.intersect1d(groups[left],groups[right]).size
        model=CatBoostClassifier(iterations=270,depth=6,learning_rate=.065,random_seed=114,
                                 verbose=False,thread_count=2,allow_writing_files=False)
        model.fit(frame.iloc[left],observed[left],cat_features=CATEGORICAL)
        train_scores[right]=model.predict(frame.iloc[right],prediction_type="RawFormulaVal")
        score=np.full(len(frame),np.nan)
        score[query_ids]=model.predict(frame.iloc[query_ids],prediction_type="RawFormulaVal")
        queries.append(score)
    assert np.isfinite(train_scores[fit_ids]).all()
    return observed,fit_ids,train_scores,query_ids,queries


def blocks(frame,groups,observed,ids,prior,mode):
    # Each member uses the context from its true, full booking in every arm.
    x=frame.iloc[ids]
    p=expit(prior[ids])
    h=np.c_[np.ones(len(x)),prior[ids],(strata(x)==2),x.CryoSleep.eq("True"),x.Age.fillna(27)/40,
            np.log(x.BookingSize),x.HomePlanet.eq("Earth")]
    unique,inv=np.unique(groups[ids],return_inverse=True)
    agg=pd.DataFrame({"p":p,"paid":strata(x)==2}).groupby(inv).agg(["mean","std"])
    ctx=np.c_[np.ones(len(agg)),agg.fillna(0).to_numpy(),np.log(np.bincount(inv))]
    members=pd.Series(np.arange(len(ids))).groupby(inv,sort=True).agg(list)
    parts={}
    for n in np.unique(np.bincount(inv)):
        gids=np.flatnonzero(np.bincount(inv)==n)
        arr=np.array(members.iloc[gids].tolist())
        known=observed[ids[arr]]>=0
        if mode=="complete_only":
            take=known.all(1);arr,gids=arr[take],gids[take]
        if mode=="ignore":
            for count in np.unique(known.sum(1)):
                sel=known.sum(1)==count
                if count>0:
                    cut=arr[sel][known[sel]].reshape(-1,count)
                    parts[n,int(count)]=cut,ctx[gids[sel]]
        elif len(arr):
            parts[n,n]=arr,ctx[gids]
    output=[]
    for (_,n),(arr,context) in parts.items():
        states=np.array(list(product((0.,1.),repeat=n)))
        k=states.sum(1)
        # The independent model shares the entire first-order design. Joint
        # parameters can additionally favor extremes and near-unanimous counts.
        coupling=np.column_stack([k==0,k==n,k*(n-k)/max(n,1)**2]) if n>1 else np.zeros((2,3))
        interactions=(context[:,None,:,None]*coupling[None,:,None,:]).reshape(len(arr),len(states),-1)
        h_group=h[arr]
        # Mean prior of the entire booking also enters the individual model.
        p_group=np.broadcast_to(context[:,None,1,None],(len(arr),n,1))
        unary=np.einsum("si,bid->bsd",states,np.concatenate([h_group,p_group],axis=2))
        design=np.concatenate([unary,interactions],axis=2)
        truth=observed[ids[arr]]
        mask=((truth[:,None,:]<0)|(truth[:,None,:]==states[None,:,:])).all(2)
        assert mask.any(1).all()
        output.append((ids[arr],states,prior[ids[arr]]@states.T,design,mask,unary.shape[2]))
    return output


def train(blocks,joint):
    dims=blocks[0][3].shape[2] if joint else blocks[0][-1]
    penalty=np.r_[np.full(blocks[0][-1],5.),np.full(dims-blocks[0][-1],12.)]
    def objective(w):
        loss=.5*penalty@(w*w)
        grad=penalty*w
        for _,_,offset,design,allowed,_ in blocks:
            a=design[:,:,:dims]
            z=offset+a@w
            full=logsumexp(z,axis=1,keepdims=True)
            kept=np.where(allowed,z,-np.inf)
            partial=logsumexp(kept,axis=1,keepdims=True)
            loss+=(full-partial).sum()
            grad+=np.einsum("ij,ijk->k",np.exp(z-full)-np.exp(kept-partial),a)
        return loss,grad
    sol=minimize(objective,np.zeros(dims),jac=True,method="L-BFGS-B",options={"maxiter":350,"ftol":1e-10})
    assert sol.success,sol.message
    return sol.x


def predict(parts,w,raw_n):
    out=np.full(raw_n,np.nan)
    for ids,states,offset,design,_,_ in parts:
        z=offset+design[:,:,:len(w)]@w
        out[ids.ravel()]=(np.exp(z-logsumexp(z,axis=1,keepdims=True))@states).ravel()
    return out


def run():
    row,y,local,g,_=load()
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Name","Transported"))
    groups=raw.PassengerId.str.split("_").str[0].astype(int)
    frame=compact_features(raw)
    frame["BookingSize"]=groups.map(groups.value_counts())
    groups=groups.to_numpy()
    np.testing.assert_array_equal(groups[row],g)
    pd.testing.assert_frame_equal(frame.iloc[row].reset_index(drop=True),local)
    labels,release,masks=development()
    np.testing.assert_array_equal(labels,y)
    miss=np.logical_and.reduce([s!=y for s in release.values()])
    state=strata(local)
    core=miss&(((state==2)&y)|((state==0)&~y))
    partial=np.bincount(g)[g]<local.BookingSize.to_numpy()
    masks={**masks,"release":miss,"core":core,"incomplete_core":core&partial,"multiple_core":core&(local.BookingSize.to_numpy()>1)}
    assert core.sum()==696 and partial.sum()==1119
    results,history={},{}
    for name in map(str,SEEDS):
        fold,base,ref,_,_=bank(name,row,y)
        scores={}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            obs,it,zt,iq,zq=contexts(frame,groups,row,y,t,q)
            inputs=blocks(frame,groups,obs,it,zt,"integrated")
            policies={"individual":train(inputs,False),"integrated":train(inputs,True),
                      "ignore":train(blocks(frame,groups,obs,it,zt,"ignore"),True),
                      "complete_only":train(blocks(frame,groups,obs,it,zt,"complete_only"),True)}
            query=[blocks(frame,groups,obs,iq,pr,"integrated") for pr in zq]
            for k,w in policies.items():
                p=np.mean([predict(part,w,len(frame))[row[q]] for part in query],axis=0)
                scores.setdefault(k,np.full(len(y),np.nan))[q]=p
        for k in list(scores):
            design=np.c_[logit(base.clip(.001,.999)),logit(scores["individual"].clip(.001,.999))]
            if k!="individual":
                design=np.c_[design,logit(scores[k].clip(.001,.999))-design[:,1]]
            pred=np.full(len(y),np.nan)
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    pred[q]=LogisticRegression(C=.5,max_iter=400).fit(design[t],y[t]).predict_proba(design[q])[:,1]
            scores["update_"+k]=pred
        results[name],hits=evaluate(y,base,ref,scores,fold,False,masks)
        for k,(hit,_) in hits.items():history.setdefault(k,[]).append(hit)
        out=ART/"nov23_partial_booking_likelihood"
        out.mkdir(exist_ok=True)
        np.savez_compressed(out/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**scores)
        print(name,[(k,v["fixed"],v["harmed"]) for k,v in results[name].items()],flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in masks.items()} for k,v in history.items()}
    write(REPORT/"partial_likelihood.json",dict(plans=results,stable=stable,scope={s:int(m.sum()) for s,m in masks.items()}))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
