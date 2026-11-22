"""Smooth unlabeled booking marks along each deck and side with a hidden Markov model."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_geographic_transfer_n23 import BILLS, load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from exaone_validation_data import CATEGORICAL
from released_correction_audit_n23 import development

REPORT = ROOT / "reports/nov23_solo_structure"


def manifest():
    raw = pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv", usecols=lambda c:c not in ("Transported","Name"))
                     for s in ("train","test")],ignore_index=True)
    group = raw.PassengerId.str.split("_").str[0].astype(int)
    cabin = raw.Cabin.str.split("/",expand=True)
    paid = raw[BILLS].gt(0).any(axis=1)
    reg = pd.DataFrame(dict(group=group, channel=cabin[0]+"-"+cabin[2], planet=raw.HomePlanet,
                            dest=raw.Destination, sleep=raw.CryoSleep, paid=paid,
                            age=raw.Age, spend=raw[BILLS].sum(axis=1,min_count=1)))
    def mode(x):
        counts=x.dropna().value_counts(sort=False)
        return str(sorted(counts[counts==counts.max()].index)[0]) if len(counts) else "?"
    bookings=reg.groupby("group",sort=True)
    frame=bookings[["channel","planet","dest","sleep"]].agg(mode)
    frame["party"]=bookings.size().clip(upper=4)
    frame["age_band"]=np.digitize(bookings.age.median().fillna(-1),[0,13,22,35,55])
    frame["payer_fraction"]=pd.cut(bookings.paid.mean(),[-1,0,.5,.99999,1],labels=False)
    frame["spending_band"]=np.digitize(bookings.spend.median().fillna(-1),[0,1,400,1000,2000])
    order=np.lexsort((frame.index,frame.channel))
    channels=frame.channel.to_numpy()[order]
    blocks=np.split(np.arange(len(order)),np.flatnonzero(channels[1:]!=channels[:-1])+1)
    marks=frame.drop(columns="channel").apply(lambda col:pd.factorize(col,sort=True)[0]).to_numpy()[order]
    ids=frame.index.to_numpy()[order]
    return ids,marks,blocks


def smooth(lik,transition,blocks):
    n,k=lik.shape
    prior=np.empty((n,k))
    future=np.empty((n,k))
    for idx in blocks:
        prior[idx[0]]=1/k
        future[idx[-1]]=1
        for i in idx[:-1]:
            u=prior[i]*lik[i]
            prior[i+1]=(u/u.sum())@transition
        for i in idx[:0:-1]:
            u=transition@(lik[i]*future[i])
            future[i-1]=u/u.sum()
    loo=prior*future
    loo/=loo.sum(1,keepdims=True)
    post=loo*lik
    post/=post.sum(1,keepdims=True)
    return loo,post,prior,future


def context(marks,blocks):
    x=np.column_stack([np.eye(c.max()+1)[c] for c in marks.T])
    k=5
    start=KMeans(n_clusters=k,n_init=3,random_state=229).fit_predict(x)
    post=np.eye(k)[start] + .01
    transition=np.eye(k)*.7+np.full((k,k),.3/k)
    prev=np.concatenate([b[:-1] for b in blocks])
    for _ in range(15):
        emissions=[]
        for c in marks.T:
            table=np.ones((int(c.max())+1,k))
            np.add.at(table,c,post)
            emissions.append(table[c]/table.sum(0))
        lik=np.prod(emissions,axis=0)
        loo,post,prior,future=smooth(lik,transition,blocks)
        u=prior[prev]*lik[prev]
        v=lik[prev+1]*future[prev+1]
        joint=u[:,:,None]*transition[None,:,:]*v[:,None,:]
        counts=(joint/joint.sum((1,2))[:,None,None]).sum(0)+1
        transition=counts/counts.sum(1,keepdims=True)
    local=np.zeros_like(x)
    fake=loo.copy()
    rng=np.random.default_rng(676)
    for b in blocks:
        fake[b]=loo[rng.permutation(b)]
        for j,i in enumerate(b):
            near=b[max(0,j-4):min(len(b),j+5)]
            near=near[near!=i]
            local[i]=x[near].mean(0) if len(near) else x.mean(0)
    return dict(hmm=loo,shuffled=fake,window=local),dict(mean_entropy=float(-(loo*np.log(loo)).sum(1).mean()),
            mean_self_transition=float(np.mean(np.diag(transition))),booking_count=len(marks),corridors=len(blocks))


def run():
    row,y,frame,groups,_=load()
    labels,released,masks=development()
    np.testing.assert_array_equal(labels,y)
    missed=np.logical_and.reduce([p!=y for p in released.values()])
    paid=missed&y&(strata(frame)==2)
    zero=missed&~y&(strata(frame)==0)
    solo=(frame.BookingSize==1).to_numpy()
    masks=dict(all=np.ones(len(y),bool),repeat=masks["repeat"],release=missed,core=paid|zero,
               solo=solo,solo_release=missed&solo,solo_core=(paid|zero)&solo,solo_paid=paid&solo,solo_zero=zero&solo)
    ids,marks,blocks=manifest()
    fields,diagnostic=context(marks,blocks)
    index=pd.Index(ids).get_indexer(groups)
    np.testing.assert_array_equal(ids[index],groups)
    base_frame=frame.copy()
    base_frame["booking_order"]=groups
    views=dict(own=base_frame)
    for name,matrix in fields.items():
        cols=pd.DataFrame(matrix[index],columns=[f"context_{i}" for i in range(matrix.shape[1])])
        views[name]=pd.concat([base_frame,cols],axis=1)
    record,history={},{}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            base,fold=z["base"],z["fold"]
        scores={"baseline":base}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(groups[t],groups[q]).size
            for k,v in views.items():
                fit=CatBoostClassifier(iterations=300,depth=6,learning_rate=.065,random_seed=813,
                    verbose=False,thread_count=2,allow_writing_files=False)
                fit.fit(v.iloc[t],y[t],cat_features=CATEGORICAL)
                scores.setdefault(k,np.full(len(y),np.nan))[q]=fit.predict_proba(v.iloc[q])[:,1]
            for k in views:
                p=np.column_stack([logit(base[q].clip(.002,.998)),logit(scores["own"][q].clip(.002,.998))])
                if k!="own":
                    add=logit(scores[k][q].clip(.002,.998))-p[:,1]
                    p=np.column_stack([p,add,add*solo[q]])
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(q,groups=groups[q]):
                    assert not np.intersect1d(groups[q[a]],groups[q[b]]).size
                    fit=LogisticRegression(C=.8,max_iter=400).fit(p[a],y[q[a]])
                    scores.setdefault("combined_"+k,np.full(len(y),np.nan))[q[b]]=fit.predict_proba(p[b])[:,1]
        record[seed]={}
        for k,p in scores.items():
            assert np.isfinite(p).all()
            hit=(p>=.5)==y
            good=(base>=.5)==y
            record[seed][k]=dict(accuracy=float(hit.mean()),by_scope={s:dict(correct=int((hit&m).sum()),
                fixed=int((hit&~good&m).sum()),harmed=int((~hit&good&m).sum())) for s,m in masks.items()})
            history.setdefault(k,[]).append(hit)
        print(seed,{k:round(v["accuracy"],4) for k,v in record[seed].items()},flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in masks.items()} for k,v in history.items()}
    write(REPORT/"results.json",dict(scopes={s:int(m.sum()) for s,m in masks.items()},representation=diagnostic,
                                      plans=record,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
