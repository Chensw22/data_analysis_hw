"""Fit conditional registration effects to discordant outcomes of economically similar bookings."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from exaone_validation_data import compact_features
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART, ROOT, SEEDS, read_panel
from relational_error_corrector import write
from residual_registration_n23 import registration
from spend_simplex_likelihood_n23 import BILLS
from transported_subclasses_n23 import stats


def neighbors(spend, group, learn, ids, seed):
    rng=np.random.default_rng(seed)
    k=64
    output=[]
    bandwidth=[]
    for part in np.array_split(ids,int(np.ceil(len(ids)/160))):
        raw=cdist(spend[part],spend[learn],"sqeuclidean")
        raw[group[part,None]==group[learn][None,:]]=np.inf
        # Equal bills, especially the large zero-spend cohort, sample distinct
        # neighboring bookings' passengers independently for each target person.
        order=raw+rng.uniform(0,1e-10,size=raw.shape)
        ix=np.argpartition(order,k-1,axis=1)[:,:k]
        selected=np.take_along_axis(order,ix,axis=1)
        ix=np.take_along_axis(ix,np.argsort(selected,axis=1),axis=1)
        dist=np.take_along_axis(raw,ix,axis=1)
        output.append(learn[ix])
        bandwidth.append(np.exp(-dist/np.maximum(dist[:,-1:],1.e-6)))
    return np.vstack(output),np.vstack(bandwidth)


def rank_scores(x,y,train,near):
    # Each edge is selected without its outcome. Fit to differing outcomes,
    # symmetrically orienting each unique pair once with no intercept.
    left=np.repeat(train,12)
    right=near[:len(train),:12].ravel()
    edges=np.unique(np.sort(np.stack([left,right],axis=1),axis=1),axis=0)
    left,right=edges.T
    discordant=y[left]!=y[right]
    left,right=left[discordant],right[discordant]
    delta=x[left]-x[right]
    model=LogisticRegression(C=.035,fit_intercept=False,max_iter=200,solver="lbfgs")
    model.fit(delta,y[left])
    return model.decision_function(x),int(len(left))


def local_probability(utility, ids, near, weights, y):
    # Solve the intercept so the neighborhood's predicted positives match its
    # observed count. Two balanced pseudo-observations at the query utility
    # keep all-positive and all-negative neighborhoods finite.
    neighbor_scores=utility[near]-utility[ids,None]
    target=(weights*y[near]).sum(1)+1
    low=np.full(len(ids),-25.)
    high=-low
    for _ in range(36):
        mid=(low+high)/2
        count=(weights*expit(mid[:,None]+neighbor_scores)).sum(1)+2*expit(mid)
        low,high=np.where(count<target,mid,low),np.where(count<target,high,mid)
    return expit((low+high)/2)


def run():
    rows=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"})
    book=raw.PassengerId.str.split("_").str[0]
    solo=book.map(book.value_counts()).to_numpy()[rows]==1
    group=book.to_numpy()[rows]
    x=compact_features(raw.iloc[rows]).reset_index(drop=True)
    fields=registration(x)
    b=x[BILLS].to_numpy(float)
    positive=(b>0).sum(1)
    spend=np.column_stack([np.log1p(np.nan_to_num(b)),3*np.isnan(b),3*(b>0)])
    y,old,other=read_panel()
    persistent=~old.any(0)
    hard=persistent&~other.any(axis=(0,1))
    output=ART/"nov23_registration_rank"
    output.mkdir(exist_ok=True)
    result={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            base,fold=z["base"],z["fold"]
            verify(z,rows,y,fold)
        prob={k:np.full(len(y),np.nan) for k in ["economic_neighbors","ordinary_registration","matched_registration"]}
        numbers=[]
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(group[t],group[q]).size
            enc=OneHotEncoder(handle_unknown="ignore",min_frequency=18)
            enc.fit(fields.iloc[t])
            phi=enc.transform(fields)
            near,w=neighbors(spend,group,t,np.r_[t,q],seed+int(f))
            matched,edges=rank_scores(phi,y,t,near)
            numbers.append(edges)
            regular=LogisticRegression(C=.035,max_iter=250).fit(phi[t],y[t]).decision_function(phi)
            for name,s in [("economic_neighbors",np.zeros(len(y))),("ordinary_registration",regular),("matched_registration",matched)]:
                prob[name][q]=local_probability(s,q,near[len(t):],w[len(t):],y)
        summaries={}
        for name,p in prob.items():
            assert np.isfinite(p).all()
            for label,score in [(name,p),(name+"_blend",.75*base+.25*p)]:
                d=score>=.5
                m=stats(d,base>=.5,y,persistent,hard,solo)
                m["multiple_positive_bills_accuracy"]=float(np.mean(d[positive>=2]==y[positive>=2]))
                m["zero_or_missing_bills_accuracy"]=float(np.mean(d[positive==0]==y[positive==0]))
                summaries[label]=m
        np.savez_compressed(output/f"{seed}.npz",row=rows,y=y,fold=fold,base=base,**prob)
        result[seed]=dict(metrics=summaries,training_discordant_edges=numbers)
        write(ROOT/"reports/nov23_registration_rank/result.json",dict(results=result))
        print(seed,numbers,[(k,round(m["accuracy"]*100,3),m["fixed"],m["harmed"]) for k,m in summaries.items()],flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
