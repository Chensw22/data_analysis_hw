"""Audit both possible outcomes by refitting a ridge learner and scoring it on separate bookings."""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_train_error_margins_n23 import ART,ROOT,SEEDS,features
from broad_dispute_leaf_evidence_n23 import cohorts
from exaone_validation_data import compact_features
from relational_error_corrector import write

OUT=ART/"nov23_broad_hypothetical_labels"
REPORT=ROOT/"reports/nov23_broad_hypothetical_labels"


def audits(x,y,a,b,q,seed):
    gram=x[a].T@x[a]+12*np.eye(x.shape[1])
    gram[0,0]-=11.99
    target=x[a].T@y[a].astype(float)
    solve=cho_factor(gram)
    beta=cho_solve(solve,target)
    prior=x[q]@beta
    check=x[b]@beta
    influence=cho_solve(solve,x[q].T)
    leverage=np.einsum("ij,ji->i",x[q],influence)
    update=x[b]@influence/(1+leverage)
    # Verify a full hypothetical refit numerically in both directions.
    v=x[q[0]]
    for label in (0,1):
        exact=cho_solve(cho_factor(gram+np.outer(v,v)),target+label*v)
        np.testing.assert_allclose(x[b]@exact,check+update[:,0]*(label-prior[0]),atol=1e-8)
    err=y[b].astype(float)-check
    hard=(check>=.5)!=y[b]
    shuffled=hard.copy()
    rng=np.random.default_rng(seed)
    for bin_id in np.array_split(np.argsort(check,kind="stable"),12):
        shuffled[bin_id]=rng.permutation(hard[bin_id])
    def signal(mask):
        # Exactly SSE(label=0) - SSE(label=1) on the specified audit records.
        k=update[mask]
        energy=np.square(k).sum(0)
        diff=2*(err[mask]@k)+(2*prior-1)*energy
        return np.column_stack([diff/np.sqrt(energy+1e-8),np.log1p(energy)])
    return prior, signal(np.ones(len(b),bool)),np.column_stack([signal(hard),signal(~hard)]),np.column_stack([signal(shuffled),signal(~shuffled)])


def run():
    frame0=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Transported","Name"))
    group=frame0.PassengerId.str.split("_").str[0]
    OUT.mkdir(exist_ok=True)
    hits={}
    refs=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            row,y,base,fold=[z[n] for n in ("row","y","base","fold")]
        gs=group.to_numpy()[row]
        size=group.map(group.value_counts()).to_numpy()[row]
        frame=compact_features(frame0.iloc[row])
        out={}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(gs[t],gs[q]).size
            # Only feature scaling and feature-map construction use the complete
            # outer training bookings. Neither needs their labels.
            _,xt,xq=features(frame,size,t,q)
            x=np.empty((len(y),xt.shape[1]))
            x[t],x[q]=xt,xq
            sides=[]
            for a,b in GroupKFold(2,shuffle=True,random_state=119).split(t,groups=gs[t]):
                assert not np.intersect1d(gs[t[a]],gs[t[b]]).size
                sides.append(audits(x,y,t[a],t[b],q,143+int(k)))
            ridge,signal,by_error,placebo=[np.mean([s[j] for s in sides],axis=0) for j in range(4)]
            # These sign decisions use no outcome from the outer evaluation fold.
            for name,values in (("direct_all_audits",signal[:,0]),("direct_error_audits",by_error[:,0]),
                                ("direct_shuffled_audits",placebo[:,0])):
                out.setdefault(name,np.full(len(y),np.nan))[q]=expit(values)
            b=logit(base[q].clip(.002,.998))
            designs={"reference":b[:,None],"ridge":np.column_stack([b,ridge]),
                     "all":np.column_stack([b,ridge,signal]),"error_aware":np.column_stack([b,ridge,signal,by_error]),
                     "placebo":np.column_stack([b,ridge,signal,placebo]),"signal_only":np.column_stack([ridge,signal,by_error])}
            for a,c in GroupKFold(3,shuffle=True,random_state=647).split(q,groups=gs[q]):
                assert not np.intersect1d(gs[q[a]],gs[q[c]]).size
                for n,vals in designs.items():
                    scale=StandardScaler().fit(vals[a])
                    model=LogisticRegression(C=.1,max_iter=300).fit(scale.transform(vals[a]),y[q[a]])
                    p=model.predict_proba(scale.transform(vals[c]))[:,1]
                    out.setdefault(n,np.full(len(y),np.nan))[q[c]]=p
        for n,p in out.items():
            assert np.isfinite(p).all()
            hits.setdefault(n,[]).append((p>=.5)==y)
        refs.append((base>=.5)==y)
        np.savez_compressed(OUT/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**out)
        print(seed,{n:round(float(((p>=.5)==y).mean()),4) for n,p in out.items()},flush=True)
    ref=np.stack(refs)
    masks=cohorts(row,y)
    result={}
    for n,v in hits.items():
        good=np.stack(v)
        result[n]=dict(accuracy=good.mean(1).tolist(),fixed=(good&~ref).sum(1).tolist(),
                       harmed=(~good&ref).sum(1).tolist(),stable={k:int((m&good.all(0)).sum()) for k,m in masks.items()})
    write(REPORT/"result.json",result)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
