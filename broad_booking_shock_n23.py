"""Estimate all-negative and all-positive booking shocks, integrating unseen outcomes."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp, softmax
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS, panel, predictions
from fresh_family_rescue_n23 import NEW


def prepare(frame, groups, base):
    b = logit(base.clip(.001,.999))
    x = np.column_stack((b, np.abs(b)))
    table = pd.DataFrame(dict(b=b, age=frame.Age.fillna(28), sleep=frame.CryoSleep.eq(True),
                              paid=frame[["RoomService","Spa","VRDeck","FoodCourt","ShoppingMall"]]
                                   .gt(0).any(axis=1).to_numpy()),index=frame.index)
    stats = table.groupby(groups).agg(["mean","std"]).fillna(0)
    stats[("size","log1p")] = np.log1p(table.groupby(groups).size())
    return x, stats.loc[groups].to_numpy()


def components(x, g, z, groups, y, params, joint):
    nvar = x.shape[1]
    coeff = params[nvar:].reshape(g.shape[1],2)
    _, first, group = np.unique(groups,return_index=True,return_inverse=True)
    if joint:
        loc = first
    else:
        loc = np.arange(len(group))
        group = loc
    raw = np.column_stack((g[loc] @ coeff - 3., np.zeros(len(loc))))
    lprior = raw - logsumexp(raw,axis=1,keepdims=True)
    logits = z + x @ params[:nvar]
    individual_logp = -np.logaddexp(0,(1-2*y)*logits)
    n = np.bincount(group)
    count = np.bincount(group,weights=y)
    ll = np.column_stack((np.where(count==0,0.,-np.inf),np.where(count==n,0.,-np.inf),
                          np.bincount(group,weights=individual_logp)))
    logweights = lprior + ll
    evidence = logsumexp(logweights,axis=1)
    posterior = softmax(logweights,axis=1)
    prior = np.exp(lprior)
    individual = expit(logits)
    grad1 = x.T @ ((individual-y)*posterior[group,2])
    grad2 = g[loc].T @ (prior[:,:2]-posterior[:,:2])
    prob = prior[group,1]+prior[group,2]*individual
    return evidence,np.r_[grad1,grad2.ravel()],prob,n


def run():
    row, y, base, old, _, _, _ = panel()
    extra = dict(**NEW,native_carte=("carte","p"))
    archive = [predictions(row,y,s,extra) for s in SEEDS]
    older = np.concatenate([old,np.stack([a[1] for a in archive])],axis=2)
    historical = (older>=.5)==y[None,:,None]
    correct_base = (base>=.5)==y
    repeat = ~correct_base.any(0)
    never = repeat & ~historical.any(axis=(0,2))
    raw = pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Transported","Name")).iloc[row]
    groups = raw.PassengerId.str.split("_").str[0].to_numpy()
    out = ART/"nov23_broad_booking_shock"
    out.mkdir(exist_ok=True)
    results,hits = {},{}
    for i,(seed,(ref,_,fold)) in enumerate(zip(SEEDS,archive)):
        np.testing.assert_array_equal(base[i],ref)
        z = logit(ref.clip(.001,.999))
        x0,g0 = prepare(raw,groups,ref)
        scores = {name:np.full(len(y),np.nan) for name in ("independent","joint")}
        losses = {name:[] for name in scores}
        for f in np.unique(fold):
            rows = np.flatnonzero(fold==f)
            assert not np.intersect1d(groups[rows],groups[fold!=f]).size
            for a,b in GroupKFold(3,shuffle=True,random_state=527).split(rows,groups=groups[rows]):
                t,q = rows[a],rows[b]
                assert not np.intersect1d(groups[t],groups[q]).size
                x = np.column_stack([np.ones(len(y)),StandardScaler().fit(x0[t]).transform(x0)])
                g = np.column_stack([np.ones(len(y)),StandardScaler().fit(g0[t]).transform(g0)])
                for name in scores:
                    joint = name=="joint"
                    def loss(params):
                        ll,grad,_,_ = components(x[t],g[t],z[t],groups[t],y[t].astype(float),params,joint)
                        return -ll.sum()+4*(params@params),grad+8*params
                    sol=minimize(loss,np.zeros(x.shape[1]+2*g.shape[1]),jac=True,method="L-BFGS-B",
                                 options=dict(maxiter=250,ftol=1.e-11))
                    assert sol.success, sol.message
                    # Use arbitrary zeros during inference. Marginals only use the prior.
                    _,_,p,_=components(x[q],g[q],z[q],groups[q],np.zeros(len(q)),sol.x,True)
                    ll,_,check,n=components(x[q],g[q],z[q],groups[q],y[q].astype(float),sol.x,True)
                    np.testing.assert_array_equal(p,check)
                    scores[name][q]=p
                    losses[name].extend((-ll[n>1]).tolist())
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=ref,**scores)
        results[seed]={}
        for name,p in scores.items():
            assert np.isfinite(p).all()
            ok=(p>=.5)==y
            hits.setdefault(name,[]).append(ok)
            results[seed][name]=dict(accuracy=ok.mean(),non_singleton_joint_loss=np.mean(losses[name]))
            for other,correct in (("baseline",correct_base[i]),("independent",(scores["independent"]>=.5)==y)):
                results[seed][name][other]=[int((ok & ~correct).sum()),int((~ok & correct).sum())]
        print(seed,results[seed],flush=True)
    coverage={}
    for name,h in hits.items():
        a=np.asarray(h)
        coverage[name]=dict(ever_stable=int((a.all(0)&~correct_base.all(0)).sum()),
                            persistent_stable=int((a.all(0)&repeat).sum()),never_any=int((a.any(0)&never).sum()),
                            never_stable=int((a.all(0)&never).sum()))
    folder=ROOT/"reports/nov23_broad_booking_shock"
    folder.mkdir(exist_ok=True)
    (folder/"result.json").write_text(json.dumps(dict(results=results,coverage=coverage),indent=2)+"\n")


if __name__=="__main__":
    with threadpool_limits(2):
        run()
