"""Separate passenger and shared-booking response traits using the complete unlabeled manifest."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from broad_refit_dissent_n23 import features
from broad_training_membership_n23 import model
from broad_boosting_path_n23 import calibrated
from broad_bill_exception_n23 import ART, ROOT, SEEDS, load, historical, evaluate, write
from broad_geographic_transfer_n23 import BILLS
from exaone_validation_data import CATEGORICAL, compact_features

REPORT = ROOT / "reports/nov23_hidden_context"
OUT = ART / "nov23_hidden_context"


def codes(frame):
    fields = []
    for name in frame:
        if name == "BookingSize":
            continue
        col = frame[name]
        if name in CATEGORICAL:
            val = pd.factorize(col, sort=True)[0]
        else:
            x = col.to_numpy(float)
            ok = np.isfinite(x)
            informative = ok & (x > 0) if name in BILLS else ok
            cuts = np.unique(np.quantile(x[informative], [.25,.5,.75]))
            val = np.where(ok, 1, 0)
            val[informative] = 2 + np.searchsorted(cuts, x[informative])
        fields.append(np.unique(val,return_inverse=True)[1])
    return fields


def posterior(emissions, inv, prior):
    local = logsumexp(emissions + prior[None,None,:], axis=2)
    pooled = np.zeros((inv.max()+1,len(prior)))
    np.add.at(pooled, inv, local)
    group = softmax(pooled+prior, axis=1)
    members = softmax(emissions + prior[None,None,:], axis=2)
    weights = members * group[inv,:,None]
    return weights, np.sum(logsumexp(pooled + prior,axis=1))


def latent(fields, inv, steps=32):
    nodes, mass = np.polynomial.hermite.hermgauss(5)
    nodes *= np.sqrt(2)
    prior = np.log(mass / np.sqrt(np.pi))
    g, i = np.meshgrid(nodes, nodes, indexing="ij")
    coords = np.column_stack([np.ones(g.size), g.ravel(), i.ravel()])
    weights = []
    rng = np.random.default_rng(493)
    for v in fields:
        w = rng.normal(0,.35,size=(coords.shape[1],v.max()+1))
        w[0] = np.log(np.bincount(v)/len(v))
        weights.append(w)
    losses = []
    for step in range(steps+1):
        logs = [coords@w - logsumexp(coords@w,axis=1,keepdims=True) for w in weights]
        parts = [p[:,v].T.reshape(len(v),5,5) for p,v in zip(logs,fields)]
        total = sum(parts)
        post, ll = posterior(total, inv, prior)
        losses.append(float(-ll))
        if step == steps:
            break
        for idx, v in enumerate(fields):
            counts = np.stack([np.bincount(v, weights=col) for col in post.reshape(len(v),-1).T])
            def cost(par):
                w = par.reshape(weights[idx].shape)
                z = coords@w
                lp = z-logsumexp(z,axis=1,keepdims=True)
                grad = coords.T @ (np.exp(lp)*counts.sum(1,keepdims=True)-counts) + .1*w
                return -(lp*counts).sum() + .05*(w*w).sum(), grad.ravel()
            opt = minimize(cost, weights[idx].ravel(), jac=True, method="L-BFGS-B",
                           options=dict(maxiter=500, ftol=1e-8))
            assert opt.success, opt.message
            weights[idx] = opt.x.reshape(weights[idx].shape)
    assert losses[-1] < losses[0]
    moments = np.column_stack([(post*b).sum((1,2)) for b in (g,i,g**2,i**2,g*i)])
    # The group posterior before observing this passenger supplies an honest
    # own-field conditional even for a solo booking (population prior).
    local = logsumexp(total+prior[None,None,:],axis=2)
    pooled = np.zeros((inv.max()+1,len(prior)))
    np.add.at(pooled,inv,local)
    context = pooled[inv]-local+prior
    predictive = []
    for part in parts:
        left = total-part + context[:,:,None] + prior[None,None,:]
        predictive.append(logsumexp(left+part,axis=(1,2))-logsumexp(left,axis=(1,2)))
    assert np.max(predictive) < 1e-8
    return np.column_stack([moments,*predictive]), losses


def run():
    row,y,frame,g,_ = load()
    roster = pd.concat([pd.read_csv(ROOT/f"data/raw/{split}.csv", usecols=lambda c:c not in ("Name","Transported"))
                        for split in ("train","test")],ignore_index=True)
    groups = roster.PassengerId.str.split("_").str[0].astype(int).to_numpy()
    raw = compact_features(roster)
    np.testing.assert_array_equal(groups[row],g)
    obs,labels,masks = historical()
    np.testing.assert_array_equal(obs,row)
    np.testing.assert_array_equal(labels,y)
    fields = codes(raw)
    _,inv = np.unique(groups,return_inverse=True)
    shuffled = inv.copy()
    sizes = np.bincount(inv)[inv]
    rng = np.random.default_rng(577)
    for n in np.unique(sizes):
        idx = np.flatnonzero(sizes==n)
        shuffled[idx] = rng.permutation(inv[idx])
    x = features(frame)[0]
    designs = dict(raw=x)
    losses = {}
    for k,gi in (("shared",inv),("random_groups",shuffled)):
        latent_x, losses[k] = latent(fields,gi)
        designs[k] = np.c_[x,latent_x[row]]
    report, history = {},{}
    OUT.mkdir(exist_ok=True)
    for s in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{s}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            base,fold=z["base"],z["fold"]
        ref=base>=.5
        scores={}
        for k,design in designs.items():
            pred=np.full(len(y),np.nan)
            for f in np.unique(fold):
                t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
                assert not np.intersect1d(g[t],g[q]).size
                fit=model(210).fit(design[t],y[t])
                pred[q]=fit.predict_proba(design[q])[:,1]
            scores[k]=pred
            joint=np.column_stack([logit(base.clip(.001,.999)),logit(pred.clip(.001,.999))])
            p=np.full(len(y),np.nan)
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    p[q]=calibrated(joint[t],joint[q],y,t)
            scores["combined_"+k]=p
        report[str(s)],hits=evaluate(y,base,ref,scores,fold,False,masks)
        for k,(h,_) in hits.items():
            history.setdefault(k,[]).append(h)
        np.savez_compressed(OUT/f"{s}.npz",row=row,y=y,fold=fold,base=base,**scores)
        print(s,report[str(s)]["combined_shared"]["accuracy"],flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in masks.items()} for k,v in history.items()}
    write(REPORT/"results.json",dict(negative_log_likelihood=losses,metrics=report,stable=stable))
    intervals(history,g,masks)


def intervals(history,g,masks):
    # Resample complete bookings with the three paired predictions attached.
    # Intervals describe variation in the scoring roster conditional on these fits.
    rng=np.random.default_rng(823)
    inv=np.unique(g,return_inverse=True)[1]
    summaries={}
    for suffix in ("shared","combined_shared"):
        for control in ("raw","random_groups"):
            b=suffix.replace("shared",control)
            delta=np.asarray(history[suffix],float).mean(0)-np.asarray(history[b],float).mean(0)
            record={}
            for key,scope in {"all":np.ones(len(g),bool),"ever":masks["ever"],"repeat":masks["repeat"]}.items():
                n=np.bincount(inv,weights=scope)
                s=np.bincount(inv,weights=scope*delta)
                ids=np.flatnonzero(n)
                draws=rng.choice(ids,size=(1200,len(ids)))
                sample=s[draws].sum(1)/n[draws].sum(1)
                record[key]=[float(s.sum()/n.sum()),*np.quantile(sample,[.025,.975]).tolist()]
            summaries[f"{suffix}_vs_{b}"]=record
    write(REPORT/"intervals.json",summaries)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
