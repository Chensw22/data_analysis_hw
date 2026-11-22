"""Use independent bookings in the same decision-tree leaves to evaluate proposed reversals."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS, panel, predictions
from broad_booking_policy_n23 import EXTRA
from broad_matched_companions_n23 import fit
from broad_shared_passenger_modes_n23 import expanded_design, peer_pairs, axes
from exaone_validation_data import compact_features
from relational_error_corrector import write

OUT=ART/"nov23_broad_dispute_leaves"
REPORT=ROOT/"reports/nov23_broad_dispute_leaves"


def tree_leaves(model,x):
    leaf_ids=[]
    raw=np.full(len(x),model._baseline_prediction.item())
    for (tree,) in model._predictors:
        nodes=tree.nodes
        assert not nodes["is_categorical"].any()
        ids=np.zeros(len(x),dtype=int)
        active=np.flatnonzero(~nodes["is_leaf"][ids].astype(bool))
        while len(active):
            node=nodes[ids[active]]
            value=x[active,node["feature_idx"]]
            left=np.where(np.isnan(value),node["missing_go_to_left"],value<=node["num_threshold"])
            ids[active]=np.where(left,node["left"],node["right"])
            active=active[~nodes["is_leaf"][ids[active]].astype(bool)]
        raw+=nodes["value"][ids]
        leaf_ids.append(ids)
    np.testing.assert_allclose(raw,model.decision_function(x),rtol=1e-9,atol=1e-9)
    return np.column_stack(leaf_ids)


def cohorts(row,y):
    original,labels,base,old,_,_,_=panel()
    np.testing.assert_array_equal(original,row)
    np.testing.assert_array_equal(labels,y)
    rest=np.stack([predictions(row,y,seed,EXTRA)[1] for seed in SEEDS])
    older=(np.concatenate([old,rest],axis=2)>=.5)==y[None,:,None]
    repeat=((base>=.5)!=y).all(0)
    new=np.zeros(len(y),bool)
    for source,key in [("broad_shared_passenger_modes","real_only"),("broad_quantile_forest","standalone")]:
        stable=np.ones(len(y),bool)
        for seed in SEEDS:
            with np.load(ART/f"nov23_{source}/{seed}.npz") as z:
                np.testing.assert_array_equal(z["row"],row)
                np.testing.assert_array_equal(z["y"],y)
                stable &= (z[key]>=.5)==y
        new |= stable
    return {"ever":((base>=.5)!=y).any(0),"repeat":repeat,"new50":new&repeat&~older.all(0).any(1),
            "never212":repeat&~older.any((0,2))}


def matrices(frame,groups,size,row,t,q,y,cache):
    held=np.isin(groups,groups[row[q]])
    unsup=np.flatnonzero(~held)
    phi,z=expanded_design(frame,row[t],unsup)
    a,b,_,w=peer_pairs(groups,unsup)
    own=np.column_stack([phi[row],np.log(size[row])])
    shared=z[row]@axes(z,a,b,w)
    prob=[]
    leaves=[]
    for name,x in [("own",own),("real_only",shared)]:
        model=fit(x[t],y[t])
        p=model.predict_proba(x[q])[:,1]
        np.testing.assert_allclose(p,cache[name][q],rtol=1e-8,atol=1e-9)
        prob.append(p)
        ids=tree_leaves(model,x[q])
        leaves.append(OneHotEncoder().fit_transform(ids)/np.sqrt(ids.shape[1]))
    return np.column_stack(prob),leaves


def run():
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ("Name","Transported"))
                   for s in ("train","test")],ignore_index=True)
    groups=raw.PassengerId.str.split("_").str[0].to_numpy()
    size=pd.Series(groups).groupby(groups).transform("size").to_numpy()
    frame=compact_features(raw)
    OUT.mkdir(exist_ok=True)
    reference=[]
    success={}
    rankings={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_broad_shared_passenger_modes/{seed}.npz") as z:
            cache={k:z[k] for k in ("row","y","fold","base","own","real_only")}
        row,y,fold,base=[cache[k] for k in ("row","y","fold","base")]
        before=(base>=.5)==y
        reference.append(before)
        dispute=(base>=.5)!=(cache["real_only"]>=.5)
        out={}
        matched={}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(groups[row[t]],groups[row[q]]).size
            p,leaves=matrices(frame,groups,size,row,t,q,y,cache)
            logits=logit(np.column_stack([base[q],p]).clip(.002,.998))
            cols=sparse.csr_matrix(logits)
            models={"score":cols,"own_leaves":sparse.hstack([cols,leaves[0]]).tocsr(),
                    "both_leaves":sparse.hstack([cols,*leaves]).tocsr()}
            # Leaf IDs and expert scores are held-out from a single first-stage
            # fit. Only distinct bookings within that fold may calibrate them.
            for a,b in GroupKFold(3,shuffle=True,random_state=479).split(q,groups=groups[row[q]]):
                assert not np.intersect1d(groups[row[q[a]]],groups[row[q[b]]]).size
                aa=a[dispute[q[a]]]
                for key,x in models.items():
                    reg=LogisticRegression(C=2,max_iter=500).fit(x[aa],y[q[aa]])
                    p1=reg.predict_proba(x[b])[:,1]
                    out.setdefault(key,np.full(len(y),np.nan))[q[b]]=p1
                    benchmark=matched.setdefault(key,base>=.5)
                    for direction in (True,False):
                        take=dispute[q[b]] & ((base[q[b]]>=.5)==direction)
                        n=int((take&((p1>=.5)!=direction)).sum())
                        options=q[b[take]]
                        ix=np.argsort(abs(base[options]-.5),kind="stable")[:n]
                        benchmark[options[ix]]=not direction
        turns={}
        for key,p in out.items():
            assert np.isfinite(p).all()
            label=np.where(dispute,p>=.5,base>=.5)
            success.setdefault(key,[]).append(label==y)
            success.setdefault(key+"_matched",[]).append(matched[key]==y)
            flip_risk=np.where(base>=.5,1-p,p)
            turns[key]=dict(accuracy_on_disputes=float((label[dispute]==y[dispute]).mean()),
                            error_auc=float(roc_auc_score(~before[dispute],flip_risk[dispute])),
                            error_logloss=float(log_loss(~before[dispute],flip_risk[dispute])))
        rankings[seed]=dict(rows=int(dispute.sum()),actual_base_errors=int((dispute&~before).sum()),turns=turns)
        np.savez_compressed(OUT/f"{seed}.npz",row=row,y=y,fold=fold,base=base,dispute=dispute,**out)
        print(seed,rankings[seed],flush=True)
    masks=cohorts(row,y)
    assert masks["new50"].sum()==50
    ref=np.stack(reference)
    metrics={}
    for key,v in success.items():
        hit=np.stack(v)
        metrics[key]=dict(accuracy=hit.mean(1).tolist(),fixed=(hit&~ref).sum(1).tolist(),
                          harmed=(~hit&ref).sum(1).tolist(),
                          stable={k:int((hit.all(0)&m).sum()) for k,m in masks.items()})
    write(REPORT/"result.json",dict(metrics=metrics,disputes=rankings))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
