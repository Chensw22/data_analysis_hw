"""Marginalize all joint outcomes for each booking, with honest calibration and unknown peer targets."""
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, logit, softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from grouped_five_way_coverage_n23 import verify
from nested_manifest_expressions_n23 import BILLS, fit
from exaone_validation_data import compact_features
from population_rescue_map_n23 import ART, ROOT, SEEDS, read_panel
from relational_error_corrector import write


def structure(raw, groups, pool, scores, target):
    blocks=[]
    observed=raw.Cabin.fillna("").to_numpy()
    sleep=raw.CryoSleep.fillna("unknown").to_numpy()
    paid=raw[BILLS].gt(0).any(axis=1).to_numpy()
    keys=groups.iloc[pool]
    sizes=keys.value_counts()
    for size, ids_by_size in sizes.groupby(sizes):
        if size==1:
            continue
        members=keys[keys.isin(ids_by_size.index)].groupby(keys).groups
        ids=np.array([np.asarray(v) for v in members.values()])
        states=np.array(list(product([-1.,1.],repeat=size)))
        spin=(states+1)/2
        even=states[:,None,:]*states[:,:,None]
        left,right=np.triu_indices(size,k=1)
        i,j=ids[:,left],ids[:,right]
        edge=np.stack([np.ones_like(i), (observed[i]==observed[j])&(observed[i]!=""),
                       (sleep[i]==sleep[j])&(sleep[i]!="unknown"), (paid[i]==paid[j])],axis=2)/(size-1)
        pair=np.einsum("bpf,sp->bsf",edge,even[:,left,right])
        prior=(scores[ids]-.5)*2
        peer=(prior.sum(axis=1)[:,None]-prior)/(size-1)
        unary=(scores[ids].clip(.005,.995))
        logodds=logit(unary)
        base=(logodds@states.T)*.5
        control=np.einsum("bn,sn->bs",peer,states)[:,:,None]/2
        x=np.concatenate([control,pair],axis=2)
        # Observed labels from calibration bookings alone enter the optimizer.
        # For bookings with partially unlabelled members sum over every compatible state.
        labels=target[ids]
        compatible=((labels[:,None,:]<0)|(labels[:,None,:]==spin[None,:,:])).all(axis=2)
        blocks.append((ids,spin,base,x,compatible,groups.to_numpy()[ids[:,0]]))
    return blocks


def calibrate(blocks, fit_groups, query_groups, width, output):
    selected=[]
    for _,_,base,x,compatible,keys in blocks:
        mask=np.isin(keys,fit_groups)
        if mask.any():
            selected.append((base[mask],x[mask,:,:width],compatible[mask]))
    def objective(w):
        loss=.5*12*(w@w)
        grad=12*w.copy()
        for offset,x,match in selected:
            score=offset+x@w
            observed=np.where(match,score,-np.inf)
            loss+=(logsumexp(score,axis=1)-logsumexp(observed,axis=1)).sum()
            grad+=np.einsum("bs,bsf->f",softmax(score,axis=1)-softmax(observed,axis=1),x)
        return loss,grad
    res=minimize(objective,np.zeros(width),jac=True,method="L-BFGS-B",options=dict(ftol=1e-12,maxiter=150))
    if not res.success:
        raise RuntimeError(res.message)
    for ids,spin,base,x,_,keys in blocks:
        mask=np.isin(keys,query_groups)
        if mask.any():
            joint=softmax(base[mask]+x[mask,:,:width]@res.x,axis=1)
            output[ids[mask]] = joint@spin
    return res.x.tolist()


def run():
    inventory=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    row=inventory.row.to_numpy()
    y,hits,others=read_panel()
    hard=~hits.any(0)&~others.any(axis=(0,1))
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ["Name","Transported"])
    x=compact_features(raw)
    groups=raw.PassengerId.str.split("_").str[0]
    known=np.full(len(raw),-1)
    known[row]=y
    group=groups.iloc[row].to_numpy()
    out=ART/"nov23_exact_booking_field"
    out.mkdir(exist_ok=True)
    results={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,row,y,fold)
        preds={name:base.copy() for name in ["peer_only","joint"]}
        weights=[]
        background=set()
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            qgroups=np.unique(group[q])
            assert not np.intersect1d(group[t],qgroups).size
            pool=np.flatnonzero(groups.isin(qgroups))
            opaque=np.setdiff1d(pool,row[q])
            assert (known[opaque]<0).all()
            scores=np.full(len(raw),np.nan)
            scores[row[q]]=base[q]
            if len(opaque):
                teacher=fit(x,known,row[t],155)
                scores[opaque]=teacher.predict_proba(x.iloc[opaque])[:,1]
                background.update(opaque)
            assert np.isfinite(scores[pool]).all()
            blocks=structure(raw,groups,pool,scores,known)
            # At zero coupling exact marginals must reproduce each original probability.
            for ids,spin,offset,_,_,_ in blocks:
                np.testing.assert_allclose(softmax(offset,axis=1)@spin,scores[ids].clip(.005,.995),atol=1e-12)
            for a,b in GroupKFold(3,shuffle=True,random_state=117).split(q,groups=group[q]):
                fit_groups=np.unique(group[q[a]])
                test_groups=np.unique(group[q[b]])
                assert not np.intersect1d(fit_groups,test_groups).size
                for name,w in [("peer_only",1),("joint",5)]:
                    corrected=scores.copy()
                    weights.append(dict(fold=int(k),method=name,params=calibrate(blocks,fit_groups,test_groups,w,corrected)))
                    preds[name][q[b]]=corrected[row[q[b]]]
        old=(base>=.5)==y
        metrics={}
        for name,p in preds.items():
            assert np.isfinite(p).all()
            ok=(p>=.5)==y
            metrics[name]=dict(accuracy=float(ok.mean()),fixed=int((ok&~old).sum()),harmed=int((~ok&old).sum()),
                               persistent1070=int(ok[~hits.any(0)].sum()),unresolved466=int(ok[hard].sum()))
        results[seed]=dict(metrics=metrics,unlabelled_peers=len(background),calibrations=weights)
        np.savez_compressed(out/f"{seed}.npz",row=row,fold=fold,y=y,base=base,**preds)
        print(seed,metrics,flush=True)
    write(ROOT/"reports/nov23_exact_booking_field/result.json",dict(results=results,
          definition="Exact binary configuration marginals for each complete booking; singleton probabilities untouched. Peer-share logit pooling is the matched control. Unknown labels are marginalized, never imputed."))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
