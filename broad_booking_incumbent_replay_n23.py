"""Refit the complementary trio for the original v131 outer partitions."""
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from awake_negative_recovery import BILLS, matrices
from broad_booking_policy_n23 import ART, OUT, ROOT, SEEDS
from conditional_view_density_n23 import fit_score
from exaone_validation_data import CATEGORICAL, compact_features
from grouped_field_factors_n23 import encode, fit as fit_factors
from projection_shapes_n23 import fit as fit_projections
from relational_error_corrector import write


def run():
    with np.load(ART/"nov23_incumbent_budget_transfer/oof.npz") as z:
        row,y,fold,incumbent,prob,unseen = [z[n] for n in ("row","y","fold","v131","v108","unseen")]
    inv = pd.read_csv(ART/"nov21_case_checks/v136_final_oof.csv")
    for n,v in [("row",row),("true",y),("fold",fold),("v131",incumbent)]:
        np.testing.assert_array_equal(inv[n],v)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ("Transported","Name"))
                   for s in ("train","test")],ignore_index=True)
    group=raw.PassengerId.str.split("_").str[0]
    size=group.map(group.value_counts()).to_numpy()[row]
    groups=group.iloc[row].to_numpy()
    np.testing.assert_array_equal(raw.PassengerId.iloc[row],inv.PassengerId)
    frame=compact_features(raw.iloc[row])
    bills=frame[BILLS].fillna(0)
    total=bills.sum(1).to_numpy()
    xp=np.column_stack([matrices(frame)[0],frame.CabinNumber.fillna(900)/1000,frame.CabinNumber.isna(),
                        np.log(size),np.log1p(total)/6,bills.div(1+total,axis=0)])
    masks=[np.ones(len(y),bool),total>0,total==0]
    factor=encode(raw)[row]
    target=torch.tensor(y,dtype=torch.float32)
    reg=frame[CATEGORICAL].apply(lambda c: pd.Categorical(c).codes).copy()
    reg["AgeBand"]=np.where(frame.Age.isna(),7,np.digitize(frame.Age,[6,13,18,25,40,60]))
    reg=reg.to_numpy(float)
    src=np.column_stack([prob,np.full((len(y),3),np.nan)])
    decisions={n:np.full(len(y),np.nan) for n in ("calibrate_base","combine_trio")}
    for k in np.unique(fold):
        q=np.flatnonzero(fold==k)
        t=np.flatnonzero((fold!=k)&~np.isin(groups,groups[q]))
        assert not np.intersect1d(groups[t],groups[q]).size
        np.testing.assert_array_equal(unseen[q],~np.isin(groups[q],groups[fold!=k]))
        src[q,1]=fit_projections(xp,y,masks,t,q)["projection_shapes"]
        src[q,2]=expit(fit_score(reg[t],y[t],reg[q],np.ones(reg.shape[1],bool)))
        src[q,3]=fit_factors(factor,target,t,q,3)
        x=logit(src[q].clip(.001,.999))
        for a,b in GroupKFold(3,shuffle=True,random_state=513).split(q,groups=groups[q]):
            assert not np.intersect1d(groups[q[a]],groups[q[b]]).size
            for key,n in (("calibrate_base",1),("combine_trio",4)):
                model=LogisticRegression(C=.1,max_iter=400).fit(x[a,:n],y[q[a]])
                decisions[key][q[b]]=model.predict_proba(x[b,:n])[:,1]
        print(f"incumbent replay: fold {k} complete",flush=True)
    repeat=np.ones(len(y),bool)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as prior:
            np.testing.assert_array_equal(row,prior["row"])
            np.testing.assert_array_equal(y,prior["y"])
            repeat &= (prior["base"]>=.5)!=y
    override=incumbent!=(prob>=.5)
    result={}
    for key,p in decisions.items():
        assert np.isfinite(p).all()
        hit=np.where(override,incumbent,p>=.5)==y
        previous=incumbent==y
        result[key]={}
        for name,mask in {"all":np.ones(len(y),bool),"previously_unseen_bookings":unseen,
                          "grouped_reference_persistent":repeat}.items():
            fixed=hit&~previous&mask
            harmed=~hit&previous&mask
            result[key][name]=dict(rows=int(mask.sum()),accuracy=float(hit[mask].mean()),
                                   fixed=int(fixed.sum()),harmed=int(harmed.sum()),
                                   by_fold_net=[int(fixed[fold==k].sum()-harmed[fold==k].sum()) for k in np.unique(fold)])
    dest=ART/"nov23_broad_booking_policy"
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/"incumbent.npz",row=row,fold=fold,y=y,incumbent=incumbent,
                        overrides=override,unseen=unseen,source_probs=src,**decisions)
    write(OUT/"incumbent.json",dict(reference=float((incumbent==y).mean()),overrides=int(override.sum()),results=result))
    print(result,flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
