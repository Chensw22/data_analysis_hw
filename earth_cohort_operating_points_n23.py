"""Estimate how much outcome ranking and decision thresholds can recover in two broad cohorts."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from archive_error_recovery_n23 import ART, ROOT, SEEDS, panel
from population_error_accounting_n23 import BILLS
from relational_error_corrector import write


def best_threshold(score, y):
    # Sweep entire tied blocks in descending score order; all-negative and
    # all-positive are legitimate operating points. Prefer cutoffs near .5 on ties.
    order=np.argsort(-score,kind="stable")
    vals=score[order]
    ends=np.r_[np.flatnonzero(vals[1:]!=vals[:-1])+1,len(vals)]
    totals=(~y).sum()+np.r_[0,np.cumsum(2*y[order].astype(int)-1)[ends-1]]
    cut=np.r_[np.nextafter(vals[0],np.inf),(vals[ends[:-1]-1]+vals[ends[:-1]])/2, vals[-1]]
    ix=np.flatnonzero(totals==totals.max())
    i=ix[np.argmin(abs(cut[ix]-.5))]
    assert ((score>=cut[i])==y).sum()==totals[i]
    return int(totals[i]),float(cut[i])


def select(members,labels,t,allowed):
    choices=[best_threshold(members[t,j],labels[t]) for j in allowed]
    ix=int(np.argmax([n for n,_ in choices]))
    return allowed[ix],choices[ix][1]


def run():
    row,y,base,bank,_,names,_=panel()
    names=["base",*names]
    safe=[names.index(k) for k in ["base","registration_only","budget_prior","feature_projections",
          "field_factors","bayesian_forest","tensor_train","passenger_relation","manifest_likelihood","boarding_sequence"]]
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId","HomePlanet","CryoSleep",*BILLS]).iloc[row]
    group=raw.PassengerId.str.split("_").str[0].to_numpy()
    count=raw[BILLS].gt(0).sum(axis=1)
    earth=raw.HomePlanet.eq("Earth")
    scopes={"sleep_zero":(earth&raw.CryoSleep.eq(True)&count.eq(0)).to_numpy(),
            "awake_multiple":(earth&raw.CryoSleep.eq(False)&count.ge(2)).to_numpy()}
    out=ART/"nov23_earth_operating_points"
    out.mkdir(exist_ok=True)
    result={}
    for i,seed in enumerate(SEEDS):
        members=np.column_stack([base[i],bank[i]])
        init=base[i]>=.5
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            fold=ref["fold"]
            np.testing.assert_array_equal(row,ref["row"])
            np.testing.assert_array_equal(y,ref["y"])
        deploy={n:init.copy() for n in ["threshold_only","selection"]}
        upper=init.copy()
        records={}
        for name,mask in scopes.items():
            ids=np.flatnonzero(mask)
            j,cut=select(members,y,ids,np.arange(len(names)))
            upper[ids]=members[ids,j]>=cut
            records[name]=dict(rows=len(ids),actual_positive=int(y[ids].sum()),reference_expected=float(base[i,ids].sum()),
                    reference_correct=int((init[ids]==y[ids]).sum()),majority_correct=int(max(y[ids].sum(),(~y[ids]).sum())),
                    reference_auc=float(roc_auc_score(y[ids],base[i,ids])), best_member=names[j], oracle_threshold=cut,
                    best_correct=int((upper[ids]==y[ids]).sum()),base_only_best_correct=best_threshold(base[i,ids],y[ids])[0])
            for f in np.unique(fold):
                held=ids[fold[ids]==f]
                for a,b in GroupKFold(3,shuffle=True,random_state=681).split(held,groups=group[held]):
                    t,q=held[a],held[b]
                    assert not np.intersect1d(group[t],group[q]).size
                    for kind,allowed in [("threshold_only",[0]),("selection",safe)]:
                        k,threshold=select(members,y,t,allowed)
                        deploy[kind][q]=members[q,k]>=threshold
            records[name]["heldout_correct"]={k:int((d[ids]==y[ids]).sum()) for k,d in deploy.items()}
        metrics={}
        for key,d in dict(posthoc_upper=upper,**deploy).items():
            ok=d==y
            ref=init==y
            metrics[key]=dict(correct=int(ok.sum()),accuracy=float(ok.mean()),fixes=int((ok&~ref).sum()),harms=int((~ok&ref).sum()))
        result[seed]=dict(cohorts=records,whole_roster=metrics)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base[i],**deploy)
        print(seed,records,metrics,flush=True)
    write(ROOT/"reports/nov23_earth_operating_points/result.json",dict(members=names,safe=[names[j] for j in safe],results=result))


if __name__=="__main__":
    run()
