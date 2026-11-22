"""Randomly reassign complete multi-seed reversal patterns to equally confident passengers."""
import numpy as np
import pandas as pd
from archive_error_recovery_n23 import ART,ROOT,SEEDS,panel,predictions
from broad_booking_policy_n23 import EXTRA
from relational_error_corrector import write

SOURCES={"shared":("broad_shared_passenger_modes","real_only"),
         "bills":("broad_quantile_forest","standalone"),
         "rotation":("broad_rotation_forest","rotation"),
         "mondrian":("broad_mondrian_partition","mondrian"),
         "margin_transfer":("broad_train_error_margins","margin_transfer"),
         "robust":("broad_train_error_margins","robust"),
         "robust_transfer":("broad_train_error_margins","robust_transfer"),
         "audit_errors_only":("broad_hypothetical_labels","direct_error_audits"),
         "audit_calibrated":("broad_hypothetical_labels","error_aware")}


def run():
    row,y,base,old,_,_,_=panel()
    extra=[predictions(row,y,s,EXTRA) for s in SEEDS]
    candidates=np.stack([read(row,y,extra,i,folder,key) for folder,key in SOURCES.values()
                         for i in range(len(SEEDS))]).reshape(len(SOURCES),len(SEEDS),len(y))
    archive=np.concatenate([old,np.stack([p for _,p,_ in extra])],axis=2)
    archive_ok=(archive>=.5)==y[None,:,None]
    ref=base>=.5
    ref_ok=ref==y
    repeat=~ref_ok.any(0)
    new_pool=repeat&~archive_ok.all(0).any(1)
    never=repeat&~archive_ok.any((0,2))
    confidence=np.mean(np.maximum(base,1-base),axis=0)
    dissent=((archive>=.5)!=ref[:,:,None]).mean((0,2))
    codes=(ref*np.array([1,2,4])[:,None]).sum(0)
    # Exact zero is separated so historically unanimous predictions can only
    # exchange patterns with other unanimous rows in the stricter control.
    risk_bins=np.digitize(confidence,np.quantile(confidence,[.2,.4,.6,.8]))
    opinion_bins=np.digitize(dissent,[1e-9,.05,.2])
    masks=[repeat,new_pool,never,ref_ok.all(0)]
    assert [int(m.sum()) for m in masks[:3]]==[1070,567,212]
    # A label-blind representative per booking checks whether several related
    # passengers in a party account for the portfolio comparison.
    bookings=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.str.split("_").str[0].to_numpy()[row]
    order=np.random.default_rng(729).permutation(len(y))
    _,idx=np.unique(bookings[order],return_index=True)
    reps=np.zeros(len(y),bool)
    reps[order[idx]]=True
    names=list(SOURCES)+["union_four"]
    columns=["stable_correct_on_repeat","new_stable","never212_to_stable","stable_harmed_on_always_correct",
             "mean_net_vs_reference","new_stable_one_per_booking"]
    def counts(bank):
        ok=bank==y
        stable=ok.all(1)
        lost=~ok.any(1)
        score=np.column_stack([(stable&m).sum(1) for m in masks[:3]]+
                   [(lost&masks[3]).sum(1), (ok.mean(1)-ref_ok.mean(0)).sum(1), (stable&(reps&new_pool)).sum(1)])
        union=stable[:4].any(0)
        # Only the three coverage counts are meaningful for an oracle union.
        return np.vstack([score,[*(int((union&m).sum()) for m in masks[:3]),0,0,int((union&reps&new_pool).sum())]])
    actual=counts(candidates)
    result={}
    rows=np.arange(len(y))
    for control,block in [("confidence",codes*5+risk_bins),("confidence_and_archive",codes*20+risk_bins*4+opinion_bins),
                          ("one_per_booking",2*(codes*20+risk_bins*4+opinion_bins)+reps.astype(int))]:
        parts=[np.flatnonzero(block==v) for v in np.unique(block)]
        rng=np.random.default_rng(841)
        draws=[]
        for _ in range(800):
            shuffle=rows.copy()
            for p in parts:
                shuffle[p]=rng.permutation(p)
            # The same permutation applies to every model and every seed. Since
            # reference votes are identical within each cell, amounts and sides
            # of overrides, and all agreements between methods, are preserved.
            draws.append(counts(candidates[:,:,shuffle]))
        sh=np.stack(draws)
        records={}
        for j,name in enumerate(names):
            rec={}
            for c,label in enumerate(columns):
                if name=="union_four" and c in (3,4):
                    continue
                vals=sh[:,j,c]
                rec[label]=dict(observed=float(actual[j,c]),null_mean=float(vals.mean()),
                                null_interval=np.quantile(vals,[.025,.975]).tolist())
            records[name]=rec
        result[control]=dict(rows=len(row),representative_rows=int(reps.sum()),representative_novel_pool=int((reps&new_pool).sum()),blocks=len(parts),permutations=800,methods=records,
                             singleton_cells=sum(len(p)==1 for p in parts))
    write(ROOT/"reports/nov23_broad_error_placement/result.json",result)
    print(result["confidence_and_archive"]["methods"]["union_four"],flush=True)


def read(row,y,extra,i,folder,key):
    with np.load(ART/f"nov23_{folder}/{SEEDS[i]}.npz") as z:
        for field,value in (("row",row),("y",y),("fold",extra[i][2])):
            np.testing.assert_array_equal(z[field],value)
        p=z[key]
        assert np.isfinite(p).all() and p.shape==y.shape
        return p>=.5


if __name__=="__main__":
    run()
