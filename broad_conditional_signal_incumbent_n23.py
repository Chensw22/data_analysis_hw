"""Re-evaluate field-weight and contrastive likelihood corrections on saved v131 decisions."""
import numpy as np
import pandas as pd
from scipy.special import logit
from threadpoolctl import threadpool_limits
from broad_conditional_label_ratio_n23 import ART,ROOT,REPORT,fold_scores
from broad_honest_field_weights_n23 import blocks,evaluate
from broad_dispute_leaf_evidence_n23 import cohorts
from exaone_validation_data import compact_features
from relational_error_corrector import write


def run():
    with np.load(ART/"nov23_incumbent_budget_transfer/oof.npz") as z:
        row,y,fold,inc,base,unseen=[z[k] for k in ("row","y","fold","v131","v108","unseen")]
    hist=pd.read_csv(ART/"nov21_case_checks/v136_final_oof.csv")
    for k,v in [("row",row),("true",y),("fold",fold),("v131",inc)]:
        np.testing.assert_array_equal(hist[k],v)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ("Name","Transported"))
                   for s in ("train","test")],ignore_index=True)
    np.testing.assert_array_equal(raw.PassengerId.iloc[row],hist.PassengerId)
    parts,group=blocks(raw,row)
    frame=compact_features(raw.iloc[row])
    booking=raw.PassengerId.str.split("_").str[0]
    frame["BookingSize"]=booking.map(booking.value_counts()).to_numpy()[row]
    pred,_=evaluate(logit(base[:,None].clip(.002,.998)),parts,y,group,fold)
    for k in np.unique(fold):
        q=np.flatnonzero(fold==k)
        t=np.flatnonzero((fold!=k)&~np.isin(group,group[q]))
        assert not np.intersect1d(group[t],group[q]).size
        np.testing.assert_array_equal(unseen[q],~np.isin(group[q],group[fold!=k]))
        v=fold_scores(frame,y,group,t,q,base,20261112+int(k))
        for n in ("transfer","shuffled_transfer"):
            pred.setdefault(n,np.full(len(y),np.nan))[q]=v[n]
    override=inc!=(base>=.5)
    before=inc==y
    masks=cohorts(row,y)
    result={}
    for n,p in pred.items():
        assert np.isfinite(p).all()
        hit=np.where(override,inc,p>=.5)==y
        result[n]={}
        for key,m in {"all":np.ones(len(y),bool),"unseen":unseen,**masks}.items():
            good=m&hit&~before
            bad=m&~hit&before
            result[n][key]=dict(accuracy=float(hit[m].mean()),fixed=int(good.sum()),harmed=int(bad.sum()),
                              net_by_fold=[int(good[fold==k].sum()-bad[fold==k].sum()) for k in np.unique(fold)])
    write(REPORT/"incumbent.json",dict(reference=float(before.mean()),overrides=int(override.sum()),results=result))
    print({n:v["all"] for n,v in result.items()},flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
