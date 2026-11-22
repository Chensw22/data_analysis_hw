"""Replay the fixed quarter-blend field policy against historical v131 decisions."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT
from conditional_field_value_n23 import field_blocks, fold_predictions
from relational_error_corrector import write


def run():
    with np.load(ART/"nov23_incumbent_budget_transfer/oof.npz") as z:
        row,y,fold,v131,v108,unseen=(z[k] for k in ("row","y","fold","v131","v108","unseen"))
    history=pd.read_csv(ART/"nov21_case_checks/v136_final_oof.csv")
    for key,expected in [("row",row),("true",y),("fold",fold),("v131",v131)]:
        np.testing.assert_array_equal(history[key],expected)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{part}.csv",usecols=lambda c:c not in {"Name","Transported"})
                   for part in ("train","test")],ignore_index=True)
    np.testing.assert_array_equal(raw.PassengerId.iloc[row],history.PassengerId)
    blocks,group=field_blocks(raw,row)
    x=np.column_stack(list(blocks.values()))
    names=np.concatenate([np.repeat(k,b.shape[1]) for k,b in blocks.items()])
    candidates={}
    for f in np.unique(fold):
        q=np.flatnonzero(fold==f)
        t=np.flatnonzero((fold!=f)&~np.isin(group,group[q]))
        assert not np.intersect1d(group[t],group[q]).size
        np.testing.assert_array_equal(unseen[q],~np.isin(group[q],group[fold!=f]))
        modes,_,_=fold_predictions(x,names,y,group,t,q,20261112+int(f))
        for key,vals in modes.items():
            candidates.setdefault(key,np.full(len(y),np.nan))[q]=vals
    result={}
    cache={}
    baseline=v131==y
    override=v131!=(v108>=.5)
    for key,p in candidates.items():
        assert np.isfinite(p).all()
        label=np.where(override,v131,.75*v108+.25*p>=.5)
        cache[key]=label
        correct=label==y
        result[key]={}
        for scope,m in [("all",np.ones(len(y),bool)),("unseen_in_original",unseen)]:
            fixed=correct & ~baseline & m
            harmed=~correct & baseline & m
            result[key][scope]=dict(rows=int(m.sum()),accuracy=float(correct[m].mean()),
                                    fixed=int(fixed.sum()),harmed=int(harmed.sum()),
                                    fixed_fn=int((fixed&y).sum()),fixed_fp=int((fixed&~y).sum()),
                                    by_fold=[int(fixed[fold==f].sum())-int(harmed[fold==f].sum()) for f in np.unique(fold)])
    out=ART/"nov23_field_value_transfer"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/"oof.npz",row=row,y=y,fold=fold,unseen=unseen,v131=v131,**cache)
    write(ROOT/"reports/nov23_field_value_transfer/result.json",dict(results=result,
          override_count=int(override.sum()),reference_accuracy=float(baseline.mean())))
    print(result["adaptive"],flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
