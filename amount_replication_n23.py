"""Replicate the fixed quarter-weight amount ablation under a new reference and eight booking folds."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from exaone_validation_data import compact_features
from matched_booking_learning_curve_n23 import portions, OUT as REFERENCE
from population_rescue_map_n23 import ART, ROOT, read_panel
from relational_error_corrector import write
from spend_simplex_likelihood_n23 import design, learn
from transported_subclasses_n23 import stats


def run():
    row = pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    y, old, others = read_panel()
    raw = pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"})
    group = raw.PassengerId.str.split("_").str[0]
    single = group.map(group.value_counts()).to_numpy()[row] == 1
    group = group.to_numpy()[row]
    fields = compact_features(raw.iloc[row]).reset_index(drop=True)
    budget = design(fields)[0]
    full = budget.assign(**{name:fields[name] for name in fields if name not in budget})
    persistent = ~old.any(0)
    hard = persistent & ~others.any(axis=(0,1))
    refs = {}
    for name in ("v44","tabicl"):
        with np.load(REFERENCE/f"{name}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            if refs:
                np.testing.assert_array_equal(fold,z["fold"])
                np.testing.assert_array_equal(sizes,z["sizes"])
            fold,sizes = z["fold"],z["sizes"]
            refs[name] = np.array([z["small"],z["large"]])
    base=(refs["v44"]+refs["tabicl"])/2
    pred={key:np.full_like(base,np.nan) for key in ["budget","full"]}
    for k,s,t,q in portions(group):
        np.testing.assert_array_equal(q,np.flatnonzero(fold==k))
        np.testing.assert_array_equal(sizes[k],[len(s),len(t),len(q)])
        for amount,train in enumerate([s,t]):
            assert not np.intersect1d(group[train],group[q]).size
            for name,frame in [("budget",budget),("full",full)]:
                pred[name][amount,q]=learn(frame,y,train,q)
        print("completed fold",k,flush=True)
    folder=ART/"nov23_amount_replication"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/"oof.npz",row=row,y=y,fold=fold,base=base,**pred)
    result={}
    for i,key in enumerate(["small","large"]):
        result[key]={}
        for name,p in [("base",base),("budget",pred["budget"]),("full",pred["full"])]:
            assert np.isfinite(p).all()
            hit=(.75*base[i]+.25*p[i])>=.5
            metrics=stats(hit,base[i]>=.5,y,persistent,hard,single)
            diff=(hit==y).astype(int)-((base[i]>=.5)==y)
            metrics["changes_by_fold"]=[int(diff[fold==k].sum()) for k in range(8)]
            result[key][name]=metrics
    write(ROOT/"reports/nov23_amount_replication/result.json",
          dict(results=result,rows=len(row),training_sizes=sizes.tolist(),
               baseline="Equal-weight v44 and TabICL. Both challengers use the same CatBoost settings, input rows, and 25% weight. Full adds individual amounts back while retaining the identical reduced feature block."))
    print(result,flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
