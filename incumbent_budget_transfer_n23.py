"""Transfer independent passenger predictors onto the actual historical v131 development decisions."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from exaone_validation_data import compact_features
from population_rescue_map_n23 import ART, ROOT, read_panel
from relational_error_corrector import write
from spend_simplex_likelihood_n23 import design, learn


def run():
    historical=pd.read_csv(ART/"nov21_case_checks/v136_final_oof.csv")
    upstream=pd.read_csv(ART/"nov21_party_negative/oof.csv")
    row=historical.row.to_numpy()
    fold=historical.fold.to_numpy()
    y,old,other=read_panel()
    np.testing.assert_array_equal(pd.read_csv(ART/"nov23_global_error_inventory.csv").row,row)
    np.testing.assert_array_equal(historical.true,y)
    v131=historical.v131.to_numpy(bool)
    np.testing.assert_array_equal(upstream.row,row)
    np.testing.assert_array_equal(upstream.fold,fold)
    np.testing.assert_array_equal(upstream.target,y)
    np.testing.assert_array_equal(upstream.family_consensus,v131)
    with np.load(ART/"v108_exact_oof/oof.npz") as z:
        for key,a in [("dev_ids",row),("true",y),("fold",fold),("PassengerId",historical.PassengerId)]:
            # The OOF file stores passenger identifiers as Unicode.
            np.testing.assert_array_equal(z[key],a)
        v108=z["v108"]
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Transported","Name"})
    booking=raw.PassengerId.str.split("_").str[0]
    solo=booking.map(booking.value_counts()).to_numpy()[row]==1
    group=booking.to_numpy()[row]
    frame=compact_features(raw.iloc[row]).reset_index(drop=True)
    np.testing.assert_array_equal(historical.PassengerId,raw.PassengerId.iloc[row])
    budget=design(frame)[0]
    full=budget.assign(**{k:frame[k] for k in frame if k not in budget})
    context=frame.drop(columns=["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"])
    challengers={k:np.full(len(y),np.nan) for k in ("registration","budget","full")}
    unseen=np.zeros(len(y),dtype=bool)
    training_counts=[]
    for f in np.unique(fold):
        q=np.flatnonzero(fold==f)
        t=np.flatnonzero((fold!=f)&~np.isin(group,group[q]))
        assert not np.intersect1d(group[t],group[q]).size
        unseen[q]=~np.isin(group[q],group[fold!=f])
        training_counts.append([len(t),len(q)])
        for name,x in [("registration",context),("budget",budget),("full",full)]:
            challengers[name][q]=learn(x,y,t,q)
    override=v131!=(v108>=.5)
    universal=~old.any(0)
    scopes={"all":np.ones(len(y),bool),"new_booking_in_historical_fold":unseen,"single":solo,
            "baseline_persistent":universal,"eight_family_persistent":universal&~other.any(axis=(0,1))}
    result={}
    decisions={}
    for key,p in challengers.items():
        assert np.isfinite(p).all()
        combined=((.75*v108+.25*p)>=.5)
        decision=np.where(override,v131,combined)
        decisions[key]=decision
        groups={}
        for name,scope in scopes.items():
            fixed=scope&(v131!=y)&(decision==y)
            harm=scope&(v131==y)&(decision!=y)
            groups[name]=dict(rows=int(scope.sum()),baseline_accuracy=float((v131[scope]==y[scope]).mean()),
                              accuracy=float((decision[scope]==y[scope]).mean()),fixed=int(fixed.sum()),
                              harmed=int(harm.sum()),fn_fixed=int((fixed&y).sum()),fp_introduced=int((harm&~y).sum()),
                              fp_fixed=int((fixed&~y).sum()),fn_introduced=int((harm&y).sum()))
        result[key]=groups
    directory=ART/"nov23_incumbent_budget_transfer"
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/"oof.npz",row=row,y=y,fold=fold,v131=v131,v108=v108,unseen=unseen,
                        **challengers,**{"decision_"+k:v for k,v in decisions.items()})
    write(ROOT/"reports/nov23_incumbent_budget_transfer/result.json",
          dict(results=result,v131_specific_overrides=int(override.sum()),training_validation_rows=training_counts,
               definition="Incumbent predictions are the historical development replay of v131. Corrections retain all v131 decisions differing from v108. Added classifiers exclude every query booking, while the original v108 used random passenger folds."))
    print({k:(v["all"],v["new_booking_in_historical_fold"]["accuracy"]) for k,v in result.items()})


if __name__=="__main__":
    with threadpool_limits(2):
        run()
