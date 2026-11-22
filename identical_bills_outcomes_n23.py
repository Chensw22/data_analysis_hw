"""Estimate what identifies different outcomes when all five recorded bills coincide."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from archive_error_recovery_n23 import ART, ROOT, SEEDS
from earth_bill_order_n23 import BILLS
from exaone_validation_data import compact_features, CATEGORICAL
from relational_error_corrector import write


def features(raw):
    booking=raw.PassengerId.str.split("_").str[0]
    roster=compact_features(raw)
    self_fields=roster.copy()
    self_fields["party_size"]=booking.map(booking.value_counts())
    for b in BILLS:
        self_fields[b+"_missing"]=raw[b].isna().astype(float)
    party=self_fields.copy()
    observed=pd.concat([pd.get_dummies(roster[CATEGORICAL],dtype=float),roster[["Age","CabinNumber"]],
                        np.log1p(raw[BILLS])],axis=1)
    available=observed.notna().groupby(booking).transform("sum")-observed.notna().astype(int)
    totals=observed.fillna(0).groupby(booking).transform("sum")-observed.fillna(0)
    party=pd.concat([party,(totals/available.replace(0,np.nan)).add_prefix("other_")],axis=1)
    return {"registration":self_fields,"party":party}


def run():
    with np.load(ART/f"nov23_group_kernel/{SEEDS[0]}.npz") as z:
        row,y=z["row"],z["y"]
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"})
    group=raw.PassengerId.str.split("_").str[0].to_numpy()[row]
    views={k:v.iloc[row].reset_index(drop=True) for k,v in features(raw).items()}
    raw=raw.iloc[row]
    earth=raw.HomePlanet.eq("Earth").to_numpy()
    cell=raw.groupby(BILLS,dropna=False).ngroup().to_numpy()
    data={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            data[seed]=z["base"],z["fold"]
    repeat=np.logical_and.reduce([(base>=.5)!=y for base,_ in data.values()])
    out=ART/"nov23_identical_bills"
    out.mkdir(exist_ok=True)
    report={}
    for seed,(base,fold) in data.items():
        pred={k:base.copy() for k in ["bill_frequency",*views]}
        selected=np.zeros(len(y),bool)
        for f in np.unique(fold):
            t=np.flatnonzero(fold!=f)
            n=np.bincount(cell[t],minlength=cell.max()+1)
            positives=np.bincount(cell[t],weights=y[t],minlength=len(n))
            # Eligibility is established from training outcomes only.
            mixed=(positives>0)&(positives<n)
            q=np.flatnonzero((fold==f)&mixed[cell])
            t=t[mixed[cell[t]]]
            assert not np.intersect1d(group[t],group[q]).size
            selected[q]=True
            pred["bill_frequency"][q]=(positives[cell[q]]+.5)/(n[cell[q]]+1)
            for name,x in views.items():
                model=CatBoostClassifier(iterations=280,depth=5,learning_rate=.05,thread_count=2,random_seed=121,
                                         allow_writing_files=False,verbose=False,cat_features=CATEGORICAL)
                model.fit(x.iloc[t],y[t].astype(int))
                pred[name][q]=model.predict_proba(x.iloc[q])[:,1]
        pred["baseline"]=base
        pred["party_quarter"]=.75*base+.25*pred["party"]
        metrics={}
        original=(base>=.5)==y
        for k,p in pred.items():
            ok=(p>=.5)==y
            metrics[k]=dict(whole_accuracy=float(ok.mean()), selected_accuracy=float(ok[selected].mean()),
                            selected_earth_accuracy=float(ok[selected&earth].mean()),
                            fixed=int((ok&~original).sum()),harmed=int((~ok&original).sum()),
                            repeat_fixed=int((ok&repeat&selected).sum()),earth_repeat_fixed=int((ok&repeat&earth&selected).sum()))
        report[seed]=dict(scored=int(selected.sum()),earth=int((selected&earth).sum()),
                          covered_repeat=int((selected&repeat).sum()),earth_repeat=int((selected&repeat&earth).sum()),
                          scores=metrics)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,eligible=selected,**pred)
    write(ROOT/"reports/nov23_identical_bills/result.json",report)


if __name__=="__main__":
    run()
