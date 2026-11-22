"""Separate different passenger mixes from actually refitting on disjoint bookings."""
from pathlib import Path

import numpy as np
import pandas as pd

from error_overlap_diagnostic import load_authenticated
from relational_error_corrector import write

ROOT = Path(__file__).resolve().parent
PHASE = ROOT/"artifacts/top5_group_phase"


def run():
    inp,_,original,models,_=load_authenticated()
    dev,group,y=inp["dev"],np.asarray(inp["groups"]),inp["y"].astype(bool)
    train,test=[pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=["PassengerId","Cabin"]) for s in ["train","test"]]
    all_rows=pd.concat([train,test],ignore_index=True)
    groups=all_rows.PassengerId.str.split("_").str[0]
    counts=groups.map(groups.value_counts()).to_numpy()
    assert not set(groups.iloc[:len(train)]) & set(groups.iloc[len(train):])
    assert not set(train.Cabin.dropna()) & set(test.Cabin.dropna())
    assert pd.DataFrame({"party":groups,"cabin":all_rows.Cabin}).groupby("cabin").party.nunique().max()==1
    unseen=np.ones(len(y),bool)
    for k in np.unique(original):
        q=original==k
        unseen[q]=~np.isin(group[q],group[~q])
    base=models["v44"]>=.5
    pair=(models["v44"]+models["v93"])>=1
    details={}
    for n in np.unique(counts):
        v=counts[dev]==n
        u=unseen&v
        details[str(n)]=dict(dev=int(v.sum()),unseen=int(u.sum()),unseen_correct=int((u&(pair==y)).sum()),
                             test=int((counts[len(train):]==n).sum()))
    reference={}
    for seed in [20261112,20261115,20261116]:
        predicted={}
        for family in ["v44","mitra"]:
            with np.load(PHASE/f"refs/{family}/group_{seed}.npz") as z:
                np.testing.assert_array_equal(z["dev_ids"],dev)
                np.testing.assert_array_equal(z["group_ids"].astype(int),group.astype(int))
                for k in np.unique(z["fold_id"]):
                    q=z["fold_id"]==k
                    assert not set(group[q]) & set(group[~q])
                predicted[family]=z["prob"]
        pred=predicted["v44"]>=.5
        nums={}
        for label,mask in [("all",np.ones(len(y),bool)),("originally_seen",~unseen),("originally_unseen",unseen)]:
            nums[label]=dict(rows=int(mask.sum()),original_v44=float((base[mask]==y[mask]).mean()),
                             group_v44=float((pred[mask]==y[mask]).mean()),
                             group_v44_raw_mitra=float((((predicted["v44"][mask]+predicted["mitra"][mask])>=1)==y[mask]).mean()))
        reference[str(seed)]=nums
    write(ROOT/"reports/nov23_locked_soft_transfer/booking_mix.json",dict(complete_cabins=all_rows.Cabin.nunique(),
        official_test_rows=len(test),original_unseen_rows=int(unseen.sum()),by_booking_size=details,comparable_reference=reference))
    print(details,reference,flush=True)


if __name__=="__main__":
    run()
