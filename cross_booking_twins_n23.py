"""Audit exact equivalence and fixed-tolerance twins on the labeled development roster."""
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from boarding_sequence_n23 import hard_rows
from exaone_validation_data import compact_features
from grouped_five_way_coverage_n23 import ART, ROOT, EXPERIMENTS, SEEDS, verify
from relational_error_corrector import write

BILLS = ["FoodCourt","Spa","RoomService","VRDeck","ShoppingMall"]
CAT = ["HomePlanet","Destination","VIP","CryoSleep","Deck","Side"]


def masks(rows):
    row,y=rows.row.to_numpy(), rows.target.to_numpy(bool)
    hard=hard_rows(row,y)
    always_wrong=np.ones(len(y),bool)
    always_right=np.ones(len(y),bool)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold=z["fold"]
            verify(z,row,y,fold)
            anchor=z["base"]
        always_wrong &= (anchor>=.5)!=y
        always_right &= (anchor>=.5)==y
        for folder,method in EXPERIMENTS.items():
            with np.load(ART/f"nov23_{folder}/{seed}.npz") as z:
                verify(z,row,y,fold)
                prob=z[method]
                for p in [prob,.75*anchor+.25*prob]:
                    hard &= (p>=.5)!=y
        with np.load(ART/f"nov23_group_disputes/{seed}.npz") as z:
            verify(z,row,y,fold)
            for k in ["two_level_table","shallow_tree"]:
                hard &= (z[k]>=.5)!=y
    assert hard.sum()==356 and always_wrong.sum()==1070 and always_right.sum()==5436
    return {"hard356":hard,"other_consistent_errors":always_wrong&~hard,
            "consistent_correct":always_right,"all":np.ones(len(y),bool)}


def summarize(count,opposite,scope):
    out={}
    for label,mask in scope.items():
        exposed=mask & (count>0)
        contradictory=exposed & (opposite>0)
        out[label]=dict(rows=int(mask.sum()),matched_rows=int(exposed.sum()),
            any_opposite_rows=int(contradictory.sum()),only_opposite_rows=int((exposed&(count==opposite)).sum()),
            mean_opposite_share=float(np.mean(opposite[exposed]/count[exposed])) if exposed.any() else None)
    return out


def exact(x,y,g,scope):
    count=np.zeros(len(x),int)
    opposite=np.zeros(len(x),int)
    patterns=x.groupby(list(x),dropna=False,sort=False).indices
    for idx in patterns.values():
        if len(idx)<2:
            continue
        different=g[idx,None]!=g[idx][None,:]
        count[idx]=different.sum(1)
        opposite[idx]=(different&(y[idx,None]!=y[idx][None,:])).sum(1)
    bounds={}
    for label,mask in scope.items():
        selected=np.unique(np.flatnonzero(mask))
        group_sizes=x.iloc[selected].assign(_y=y[selected]).groupby(list(x),dropna=False,sort=False)._y.agg(["size","sum"])
        mixed=(group_sizes["sum"]>0)&(group_sizes["sum"]<group_sizes["size"])
        unavoidable=int(np.minimum(group_sizes["sum"],group_sizes["size"]-group_sizes["sum"]).sum())
        bounds[label]=dict(rows=len(selected),mixed_patterns=int(mixed.sum()),mixed_rows=int(group_sizes.loc[mixed,"size"].sum()),
                           min_errors=unavoidable,max_correct=len(selected)-unavoidable)
    return dict(features=list(x),cross_booking=summarize(count,opposite,scope),empirical_representation_bounds=bounds)


def near(x,y,g,scope):
    keys=x[CAT].copy()
    for col in ["Age","CabinNumber",*BILLS]:
        keys[col+"_missing"]=x[col].isna()
    coords=[x.Age.fillna(0).to_numpy()/2,x.CabinNumber.fillna(0).to_numpy()/10]
    for col in BILLS:
        vals=x[col].fillna(0).to_numpy()
        keys[col+"_positive"]=vals>0
        coords.append(np.log(np.maximum(1.,vals))/np.log(1.1))
    matrix=np.column_stack(coords)
    count,opposite=np.zeros(len(x),int),np.zeros(len(x),int)
    for idx in keys.groupby(list(keys),sort=False,dropna=False).indices.values():
        if len(idx)<2:
            continue
        tree=cKDTree(matrix[idx])
        pairs=tree.query_pairs(1.+1e-10,p=np.inf,output_type="ndarray")
        i,j=idx[pairs[:,0]],idx[pairs[:,1]]
        ok=g[i]!=g[j]
        i,j=i[ok],j[ok]
        for side in [i,j]:
            np.add.at(count,side,1)
            np.add.at(opposite,side,y[i]!=y[j])
    return summarize(count,opposite,scope)


def run():
    rows=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    roster=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ["Name","Transported"]).iloc[rows.row]
    y=rows.target.to_numpy(bool)
    x=compact_features(roster).reset_index(drop=True)
    g=roster.PassengerId.str.split("_").str[0].to_numpy()
    scope=masks(rows)
    out=dict(exact_compact13=exact(x,y,g,scope),exact_without_berth_number=exact(x.drop(columns="CabinNumber"),y,g,scope),
        tolerance_cross_booking=near(x,y,g,scope),protocol=dict(age_years=2,cabin_numbers=10,positive_bill_max_ratio=1.1,
        matched="All categorical values, missing patterns and zero/positive states must match individually. Different bookings only.",
        scope="Retrospective diagnostic on 6954 development labels. Exact maxima apply only to the named feature list and observed sample; close matches impose no upper bound."))
    write(ROOT/"reports/nov23_cross_booking_twins/result.json",out)
    print(json.dumps(out))


if __name__=="__main__":
    run()
