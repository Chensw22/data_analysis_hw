"""Quantify uncertainty in passenger accuracy changes while resampling whole bookings."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ROOT,ART,SEEDS
from grouped_five_way_coverage_n23 import EXPERIMENTS,verify
from relational_error_corrector import write


def run():
    people=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    dev,y=people.row.to_numpy(),people.target.to_numpy(bool)
    group=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.iloc[dev].str.split("_").str[0]
    _,idx=np.unique(group,return_inverse=True)
    n=np.bincount(idx)
    panels=[]
    baselines=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            anchor,fold=z["base"],z["fold"]
            verify(z,dev,y,fold)
        correct=((anchor>=.5)==y).astype(float)
        baselines.append(float(correct.mean()))
        candidates={}
        for folder,name in EXPERIMENTS.items():
            with np.load(ART/f"nov23_{folder}/{seed}.npz") as z:
                verify(z,dev,y,fold)
                candidates[name]=z[name]
            candidates[name+"_blend"]=.75*anchor+.25*candidates[name]
        for folder,keys in [("group_disputes",["shallow_tree","two_level_table"]),("group_pool",["constant","contextual"])]:
            with np.load(ART/f"nov23_{folder}/{seed}.npz") as z:
                verify(z,dev,y,fold)
                candidates.update({k:z[k] for k in keys})
        names=list(candidates)
        delta=np.column_stack([((s>=.5)==y)-correct for s in candidates.values()])
        panels.append(delta)
    # Repeated splits use identical targets. Aggregate within passenger before sampling bookings.
    panel=np.stack(panels)
    person_delta=panel.mean(0)
    group_delta=np.zeros((len(n),len(names)))
    np.add.at(group_delta,idx,person_delta)
    gains=person_delta.mean(0)
    rng=np.random.default_rng(6123)
    draws=[]
    for _ in range(80):
        freq=rng.multinomial(len(n),np.full(len(n),1./len(n)),size=50)
        draws.append((freq@group_delta)/(freq@n)[:,None])
    draws=np.concatenate(draws)
    lo,hi=np.quantile(draws,[.025,.975],axis=0)
    joint_width=np.quantile(np.max(np.abs(draws-gains),axis=1),.95)
    summary={name:dict(delta_pp=float(100*gains[i]),
                       delta_by_partition_pp=(100*panel[:,:,i].mean(1)).tolist(),
                       paired_bootstrap_95_pp=[float(100*lo[i]),float(100*hi[i])],
                       simultaneous_95_pp=[float(100*(gains[i]-joint_width)),float(100*(gains[i]+joint_width))])
             for i,name in enumerate(names)}
    write(ROOT/"reports/nov23_five_way_coverage/booking_gain.json",dict(bookings=len(n),passengers=len(y),
          baseline_accuracy=baselines,results=summary,bootstrap_draws=len(draws),
          interpretation="Descriptive paired sampling uncertainty with trained models held fixed; development labels have been reused across research."))
    print(sorted((round(s["delta_pp"],3),k) for k,s in summary.items())[-4:],flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
