"""Locate stable recoveries across all baseline misses at the requested one-point tolerance."""
import numpy as np
import pandas as pd

from archive_error_recovery_n23 import ROOT, SEEDS, panel
from relational_error_corrector import write


def run():
    row,y,base,probs,hard,names,_=panel()
    hits=(probs>=.5)==y[None,:,None]
    ref=(base>=.5)==y
    stable=hits.all(axis=0)
    within_one=(hits.mean(axis=1)>=ref.mean(axis=1)[:,None]-.01).all(axis=0)
    strong=stable[:,within_one].any(axis=1)
    weak_only=stable.any(axis=1) & ~strong
    repeated=~ref.any(axis=0)
    clean=ref.all(axis=0)
    ever=~clean
    planet=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["HomePlanet"]).HomePlanet.iloc[row].to_numpy()
    party=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.str.split("_").str[0]
    size=party.map(party.value_counts()).to_numpy()[row]
    partitions={}
    for scope,cohort in [("at_least_one_base_miss",ever),("base_wrong_each_split",repeated),("hard365",hard)]:
        counts={}
        for name,mask in [("same_level_stable",strong),("only_lower_level_stable",weak_only),
                          ("intermittent_only",~stable.any(1)&hits.any(axis=(0,2))),
                          ("no_observed_hit",~hits.any(axis=(0,2)))]:
            subset=cohort & mask
            counts[name]=dict(rows=int(subset.sum()), positive=int((subset&y).sum()),
                              negative=int((subset&~y).sum()),earth=int((subset&(planet=="Earth")).sum()),
                              alone_on_training_manifest=int((subset&(size==1)).sum()))
        assert sum(v["rows"] for v in counts.values())==cohort.sum()
        partitions[scope]=counts
    d={}
    for j,name in enumerate(names):
        reliable=stable[:,j]
        harmed=~hits[:,:,j].any(axis=0)&clean
        d[name]=dict(within_one=bool(within_one[j]), actual_acc=hits[:,:,j].mean(axis=1).tolist(),
                     stable_fixes=int((reliable&repeated).sum()), stable_harms=int(harmed.sum()),
                     lower_only_fixes=int((reliable&repeated&weak_only).sum()))
    needed=int(np.floor(.85*len(y)))+1-ref.sum(axis=1)
    write(ROOT/"reports/nov23_broad_rescue_frontier/result.json",dict(
        seeds=SEEDS, participants=len(y),experiments=len(names),within_one_count=int(within_one.sum()),
        target_correct=int(np.floor(.85*len(y)))+1,baseline_correct=ref.sum(axis=1).tolist(),
        target_net_gains=needed.tolist(),partitions=partitions,methods=d))
    print(partitions["base_wrong_each_split"],needed)


if __name__=="__main__":
    run()
