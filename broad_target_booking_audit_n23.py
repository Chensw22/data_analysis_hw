"""Test the official domain's booking-size distribution at the sampling unit."""
import json

import numpy as np
import pandas as pd

from broad_noise_transition_n23 import ROOT


def run():
    sets=[]
    for split in ("train","test"):
        ids=pd.read_csv(ROOT/f"data/raw/{split}.csv",usecols=["PassengerId"]).PassengerId
        groups=ids.str.split("_").str[0].value_counts()
        sets.append(groups)
    train,test=sets
    assert not set(train.index)&set(test.index)
    hist=np.stack([np.bincount(s,minlength=9)[1:9] for s in sets])
    total=hist.sum(0)
    n=hist.sum(1)
    expected=n[:,None]*total/total.sum()
    chi=float(np.sum((hist-expected)**2/expected))
    rng=np.random.default_rng(5209)
    draws=rng.multivariate_hypergeometric(total,n[1],size=10000)
    sim=((draws-expected[1])**2/expected[1]+(total-draws-expected[0])**2/expected[0]).sum(1)
    result=dict(size=range(1,9),training_groups=hist[0].tolist(),test_groups=hist[1].tolist(),
                chi_square=chi,permutation_p=(1+int(np.sum(sim>=chi)))/(1+len(sim)))
    result["size"]=list(result["size"])
    folder=ROOT/"reports/nov23_broad_target_shift"
    folder.mkdir(exist_ok=True)
    (folder/"sampling.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__=="__main__":
    run()
