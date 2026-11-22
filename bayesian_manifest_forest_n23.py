"""Compare posterior probit BART and gradient boosting on unseen passenger bookings."""
import os
from pathlib import Path
import sys
from importlib.metadata import version

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"artifacts/nov23_bart"
os.environ.setdefault("MPLCONFIGDIR",str(OUT/"matplotlib"))
sys.path.insert(0,str(OUT/"vendor"))
from stochtree import BARTModel, OutcomeModel
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from exaone_validation_data import CATEGORICAL,compact_features
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART,SEEDS,read_panel
from relational_error_corrector import write
from spend_simplex_likelihood_n23 import BILLS
from transported_subclasses_n23 import stats


def design(frame,train,party_size):
    enc=OneHotEncoder(handle_unknown="ignore",sparse_output=False).fit(frame.iloc[train][CATEGORICAL])
    num=frame.drop(columns=CATEGORICAL).copy()
    cost=frame[BILLS].to_numpy(float)
    for name in BILLS:
        num[name]=np.log1p(num[name])
    num["total"]=np.log1p(np.nansum(cost,axis=1))
    num["positive_services"]=(cost>0).sum(1)
    num["party_size"]=party_size
    num["discretionary_share"]=(1+np.nansum(cost[:,[1,2]],axis=1))/(2+np.nansum(cost,axis=1))
    return np.column_stack([enc.transform(frame[CATEGORICAL]),num.fillna(num.iloc[train].median()).fillna(0),
                            num.isna()]).astype(np.float64)


def predict(x,y,t,q,seed,burnin,kept):
    bart=BARTModel()
    bart.sample(X_train=x[t],y_train=y[t].astype(float),num_gfr=6,num_burnin=burnin,num_mcmc=kept,
                general_params=dict(outcome_model=OutcomeModel(outcome="binary",link="probit"),random_seed=seed,
                                    num_threads=1,num_chains=2,keep_gfr=False,sample_sigma2_global=False),
                mean_forest_params=dict(num_trees=80,sample_sigma2_leaf=False))
    draws=ndtr(bart.predict(X=x[q],terms="mean_forest",type="posterior",scale="linear"))
    assert draws.shape==(len(q),2*kept) and np.isfinite(draws).all()
    classic=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,min_samples_leaf=25,
                  learning_rate=.06,l2_regularization=3,early_stopping=False,random_state=612)
    classic.fit(x[t],y[t])
    chain=np.abs(draws[:,:kept].mean(1)-draws[:,kept:].mean(1))
    return dict(bart=draws.mean(1),positive_vote=(draws>=.5).mean(1),sd=draws.std(1),
                chain_difference=chain,boosting=classic.predict_proba(x[q])[:,1])


def run(long_chain=False):
    burnin,kept=(360,384) if long_chain else (50,64)
    directory=ART/("nov23_bart_long" if long_chain else "nov23_bart")
    directory.mkdir(exist_ok=True)
    report=ROOT/"reports"/directory.name
    assert version("stochtree")=="0.4.5"
    row=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"})
    group=raw.PassengerId.str.split("_").str[0]
    size=group.map(group.value_counts()).to_numpy()[row]
    group=group.to_numpy()[row]
    frame=compact_features(raw.iloc[row]).reset_index(drop=True)
    y,old,others=read_panel()
    persistent=~old.any(0)
    hard=persistent&~others.any(axis=(0,1))
    result={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            base,fold=z["base"],z["fold"]
            verify(z,row,y,fold)
        estimates={}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(group[t],group[q]).size
            pred=predict(design(frame,t,size),y,t,q,seed+int(k),burnin,kept)
            if long_chain:
                np.savez_compressed(directory/f"{seed}_fold{k}.npz",row=row[q],y=y[q],base=base[q],**pred)
            for name,p in pred.items():
                estimates.setdefault(name,np.full(len(y),np.nan))[q]=p
            print(seed,"fold",int(k),"posterior accuracy",np.mean((pred["bart"]>=.5)==y[q]),flush=True)
        np.savez_compressed(directory/f"{seed}.npz",row=row,fold=fold,y=y,base=base,**estimates)
        bart=estimates["bart"]
        vote=estimates["positive_vote"]
        alternatives=dict(bart=bart,boosting=estimates["boosting"],quarter_bart=.75*base+.25*bart,
                          quarter_boosting=.75*base+.25*estimates["boosting"],
                          posterior_override=np.where((vote>=.9)|(vote<=.1),bart,base))
        metrics={name:stats(p>=.5,base>=.5,y,persistent,hard,size==1) for name,p in alternatives.items()}
        result[seed]=dict(metrics=metrics, median_chain_probability_difference=float(np.median(estimates["chain_difference"])),
                          p95_chain_probability_difference=float(np.quantile(estimates["chain_difference"],.95)))
        write(report/"result.json",dict(stochtree=version("stochtree"),results=result,
                    sampler=dict(trees=80,chains=2,grow_from_root=6,burnin=burnin,kept_per_chain=kept)))
        print(seed,[(k,round(v["accuracy"]*100,3),v["fixed"],v["harmed"]) for k,v in metrics.items()],flush=True)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-chain",action="store_true")
    with threadpool_limits(2):
        run(parser.parse_args().long_chain)
