"""Estimate class-dependent context inclusion and undo induced posterior-odds shift."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.special import expit, logit

from archive_error_recovery_n23 import ART, ROOT, SEEDS
from earth_bill_order_n23 import BILLS
from earth_order_exceptions_n23 import matches
from exaone_validation_data import CATEGORICAL, compact_features
from relational_error_corrector import write


def sample_context(t,y,bills,missing,group,rng):
    shared=np.concatenate([rng.choice(t[y[t]==label],256,replace=False) for label in (False,True)])
    pool=np.setdiff1d(t,shared)
    pairs=matches(pool,y,bills,missing,group,rng)
    extra=pairs[rng.choice(len(pairs),256,replace=False)].ravel()
    selected=np.concatenate([shared,extra])
    assert len(np.unique(selected))==1024 and y[selected].sum()==512
    return selected


def run():
    data={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_earth_tabpfn_context/{seed}.npz") as z:
            data[seed]={k:z[k] for k in ("base","balanced","exceptions","row","y","fold")}
    first=data[SEEDS[0]]
    row,y=first["row"],first["y"]
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda k:k not in {"Name","Transported"}).iloc[row]
    group=raw.PassengerId.str.split("_").str[0].to_numpy()
    earth=raw.HomePlanet.eq("Earth").to_numpy()
    bills=raw[BILLS].fillna(0).to_numpy()*[1,1,-1,-1,-1]
    missing=raw[BILLS].isna().to_numpy()@(2**np.arange(5))
    frame=compact_features(raw).reset_index(drop=True)
    repeat=earth&np.logical_and.reduce([(v["base"]>=.5)!=y for v in data.values()])
    resolved=repeat&np.logical_and.reduce([(v["exceptions"]>=.5)==y for v in data.values()])
    result,all_hits={},{}
    out=ART/"nov23_earth_context_selection_bias"
    out.mkdir(exist_ok=True)
    for seed,v in data.items():
        np.testing.assert_array_equal(row,v["row"])
        np.testing.assert_array_equal(y,v["y"])
        fold,base=v["fold"],v["base"]
        predictions={k:base.copy() for k in ("balanced_prior","exception_prior","selection_corrected")}
        records=[]
        for f in np.unique(fold):
            t=np.flatnonzero(earth&(fold!=f))
            q=np.flatnonzero(earth&(fold==f))
            assert not np.intersect1d(group[t],group[q]).size
            original_selection=sample_context(t,y,bills,missing,group,np.random.default_rng(seed+int(f)))
            with np.load(ART/f"nov23_earth_tabpfn_context/{seed}_{f}_exceptions.npz") as saved:
                np.testing.assert_array_equal(np.sort(row[original_selection]),saved["support"])
            rng=np.random.default_rng(seed+704+int(f))
            count=np.zeros(len(y))
            for _ in range(40):
                count[sample_context(t,y,bills,missing,group,rng)]+=1
            exposure=count/40
            propensities={}
            variability={}
            for label in (False,True):
                rows=t[y[t]==label]
                model=CatBoostRegressor(iterations=180,depth=4,learning_rate=.075,
                                        cat_features=CATEGORICAL,thread_count=2,
                                        random_seed=954,verbose=False,allow_writing_files=False)
                model.fit(frame.iloc[rows],exposure[rows])
                # Everyone has at least the class's shared-random inclusion probability.
                propensities[label]=np.clip(model.predict(frame.iloc[q]),256/len(rows),1)
                variability[str(label)]=np.quantile(exposure[rows],[.1,.5,.9]).tolist()
            prior=np.log(y[t].sum()/(~y[t]).sum())
            offset=np.log(propensities[False])-np.log(propensities[True])
            for key,source,shift in [("balanced_prior","balanced",prior),("exception_prior","exceptions",prior),
                                     ("selection_corrected","exceptions",offset)]:
                predictions[key][q]=expit(logit(v[source][q].clip(.001,.999))+shift)
            records.append(dict(context_exposures_q10_q50_q90=variability,
                                held_log_odds_adjustment=np.quantile(offset,[.1,.5,.9]).tolist()))
        for k,p in list(predictions.items()):
            predictions[k+"_quarter"]=.75*base+.25*p
        predictions["uncorrected"]=v["exceptions"]
        scores={}
        ref=(base>=.5)==y
        for name,p in predictions.items():
            ok=(p>=.5)==y
            truth=np.where(y,p,1-p).clip(1e-9,1)
            scores[name]=dict(accuracy=float(ok.mean()),earth_logloss=float(-np.log(truth[earth]).mean()),
                              fixed=int((ok&~ref).sum()),harmed=int((~ok&ref).sum()),
                              earth_repeat=int((ok&repeat).sum()),retained_of_115=int((ok&resolved).sum()))
            all_hits.setdefault(name,[]).append(ok)
        result[seed]=dict(scores=scores,folds=records)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**predictions)
        print(seed,scores["selection_corrected"],flush=True)
    stable={k:dict(earth804=int((repeat&np.logical_and.reduce(h)).sum()),
                   retained115=int((resolved&np.logical_and.reduce(h)).sum())) for k,h in all_hits.items()}
    write(ROOT/"reports/nov23_earth_context_selection_bias/result.json",dict(stable=stable,results=result))


if __name__=="__main__":
    run()
