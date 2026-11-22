"""Estimate the outcome contribution of registration information unexpected from the bill."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from conditional_calibration_n23 import offsets
from exaone_validation_data import CATEGORICAL, compact_features
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART, ROOT, SEEDS, read_panel
from relational_error_corrector import write
from spend_simplex_likelihood_n23 import BILLS
from transported_subclasses_n23 import stats


def registration(frame):
    fields = frame[CATEGORICAL].copy()
    fields["age_band"] = np.digitize(frame.Age.fillna(-1), [0, 7, 14, 19, 26, 40, 60]).astype(str)
    fields["cabin_band"] = np.digitize(frame.CabinNumber.fillna(-1), [0, 200, 500, 900, 1400]).astype(str)
    pairs = {a+"_"+b:fields[a]+"/"+fields[b] for a,b in combinations(fields,2)}
    return pd.concat([fields,pd.DataFrame(pairs)],axis=1)


def nuisance(spend, phi, y, groups, train, query, seed):
    # Each training residual excludes that person's entire booking.
    fit_ids = np.r_[train,query]
    p = np.zeros(len(fit_ids))
    mean = np.zeros((len(fit_ids),phi.shape[1]),dtype=np.float32)
    pred_rows = np.arange(len(train),len(fit_ids))
    for a,b in GroupKFold(3,shuffle=True,random_state=seed).split(train,groups=groups[train]):
        assert not np.intersect1d(groups[train[a]],groups[train[b]]).size
        learn = train[a]
        assess = np.r_[train[b],query]
        classifier = HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=15,
                          min_samples_leaf=24,l2_regularization=3,early_stopping=False,random_state=41)
        classifier.fit(spend[learn],y[learn])
        estimates = classifier.predict_proba(spend[assess])[:,1]
        # All dummy targets are fitted together, keeping their expected prevalences compatible.
        regressor = ExtraTreesRegressor(n_estimators=48, min_samples_leaf=18, max_depth=10,
                                        max_features=.85, n_jobs=2, random_state=572)
        regressor.fit(spend[learn],phi[learn])
        expected = regressor.predict(spend[assess])
        p[b],mean[b] = estimates[:len(b)],expected[:len(b)]
        p[pred_rows] += estimates[len(b):]/3
        mean[pred_rows] += expected[len(b):]/3
    return fit_ids,p,mean


def run():
    row = pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    raw = pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"})
    groups = raw.PassengerId.str.split("_").str[0]
    solo = groups.map(groups.value_counts()).to_numpy()[row] == 1
    groups = groups.to_numpy()[row]
    frame=compact_features(raw.iloc[row]).reset_index(drop=True)
    fields=registration(frame)
    bills=frame[BILLS].to_numpy(float)
    spend=np.column_stack([np.log1p(np.nan_to_num(bills)),np.isnan(bills),bills>0,
                           np.log1p(np.nansum(bills,axis=1))])
    y,old,other=read_panel()
    persistent=~old.any(0)
    hard=persistent & ~other.any(axis=(0,1))
    directory=ART/"nov23_residual_registration"
    directory.mkdir(exist_ok=True)
    result={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            base,fold=z["base"],z["fold"]
            verify(z,row,y,fold)
        scores={name:np.full(len(y),np.nan) for name in ["bill","registration","unexpected","shuffled"]}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(groups[t],groups[q]).size
            enc=OneHotEncoder(handle_unknown="ignore",min_frequency=18,sparse_output=False,dtype=np.float32)
            enc.fit(fields.iloc[t])
            phi=enc.transform(fields)
            ids,prior,mu=nuisance(spend,phi,y,groups,t,q,seed+int(k))
            train=np.arange(len(t))
            query=np.arange(len(t),len(ids))
            z=logit(prior.clip(.001,.999))
            scores["bill"][q]=prior[query]
            residual=phi[ids]-mu
            null=residual.copy()
            rng=np.random.default_rng(seed+int(k))
            strata=(prior*10).astype(int)*10+(bills[ids]>0).sum(1)
            for part in (train,query):
                for cell in np.unique(strata[part]):
                    ix=part[strata[part]==cell]
                    null[ix]=residual[rng.permutation(ix)]
            # A common calibration intercept and slope, fitted only on outer training bookings.
            calibration=np.column_stack([np.ones(len(ids)),z])
            for name,signal in [("registration",phi[ids]),("unexpected",residual),("shuffled",null)]:
                matrix=np.column_stack([calibration,signal])
                penalty=np.r_[1.,6.,np.full(signal.shape[1],12.)]
                scores[name][q]=offsets(matrix,z,y[ids].astype(float),train,query,penalty)
        perf={}
        for key,s in scores.items():
            assert np.isfinite(s).all()
            for name,p in [(key,s),(key+"_blend",.75*base+.25*s)]:
                perf[name]=stats(p>=.5,base>=.5,y,persistent,hard,solo)
        np.savez_compressed(directory/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**scores)
        result[seed]=perf
        write(ROOT/"reports/nov23_residual_registration/result.json",dict(results=result))
        print(seed,{k:(round(v["accuracy"]*100,2),v["fixed"],v["harmed"],v["persistent_false_negatives"]) for k,v in perf.items()},flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
