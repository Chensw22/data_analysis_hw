"""Test official-domain importance weighting with whole-booking validation and a depleted source."""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from broad_noise_transition_n23 import ART,ROOT,SEEDS
from exaone_validation_data import CATEGORICAL,compact_features


def classifier(iters,depth,seed):
    return CatBoostClassifier(iterations=iters,depth=depth,learning_rate=.06,thread_count=2,
                              verbose=False,allow_writing_files=False,random_seed=seed,cat_features=CATEGORICAL)


def ratios(frame,groups,source,test):
    rows=np.r_[source,test]
    domain=(np.arange(len(rows))>=len(source)).astype(int)
    gs=groups[rows]
    folds=np.full(len(rows),-1)
    for k,(_,v) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=1493).split(rows,domain,gs)):
        folds[v]=k
    odds=np.full(len(rows),np.nan)
    ranked=np.full(len(rows),np.nan)
    slopes=[]
    for k in range(3):
        train=np.flatnonzero(folds==k)
        cal=np.flatnonzero(folds==(k+1)%3)
        val=np.flatnonzero(folds==(k+2)%3)
        assert not set(gs[val]) & (set(gs[cal])|set(gs[train]))
        model=classifier(180,4,1357).fit(frame.iloc[rows[train]],domain[train])
        p=model.predict(frame.iloc[rows],prediction_type="RawFormulaVal")
        calibration=LogisticRegression(C=1.).fit(p[cal,None],domain[cal])
        prior=domain[cal].mean()
        slope=float(calibration.coef_[0,0])
        slopes.append(slope)
        ranked[val]=p[val]-logit(domain[train].mean())
        ratio=calibration.decision_function(p[val,None])-logit(prior)
        odds[val]=np.exp(np.clip(ratio,-np.log(4),np.log(4))) if slope>0 else 1.
    assert np.isfinite(odds).all() and np.isfinite(ranked).all()
    weights=odds[:len(source)]
    return weights/weights.mean(),dict(domain_auc=float(roc_auc_score(domain,ranked)),slopes=slopes,
                                      weight_quantiles=np.quantile(weights,[.05,.5,.95]).tolist(),
                                      ess_ratio=float(weights.sum()**2/((weights@weights)*len(weights))))


def make_target_mix(size,planet,rows,test):
    # Standardize the entire validation cohort to official test frequencies.
    # Categories and counts require no outcome or domain-classifier prediction.
    buckets=np.char.add(np.minimum(size,4).astype(str),planet)
    counts=pd.Series(buckets[test]).value_counts(normalize=True)
    sample=pd.Series(buckets[rows]).value_counts(normalize=True)
    assert set(counts.index)==set(sample.index)
    return np.array([(counts/sample).to_dict()[key] for key in buckets[rows]])


def run():
    frames=[pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ("Name","Transported")) for s in ("train","test")]
    raw=pd.concat(frames,ignore_index=True)
    test=np.arange(len(frames[0]),len(raw))
    groups=raw.PassengerId.str.split("_").str[0].to_numpy()
    assert not np.intersect1d(groups[:len(frames[0])],groups[test]).size
    frame=compact_features(raw)
    by=pd.Series(groups).value_counts()
    size=np.array([by[g] for g in groups])
    frame["BookingSize"]=size
    frame["MemberNumber"]=raw.PassengerId.str.split("_").str[1].astype(int)
    frame["CabinOccupants"]=raw.Cabin.map(raw.Cabin.value_counts()).fillna(0)
    # Unlabeled roster only, and available for both official domains.
    planet=raw.HomePlanet.fillna("missing").to_numpy(dtype=str)
    inventory=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    row,y=inventory.row.to_numpy(),inventory.target.to_numpy(bool)
    target_mix=make_target_mix(size,planet,row,test)
    report={}
    hits={}
    baselines=[]
    size_share={s:np.bincount(np.minimum(size[v],4),minlength=5)[1:].tolist()
                for s,v in (("official_train",np.arange(len(frames[0]))),("development",row),("test",test))}
    directory=ART/"nov23_broad_target_shift"
    directory.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(row,z["row"])
            np.testing.assert_array_equal(y,z["y"])
            ref,fold=z["base"],z["fold"]
        baselines.append((ref>=.5)==y)
        scores={}
        domain_stats=[]
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not set(groups[row[t]]) & set(groups[row[q]])
            group_names,ix=np.unique(groups[row[t]],return_inverse=True)
            rg=np.random.default_rng(1703+int(k))
            draws=rg.random(len(group_names))
            keep=draws[ix]<np.where(size[row[t]]==1,.93,.34)
            for scenario,source in (("natural",t),("depleted",t[keep])):
                w,diag=ratios(frame,groups,row[source],test)
                diag.update(fold=int(k),scenario=scenario,training=len(source))
                domain_stats.append(diag)
                perm=np.random.default_rng(131).permutation(w)
                for mode,weight in (("ordinary",None),("domain",w),("placebo",perm)):
                    key=scenario+"_"+mode
                    model=classifier(440,6,749).fit(frame.iloc[row[source]],y[source],sample_weight=weight)
                    p=model.predict_proba(frame.iloc[row[q]])[:,1]
                    scores.setdefault(key,np.full(len(y),np.nan))[q]=p
        np.savez_compressed(directory/f"{seed}.npz",row=row,y=y,fold=fold,base=ref,**scores)
        metrics={}
        for key,p in scores.items():
            assert np.isfinite(p).all()
            ok=(p>=.5)==y
            hits.setdefault(key,[]).append(ok)
            ref_ok=(ref>=.5)==y
            ordinary=(scores[key.split("_")[0]+"_ordinary"]>=.5)==y
            metrics[key]=dict(accuracy=ok.mean(),official_mix=float(np.average(ok,weights=target_mix)),
                              baseline=[int((ok&~ref_ok).sum()),int((~ok&ref_ok).sum())],
                              ordinary=[int((ok&~ordinary).sum()),int((~ok&ordinary).sum())])
        report[seed]=dict(metrics=metrics,diagnostics=domain_stats)
        print(seed,metrics["natural_domain"],metrics["depleted_domain"],flush=True)
    hard=~np.any(baselines,0)
    coverage={k:int((np.all(h,0)&hard).sum()) for k,h in hits.items()}
    out=ROOT/"reports/nov23_broad_target_shift"
    out.mkdir(exist_ok=True)
    (out/"result.json").write_text(json.dumps(dict(size_shares=size_share,split=report,stable1070=coverage,
                                        groups=len(by),official_train_groups=len(set(groups[:len(frames[0])])),
                                        official_test_groups=len(set(groups[test]))),indent=2)+"\n")


if __name__=="__main__":
    run()
