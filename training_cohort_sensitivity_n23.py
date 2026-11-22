"""Predict which strong-reference scores to revise using responses to training-cohort interventions."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from conditional_calibration_n23 import offsets
from exaone_validation_data import CATEGORICAL,compact_features
from grouped_five_way_coverage_n23 import verify
from nested_manifest_expressions_n23 import BILLS
from population_rescue_map_n23 import ART,ROOT,SEEDS,read_panel
from relational_error_corrector import write


def fit(x,y,t,q,weights):
    m=CatBoostClassifier(iterations=280,depth=5,learning_rate=.065,l2_leaf_reg=6,
                         thread_count=2,random_seed=318,verbose=False,allow_writing_files=False)
    m.fit(x.iloc[t],y[t],sample_weight=weights,cat_features=CATEGORICAL)
    return m.predict_proba(x.iloc[q])[:,1]


def run():
    y,hits,others=read_panel()
    hard=~hits.any(0)&~others.any(axis=(0,1))
    rows=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ["Name","Transported"])
                   for s in ["train","test"]],ignore_index=True)
    parties=raw.PassengerId.str.split("_").str[0]
    group=parties.iloc[rows].to_numpy()
    x=compact_features(raw).iloc[rows].reset_index(drop=True)
    x["BookingSize"]=parties.map(parties.value_counts()).iloc[rows].to_numpy()
    # Downweight entire bookings containing at least one member of each named cohort.
    # The null reassigns the corresponding intervention among equally sized training bookings.
    memberships=pd.DataFrame(dict(
        any_cryo=raw.CryoSleep.eq(True),earth=raw.HomePlanet.eq("Earth"),
        positive_personal=raw[["RoomService","Spa","VRDeck"]].gt(0).any(axis=1),
        positive_retail=raw[["FoodCourt","ShoppingMall"]].gt(0).any(axis=1),
        incomplete=raw[BILLS].isna().any(axis=1)))
    memberships=memberships.groupby(parties).max()
    full_size=parties.value_counts()
    out=ART/"nov23_training_cohort_sensitivity"
    out.mkdir(exist_ok=True)
    result={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,rows,y,fold)
        predictions={name:np.full(len(y),np.nan) for name in ["teacher_only","named_cohorts","random_cohorts"]}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            ids,grp=np.unique(group[t],return_inverse=True)
            assert not np.intersect1d(ids,group[q]).size
            sizes=full_size.loc[ids].to_numpy()
            mask=memberships.loc[ids].to_numpy()
            shuffled=mask.copy()
            rng=np.random.default_rng(481+int(k))
            for size in np.unique(sizes):
                pos=np.flatnonzero(sizes==size)
                shuffled[pos]=mask[rng.permutation(pos)]
            reference=logit(fit(x,y,t,q,np.ones(len(t))).clip(.005,.995))
            zq=logit(base[q].clip(.005,.995))
            initial=np.column_stack([np.ones(len(q)),zq,reference-zq])
            designs={"teacher_only":initial}
            for name,treatment in [("named_cohorts",mask),("random_cohorts",shuffled)]:
                contrasts=[]
                for j in range(mask.shape[1]):
                    down=np.where(treatment[grp,j],.25,1.)
                    p=fit(x,y,t,q,down)
                    contrasts.append(logit(p.clip(.005,.995))-reference)
                designs[name]=np.column_stack([initial,*contrasts])
            for a,b in GroupKFold(3,shuffle=True,random_state=117).split(q,groups=group[q]):
                assert not np.intersect1d(group[q[a]],group[q[b]]).size
                for name,design in designs.items():
                    penalty=np.full(design.shape[1],12.)
                    penalty[:2]=2.
                    predictions[name][q[b]]=offsets(design,zq,y[q],a,b,penalty)
        old=(base>=.5)==y
        metrics={}
        for name,p in predictions.items():
            assert np.isfinite(p).all()
            hit=(p>=.5)==y
            metrics[name]=dict(accuracy=float(hit.mean()),fixed=int((hit&~old).sum()),harmed=int((~hit&old).sum()),
                               persistent1070=int(hit[~hits.any(0)].sum()),unresolved466=int(hit[hard].sum()))
        result[seed]=metrics
        np.savez_compressed(out/f"{seed}.npz",row=rows,y=y,fold=fold,base=base,**predictions)
        print(seed,metrics,flush=True)
    write(ROOT/"reports/nov23_training_cohort_sensitivity/result.json",dict(results=result,cohorts=list(memberships.columns),
          definition="All interventions confined to outer training; the correction policy is calibrated on separate whole bookings within each outer evaluation fold."))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
