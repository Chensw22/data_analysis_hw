"""Train with interpolated passengers matched on observable state and independent booking."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from booking_subsample_rules_n23 import ART, ROOT, SEEDS
from exaone_validation_data import CATEGORICAL, compact_features
from grouped_five_way_coverage_n23 import verify
from nested_manifest_expressions_n23 import BILLS
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write


def donors(x, group, rng):
    bills = x[BILLS]
    positive = bills.gt(0).any(axis=1).to_numpy()
    budget = np.log1p(bills.to_numpy())
    quant = np.column_stack([budget,x[["Age","CabinNumber","PartySize"]].to_numpy()])
    standardized = StandardScaler().fit_transform(quant)
    cat = OneHotEncoder(sparse_output=False).fit_transform(x[CATEGORICAL])
    coords = np.column_stack([cat,np.nan_to_num(standardized).clip(-4,4),np.isnan(quant)])
    near = np.arange(len(x))
    arbitrary = near.copy()
    # Exact matching keeps cryosleep state and presence of spending consistent.
    # An unmatched passenger yields one extra unmodified copy in both controls.
    strata = pd.DataFrame(dict(cryo=x.CryoSleep.to_numpy(),positive=positive))
    for indexes in strata.groupby(["cryo","positive"]).indices.values():
        if len(np.unique(group[indexes]))<2:
            continue
        knn = NearestNeighbors(n_neighbors=min(24,len(indexes)),n_jobs=1).fit(coords[indexes])
        _, ids = knn.kneighbors(coords[indexes])
        for local, i in enumerate(indexes):
            allowed = indexes[group[indexes]!=group[i]]
            arbitrary[i] = rng.choice(allowed)
            closest = indexes[ids[local]]
            feasible = closest[group[closest]!=group[i]][:8]
            near[i] = rng.choice(feasible) if len(feasible) else arbitrary[i]
    return near,arbitrary


def interpolate(x, y, donor, own):
    synthetic = x.reset_index(drop=True).copy()
    # Keep a single person's nominal fields, actual cabin and observed missingness pattern.
    for col in ["Age",*BILLS]:
        v = x[col].to_numpy(float)
        w = v[donor]
        known = np.isfinite(v)&np.isfinite(w)
        if col in BILLS:
            mixed = np.expm1(own*np.log1p(v)+(1-own)*np.log1p(w))
        else:
            mixed = own*v+(1-own)*w
        synthetic[col] = np.where(known,mixed,v)
    labels = own*y+(1-own)*y[donor]
    return synthetic,labels


def train(x,y,examples,targets,q):
    net=CatBoostClassifier(iterations=400,depth=6,learning_rate=.05,l2_leaf_reg=6,loss_function="CrossEntropy",
                           random_seed=311,thread_count=2,verbose=False,allow_writing_files=False)
    net.fit(pd.concat([x,examples],ignore_index=True),np.r_[y,targets],cat_features=CATEGORICAL,
            sample_weight=np.r_[np.ones(len(x)),np.full(len(x),.5)])
    return net.predict_proba(q)[:,1]


def run():
    y, base_hits, challengers = read_panel()
    hard = ~base_hits.any(0)&~challengers.any(axis=(0,1))
    row = pd.read_csv(ART/"nov23_global_error_inventory.csv",usecols=["row"]).row.to_numpy()
    raw = pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ["Name","Transported"])
                     for s in ["train","test"]],ignore_index=True)
    groups = raw.PassengerId.str.split("_").str[0]
    x = compact_features(raw).assign(PartySize=groups.map(groups.value_counts())).iloc[row].reset_index(drop=True)
    groups = groups.iloc[row].to_numpy()
    out=ART/"nov23_booking_mixup"
    out.mkdir(exist_ok=True)
    results={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,row,y,fold)
        probs={name:np.full(len(y),np.nan) for name in ["copy_control","random","neighbor"]}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(groups[t],groups[q]).size
            rng=np.random.default_rng(113+int(k))
            xt=x.iloc[t].reset_index(drop=True)
            labels=y[t].astype(float)
            near,arbitrary=donors(xt,groups[t],rng)
            own=rng.uniform(.5,1.,len(t))
            probs["copy_control"][q]=train(xt,labels,xt,labels,x.iloc[q])
            for name,d in [("random",arbitrary),("neighbor",near)]:
                features,target=interpolate(xt,labels,d,own)
                probs[name][q]=train(xt,labels,features,target,x.iloc[q])
        old=(base>=.5)==y
        metrics={}
        for name,p in probs.items():
            assert np.isfinite(p).all()
            for key,score in [(name,p),(name+"_blend",.75*base+.25*p)]:
                hit=(score>=.5)==y
                metrics[key]=dict(accuracy=float(hit.mean()),fixed=int((hit&~old).sum()),
                                  harmed=int((~hit&old).sum()),persistent1070=int(hit[~base_hits.any(0)].sum()),
                                  unresolved466=int(hit[hard].sum()))
        results[seed]=metrics
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**probs)
        print(seed,metrics,flush=True)
    write(ROOT/"reports/nov23_booking_mixup/result.json",dict(results=results,synthetic_weight=.5))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
