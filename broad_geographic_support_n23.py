"""Match the geometry of Kaggle test bookings to random and geographically blocked validations."""
import numpy as np
import pandas as pd
from broad_geographic_transfer_n23 import ART,ROOT,REPORT,SEEDS,load
from broad_booking_kriging_n23 import geometry
from broad_dispute_leaf_evidence_n23 import cohorts
from relational_error_corrector import write

def nearest(query,training):
    if not len(training):
        return np.full(len(query),np.inf)
    pool=np.sort(np.unique(training))
    j=np.searchsorted(pool,query)
    return np.minimum(abs(query-pool[np.maximum(0,j-1)]),abs(query-pool[np.minimum(len(pool)-1,j)]))


def distances(known,query,geo):
    train=np.unique(known)
    assert not np.intersect1d(train,query).size
    d_booking=nearest(query,train)
    tg=geo.reindex(train)
    qg=geo.reindex(query)
    a=(tg.deck+"-"+tg.side).fillna("?").to_numpy()
    b=(qg.deck+"-"+qg.side).fillna("?").to_numpy()
    dc=np.full(len(query),np.nan)
    for r in np.unique(b[b!="?"]):
        mask=b==r
        dc[mask]=nearest(qg["median"].to_numpy()[mask],tg["median"].to_numpy()[a==r])
    return d_booking,dc


def stats(a,b):
    geo=~np.isnan(b)
    finite=np.isfinite(b)
    return dict(rows=len(a),booking_distance_median_p90=np.quantile(a,[.5,.9]).tolist(),
                booking_within_3=float((a<=3).mean()),observed_cabin=int(geo.sum()),
                cabin_region_absent=int((geo&~finite).sum()),cabin_within_3_given_known=float((b[geo]<=3).mean()),
                cabin_distance_median_p90_given_represented=np.quantile(b[finite],[.5,.9]).tolist() if finite.any() else None)


def run():
    rows,y,_,g,regions=load()
    masks=cohorts(rows,y)
    a,b=[pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=["PassengerId"]).PassengerId.str.split("_").str[0].astype(int).to_numpy() for s in ("train","test")]
    geo=geometry()
    full=distances(a,b,geo)
    result={"official_test_from_all_training_ids":stats(*full),"validation":{}}
    ids=np.arange(len(y))
    for name in [*map(str,SEEDS),"order","cabin"]:
        with np.load(ART/f"nov23_broad_geographic_transfer/{name}.npz") as z:
            fold=z["fold"]
            np.testing.assert_array_equal(rows,z["row"])
            np.testing.assert_array_equal(y,z["y"])
        one=np.full(len(y),np.nan);two=one.copy()
        test_comparison=[]
        for k in np.unique(fold):
            t,q=ids[fold!=k],ids[fold==k]
            one[q],two[q]=distances(g[t],g[q],geo)
            test_comparison.append(stats(*distances(g[t],b,geo)))
        result["validation"][name]=dict(all=stats(one,two),repeat=stats(one[masks["repeat"]],two[masks["repeat"]]),
             consistently_right=stats(one[~masks["ever"]],two[~masks["ever"]]),official_test_from_same_training=test_comparison)
    write(REPORT/"distance_to_support.json",result)
    print(result["official_test_from_all_training_ids"],flush=True)


if __name__=="__main__":
    run()
