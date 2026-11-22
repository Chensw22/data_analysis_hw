"""Can out-of-booking calibration isolate evidence specific to real companion-derived directions?"""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART,ROOT,SEEDS,panel,predictions
from broad_booking_policy_n23 import EXTRA
from broad_quantile_forest_n23 import distribution, BILLS,CATEGORICAL,compact_features
from relational_error_corrector import write


def run():
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Transported","Name"))
    result={}
    hits={}
    references=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_broad_shared_passenger_modes/{seed}.npz") as z:
            row,y,fold,base=[z[n] for n in ("row","y","fold","base")]
            modes=np.column_stack([z[n] for n in ("own","fake_only","pca_only","real_only")])
        frame=compact_features(raw.iloc[row]).reset_index(drop=True)
        group=raw.PassengerId.str.split("_").str[0].to_numpy()[row]
        bills=frame[BILLS].fillna(-1).to_numpy(float)
        # Same label-free registration fields and missing-value convention as
        # the standalone class-conditional bill distribution experiment.
        registry=pd.get_dummies(frame.drop(columns=BILLS),columns=CATEGORICAL,dtype=float).fillna(-1).to_numpy()
        inp=logit(np.column_stack([base,modes]).clip(.002,.998))
        measures={}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(group[t],group[q]).size
            pooled,neg,pos=distribution(registry,bills,y,group,t,q)
            bill=np.column_stack([pooled,pos-neg])
            for a,b in GroupKFold(3,shuffle=True,random_state=481).split(q,groups=group[q]):
                assert not np.intersect1d(group[q[a]],group[q[b]]).size
                # Fit the bill readout on the calibration bookings themselves.
                # Stacking cached OOF predictions with different calibration
                # folds would allow their training labels to cross the boundary.
                mu=bill[a].mean(0)
                sd=bill[a].std(0).clip(.05)
                v=((bill-mu)/sd).clip(-7,7)
                controls=inp[q,:4]
                shared=inp[q,4]
                residual=shared-Ridge(alpha=3).fit(controls[a],shared[a]).predict(controls)
                designs={"controls":controls,"plus_shared":np.column_stack([controls,residual]),
                         "plus_bill":np.column_stack([controls,v]),
                         "joint":np.column_stack([controls,residual,v])}
                for kind,x in designs.items():
                    model=LogisticRegression(C=.03,max_iter=300).fit(x[a],y[q[a]])
                    p=model.predict_proba(x[b])[:,1]
                    measures.setdefault(kind,np.full(len(y),np.nan))[q[b]]=p
        for k,p in measures.items():
            assert np.isfinite(p).all()
            hits.setdefault(k,[]).append((p>=.5)==y)
        references.append((base>=.5)==y)
    ref=np.stack(references)
    old_rows,old_y,_,old,_,_,_=panel()
    np.testing.assert_array_equal(old_rows,row)
    np.testing.assert_array_equal(old_y,y)
    historical=np.concatenate([old,np.stack([predictions(row,y,s,EXTRA)[1] for s in SEEDS])],axis=2)
    old_hit=(historical>=.5)==y[None,:,None]
    never=~ref.any(0)&~old_hit.any((0,2))
    without=~ref.any(0)&~old_hit.all(0).any(1)
    recovered=np.zeros(len(y),bool)
    for folder,field in [("broad_shared_passenger_modes","real_only"),("broad_quantile_forest","standalone")]:
        stable=np.ones(len(y),bool)
        for seed in SEEDS:
            with np.load(ART/f"nov23_{folder}/{seed}.npz") as z:
                np.testing.assert_array_equal(z["row"],row)
                np.testing.assert_array_equal(z["y"],y)
                stable &= (z[field]>=.5)==y
        recovered |= stable
    new=without&recovered
    assert new.sum()==50 and never.sum()==212
    for k,v in hits.items():
        good=np.stack(v)
        result[k]=dict(accuracy=good.mean(1).tolist(),fixed=(good&~ref).sum(1).tolist(),
            harmed=(~good&ref).sum(1).tolist(),new50_stable=int((new&good.all(0)).sum()),
            never212_stable=int((never&good.all(0)).sum()),repeat_stable=int((good.all(0)&~ref.any(0)).sum()),
            at_least_one_base_error_stable=int((good.all(0)&~ref.all(0)).sum()))
    write(ROOT/"reports/nov23_broad_shared_passenger_modes/calibration.json",result)
    print(result,flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
