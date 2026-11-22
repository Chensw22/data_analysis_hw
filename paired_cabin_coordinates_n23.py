"""Estimate longitudinal deck alignment on cross-deck bookings and test its outcome signal."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS
from earth_bill_order_n23 import BILLS
from relational_error_corrector import write


def coordinate_maps(raw):
    group=raw.PassengerId.str.split("_").str[0].to_numpy()
    c=raw.Cabin.str.split("/",expand=True)
    num=pd.to_numeric(c[1]); track=(c[0]+c[2]).fillna("?")
    frac=((num+.5)/(num.groupby(track).transform("max")+1)).fillna(.5).to_numpy()
    categories,idx=np.unique(track,return_inverse=True)
    k=6
    segments=np.column_stack([np.ones(len(raw)),np.clip(k*frac[:,None]-np.arange(k),0,1)/k])
    basis=np.zeros((len(raw),len(categories),k+1))
    basis[np.arange(len(raw)),idx]=segments
    basis=basis.reshape(len(raw),-1)
    reps=pd.DataFrame(dict(group=group,track=track.to_numpy(),num=num)).dropna().drop_duplicates()
    pairs=[]
    for _,members in reps.groupby("group"):
        pairs.extend(combinations(members.index,2))
    a,b=np.asarray(pairs).T
    assert np.all(group[a]==group[b]) and np.all(idx[a]!=idx[b])
    target=np.tile([0.,*np.ones(k)],len(categories))
    regular=np.tile([.8,*np.full(k,.08)],len(categories))
    regular[np.repeat(np.isin(categories,["FP","FS"]),k+1)]=20
    def align(selected):
        matrix=np.vstack([basis[a[selected]]-basis[b[selected]],np.diag(regular)])
        response=np.r_[np.zeros(len(selected)),regular*target]
        lo=np.tile([-.15,*np.full(k,.15)],len(categories))
        hi=np.tile([.15,*np.full(k,3.)],len(categories))
        fit=lsq_linear(matrix,response,bounds=(lo,hi),tol=1e-9)
        assert fit.success
        return basis@fit.x
    hold_errors=[]
    for t,q in GroupKFold(5,shuffle=True,random_state=194).split(a,groups=group[a]):
        assert not np.intersect1d(group[a[t]],group[a[q]]).size
        v=align(t)
        hold_errors.extend(abs(v[a[q]]-v[b[q]]))
    learned=align(np.arange(len(a)))
    observed=track.ne("?").to_numpy()
    order=raw.PassengerId.str.split("_").str[0].astype(int).to_numpy()
    order=(order-order.min())/(order.max()-order.min())
    groups_per_cabin=pd.Series(group).groupby(raw.Cabin).nunique()
    stats=dict(pairs=len(a),bookings=len(np.unique(group[a])),raw_mae=float(np.mean(abs(frac[a]-frac[b]))),
               heldout_learned_mae=float(np.mean(hold_errors)),
               distinct_named_cabins=len(groups_per_cabin),cabins_reused_across_bookings=int((groups_per_cabin>1).sum()),
               correlation_with_booking_number=float(np.corrcoef(order[observed],learned[observed])[0,1]))
    return dict(fraction=frac,learned=learned,booking=order),stats


def features(raw,value):
    cabin=raw.Cabin.str.split("/",expand=True)
    deck=pd.get_dummies(cabin[0].fillna("?"),dtype=float).to_numpy()
    side=cabin[2].map({"P":-1,"S":1}).fillna(0).to_numpy()
    active=raw[BILLS].gt(0).any(axis=1).to_numpy(float)
    rest=np.column_stack([deck,side,active,side*active])
    if value is None:
        return rest
    # Piecewise linear longitudinal effects can differ by side, billing activity and deck.
    shapes=np.maximum(1-abs(10*value[:,None]-np.arange(11)),0)
    shapes*=raw.Cabin.notna().to_numpy()[:,None]
    return np.column_stack([rest,(rest[:,:,None]*shapes[:,None,:]).reshape(len(raw),-1)])


def correct(x,base,y,t,q):
    prior=logit(base.clip(.0001,.9999))
    train=x[t]
    def loss(w):
        eta=prior[t]+train@w
        return ((np.logaddexp(0,eta)-y[t]*eta).sum()+6*np.dot(w,w),
                train.T@(expit(eta)-y[t])+12*w)
    opt=minimize(loss,np.zeros(train.shape[1]),jac=True,method="L-BFGS-B")
    assert opt.success
    return expit(prior[q]+x[q]@opt.x)


def run():
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in {"Name","Transported"})
                   for s in ("train","test")],ignore_index=True)
    coordinates,stats=coordinate_maps(raw)
    with np.load(ART/f"nov23_group_kernel/{SEEDS[0]}.npz") as ref:
        row,y=ref["row"],ref["y"]
    raw=raw.iloc[row]
    bank={k:features(raw,v[row]) for k,v in coordinates.items()}
    bank["registration"]=features(raw,None)
    group=raw.PassengerId.str.split("_").str[0].to_numpy()
    earth=raw.HomePlanet.eq("Earth").to_numpy()
    data={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            data[seed]=z["base"],z["fold"]
    repeat=np.logical_and.reduce([(base>=.5)!=y for base,_ in data.values()])
    out=ART/"nov23_paired_cabin_coordinates"
    out.mkdir(exist_ok=True)
    result={}
    for seed,(base,fold) in data.items():
        estimates={k:np.full(len(row),np.nan) for k in bank}
        for f in np.unique(fold):
            held=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=705).split(held,groups=group[held]):
                t,q=held[a],held[b]
                assert not np.intersect1d(group[t],group[q]).size
                for k,x in bank.items():
                    estimates[k][q]=correct(x,base,y,t,q)
        with np.load(ART/f"nov23_identical_bills/{seed}.npz") as z:
            identical=z["eligible"]
            np.testing.assert_array_equal(z["row"],row)
        before=(base>=.5)==y
        scores={}
        for k,p in estimates.items():
            assert np.isfinite(p).all()
            ok=(p>=.5)==y
            scores[k]=dict(accuracy=float(ok.mean()),fixed=int((ok&~before).sum()),harmed=int((~ok&before).sum()),
                           repeat=int((ok&repeat).sum()),earth430=int((ok&earth&repeat&identical).sum()))
        result[seed]=scores
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**estimates)
    write(ROOT/"reports/nov23_paired_cabin_coordinates/result.json",dict(coordinates=stats,results=result))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
