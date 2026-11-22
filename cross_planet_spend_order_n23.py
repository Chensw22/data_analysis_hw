"""Transfer within-stratum spending comparisons from Europa and Mars to Earth."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART,ROOT,SEEDS,panel,verify
from population_error_accounting_n23 import BILLS
from relational_error_corrector import write


def prepare(raw):
    bill=raw[BILLS].to_numpy(float)
    total=np.nansum(bill,axis=1)
    amount=np.nan_to_num(bill)
    logs=np.log1p(amount)
    shares=amount/(total[:,None]+1)
    x=np.column_stack([logs/5,np.maximum(logs-5,0)/2,np.maximum(logs-7,0),shares,
                       shares**2,np.isnan(bill),amount>0,
                       *[shares[:,i]*shares[:,j] for i in range(5) for j in range(i)]])
    strata=pd.DataFrame(dict(planet=raw.HomePlanet.fillna("unknown"),sleep=raw.CryoSleep.astype("string").fillna("?"),
                   age=np.digitize(raw.Age.fillna(-1),[0,13,21,40,60]),
                   total=np.digitize(total,[.5,300,750,1200,2200,4500]),
                   missing=np.isnan(bill).any(1)))
    return x, strata.groupby(list(strata),sort=False).ngroup().to_numpy()


def pairs(ids, cells, groups, seed, repeats):
    rng=np.random.default_rng(seed)
    left,right=[],[]
    for key in np.unique(cells[ids]):
        pool=ids[cells[ids]==key]
        for _ in range(repeats):
            seq=rng.permutation(pool)
            a,b=seq[:len(seq)//2],seq[len(seq)//2:2*(len(seq)//2)]
            ok=groups[a]!=groups[b]
            left.extend(a[ok]);right.extend(b[ok])
    return np.array(left,dtype=int),np.array(right,dtype=int)


def study(x,y,g,cells,t,seed):
    a,b=pairs(t,cells,g,seed,12)
    edges=np.unique(np.sort(np.column_stack([a,b]),axis=1),axis=0)
    edges=edges[y[edges[:,0]]!=y[edges[:,1]]]
    a,b=edges.T
    assert len(a)>0
    delta=x[a]-x[b]
    real=y[a]
    random=np.random.default_rng(852).permutation(real)
    pred=[]
    for target in (real,random):
        fit=LogisticRegression(C=.2,fit_intercept=False,max_iter=300).fit(delta,target)
        pred.append(fit.decision_function(x))
    return pred,len(a)


def calibrate(base,score,y,cal,q):
    cov=np.column_stack([np.ones(len(base)),logit(base.clip(.001,.999)),score])
    cov[:,2]=(cov[:,2]-cov[cal,2].mean())/max(cov[cal,2].std(),.1)
    offset=cov[:,1].copy()
    def loss(v):
        z=offset[cal]+cov[cal]@v
        pen=np.array([20.,60.,30.])
        return np.logaddexp(0,z).sum()-z@y[cal]+.5*(pen*v)@v,cov[cal].T@(expit(z)-y[cal])+pen*v
    fit=minimize(loss,[0.,0.,0.],jac=True,method="L-BFGS-B")
    assert fit.success
    return expit(offset[q]+cov[q]@fit.x)


def run():
    row,y,base_all,bank,_,_,_=panel()
    hit=(bank>=.5)==y[None,:,None]
    ref=(base_all>=.5)==y
    near=(hit.mean(1)>=ref.mean(1)[:,None]-.01).all(0)
    weak=~ref.any(0)&hit.all(0).any(1)&~hit.all(0)[:,near].any(1)
    deep=~ref.any(0)&~hit.any(axis=(0,2))
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"}).iloc[row].reset_index(drop=True)
    g=raw.PassengerId.str.split("_").str[0].to_numpy()
    earth=raw.HomePlanet.eq("Earth").to_numpy()
    other=raw.HomePlanet.isin(["Europa","Mars"]).to_numpy()
    x,cells=prepare(raw)
    out=ART/"nov23_cross_planet_spend_order"
    out.mkdir(exist_ok=True)
    results={}
    gains=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            base,fold=z["base"],z["fold"]
            verify(z,row,y,fold)
        names=["baseline_calibration","non_earth","earth","non_earth_sham"]
        pred={k:base.copy() for k in names}
        concord={k:[] for k in ["non_earth","earth","non_earth_sham","baseline"]}
        sizes=[]
        for f in np.unique(fold):
            t=np.flatnonzero(fold!=f)
            q=np.flatnonzero((fold==f)&earth)
            assert not np.intersect1d(g[t],g[q]).size
            source,ns=study(x,y,g,cells,t[other[t]],seed+int(f))
            dest,nd=study(x,y,g,cells,t[earth[t]],seed+int(f))
            scores={"non_earth":source[0],"earth":dest[0],"non_earth_sham":source[1],"baseline_calibration":np.zeros(len(y))}
            a,b=pairs(q,cells,g,seed+int(f),1)
            use=(y[a]!=y[b]) & ((x[a]!=x[b]).any(1))
            a,b=a[use],b[use]
            sizes.append([ns,nd,len(a)])
            for name,s in {**scores,"baseline":base}.items():
                if name in concord:
                    d=(s[a]-s[b])*np.where(y[a],1,-1)
                    concord[name].extend((np.where(d>1e-9,1.,np.where(d < -1e-9,0.,.5))).tolist())
            for fit,test in GroupKFold(3,shuffle=True,random_state=131).split(q,groups=g[q]):
                cal,held=q[fit],q[test]
                assert not np.intersect1d(g[cal],g[held]).size
                for name,s in scores.items():
                    pred[name][held]=calibrate(base,s,y,cal,held)
        result={}
        old=(base>=.5)==y
        for name,s in pred.items():
            ok=(s>=.5)==y
            fixes=ok&~old;harms=~ok&old
            result[name]=dict(accuracy=float(ok.mean()),earth_accuracy=float(ok[earth].mean()),
                 corrected_false_negatives=int((fixes&y).sum()),corrected_false_positives=int((fixes&~y).sum()),
                 new_false_negatives=int((harms&y).sum()),new_false_positives=int((harms&~y).sum()),
                 fixed=int(fixes.sum()),harmed=int(harms.sum()),earth_weak335=int((earth&weak&ok).sum()),
                 earth_deep252=int((earth&deep&ok).sum()))
        gains.append(((pred["non_earth"]>=.5)==y).astype(float)-((pred["baseline_calibration"]>=.5)==y))
        results[seed]=dict(scores=result,training_discordant_source_dest_and_test_pairs=sizes,
                           within_stratum_concordance={n:float(np.mean(v)) for n,v in concord.items()})
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**pred)
        print(seed,results[seed],flush=True)
    _,inv=np.unique(g,return_inverse=True)
    contribution=np.bincount(inv,weights=np.mean(gains,axis=0))
    count=np.bincount(inv)
    rng=np.random.default_rng(403)
    draws=[]
    for _ in range(100):
        choice=rng.integers(0,len(count),size=(30,len(count)))
        draws.extend(100*contribution[choice].sum(axis=1)/count[choice].sum(axis=1))
    write(ROOT/"reports/nov23_cross_planet_spend_order/result.json",dict(earth=int(earth.sum()),
          source_gain_over_calibration_pp=float(100*np.mean(gains)),booking_bootstrap_95_pp=np.quantile(draws,[.025,.975]).tolist(),
          earth_weak335=int((earth&weak).sum()),earth_deep252=int((earth&deep).sum()),results=results))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
