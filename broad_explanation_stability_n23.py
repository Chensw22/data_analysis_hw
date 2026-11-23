"""Test whether a stable decision conceals unstable evidence across training bookings."""
import numpy as np
from catboost import CatBoostClassifier,Pool
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS,load
from broad_manifest_vector_ranks_n23 import REPORT
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_release_miss_coverage_n23 import summarize
from broad_twin_surfaces_n23 import scopes
from exaone_validation_data import CATEGORICAL

DOMAINS=[BILLS,["CryoSleep"],["Deck","Side","CabinNumber"],["HomePlanet","Destination","VIP"],["Age","BookingSize"]]


def replicas(frame,y,g,fold):
    logits=np.empty((len(y),4))
    parts=np.empty((len(y),4,len(DOMAINS)))
    assert sorted(sum(DOMAINS,[]))==sorted(frame.columns)
    for f in np.unique(fold):
        t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
        query=Pool(frame.iloc[q],cat_features=CATEGORICAL)
        for j,seed in enumerate((585,769)):
            for k,(_,a) in enumerate(GroupKFold(2,shuffle=True,random_state=seed).split(t,groups=g[t])):
                learn=t[a]
                assert not np.intersect1d(g[learn],g[q]).size
                fit=CatBoostClassifier(iterations=230,depth=6,learning_rate=.075,
                       random_seed=379,thread_count=2,verbose=False,allow_writing_files=False)
                fit.fit(Pool(frame.iloc[learn],y[learn],cat_features=CATEGORICAL))
                shap=fit.get_feature_importance(query,type="ShapValues",thread_count=2)
                raw=fit.predict(query,prediction_type="RawFormulaVal")
                np.testing.assert_allclose(shap.sum(1),raw,atol=1e-8)
                logits[q,2*j+k]=raw
                for b,cols in enumerate(DOMAINS):
                    parts[q,2*j+k,b]=shap[:,frame.columns.get_indexer(cols)].sum(1)
    return logits,parts


def run():
    row,y,frame,g,_=load()
    masks=scopes(y,frame)
    history,refs={},[]
    dest=ART/"nov23_explanation_stability"
    dest.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            fold,base=z["fold"],z["base"]
        odds=logit(base.clip(.001,.999))
        total,parts=replicas(frame,y,g,fold)
        mean=total.mean(1)
        sign=np.where(base>=.5,1.,-1.)
        # Fix the sum of the centered domain contributions at zero for each
        # replica. Dispersion of these contrasts does not equal total spread.
        centered=parts-parts.mean(2,keepdims=True)
        variation=centered.std(1)
        avg=centered.mean(1)
        ref=(base>=.5)==y
        refs.append(ref)
        designs={}
        means=np.c_[odds,mean,total.std(1)*sign,(total.min(1)+total.max(1))/2]
        first=np.c_[means,avg]
        for f in np.unique(fold):
            ids=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
                t,q=ids[a],ids[b]
                assert not np.intersect1d(g[t],g[q]).size
                shuffled=variation.copy()
                def strata(arr,n):
                    return np.digitize(arr,np.quantile(arr[t],np.arange(1,n)/n))
                bins=(strata(odds,6)*4+strata(mean,4))*3+strata(total.std(1),3)
                for ix in (t,q):
                    rng=np.random.default_rng(397)
                    for c in np.unique(bins[ix]):
                        cell=ix[bins[ix]==c]
                        shuffled[cell]=variation[rng.permutation(cell)]
                for key,design in (("scores",means),("mean_evidence",first),
                    ("stability",np.c_[first,variation*sign[:,None]]),("shuffled",np.c_[first,shuffled*sign[:,None]])):
                    sd=design[t].std(0).clip(.1)
                    model=LogisticRegression(C=.25,max_iter=400).fit(design[t]/sd,y[t])
                    designs.setdefault(key,np.zeros(len(y)))[q]=model.predict_proba(design[q]/sd)[:,1]
        for key,p in designs.items():
            assert ((p<1)&(p>0)).all()
            history.setdefault(key,[]).append((p>=.5)==y)
        np.savez_compressed(dest/f"{seed}.npz",row=row,y=y,base=base,fold=fold,total=total,
                            parts=parts,**designs)
        print(seed,[(k,int((v[-1]&~ref).sum()),int((~v[-1]&ref).sum())) for k,v in history.items()],flush=True)
    write(REPORT/"stability.json",{k:summarize(np.stack(v),np.stack(refs),masks) for k,v in history.items()})


if __name__=="__main__":
    with threadpool_limits(2):
        run()
