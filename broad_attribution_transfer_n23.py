"""Test whether the reasons for a tree's prediction distinguish wrong and right decisions."""
import numpy as np
from catboost import CatBoostClassifier,Pool
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS,load
from broad_manifest_vector_ranks_n23 import REPORT
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_release_miss_coverage_n23 import summarize
from broad_twin_surfaces_n23 import scopes
from exaone_validation_data import CATEGORICAL


def tree_parts(frame,y,groups,fold):
    phi=np.empty((len(y),frame.shape[1]))
    raw=np.empty(len(y))
    for f in np.unique(fold):
        q=np.flatnonzero(fold==f)
        t=np.flatnonzero((fold!=f)&~np.isin(groups,groups[q]))
        assert not np.intersect1d(groups[t],groups[q]).size
        model=CatBoostClassifier(iterations=350,depth=6,learning_rate=.07,verbose=False,
                 thread_count=2,random_seed=139,allow_writing_files=False)
        model.fit(Pool(frame.iloc[t],y[t],cat_features=CATEGORICAL))
        query=Pool(frame.iloc[q],cat_features=CATEGORICAL)
        contrib=model.get_feature_importance(query,type="ShapValues",thread_count=2)
        raw[q]=model.predict(query,prediction_type="RawFormulaVal")
        np.testing.assert_allclose(contrib.sum(1),raw[q],atol=1e-8)
        phi[q]=contrib[:,:-1]
    return phi,raw


def compose(groups,fold,y,base,raw,phi):
    base_z=logit(base.clip(.001,.999))
    sign=np.sign(base_z)
    # Remove the mean contribution so feature identity cannot transmit the
    # tree total a second time; the true total is already in every model.
    centered=phi-phi.mean(1,keepdims=True)
    misc=np.c_[base_z,raw]
    stats={}
    for f in np.unique(fold):
        idx=np.flatnonzero(fold==f)
        for a,b in GroupKFold(3,shuffle=True,random_state=925).split(idx,groups=groups[idx]):
            t,q=idx[a],idx[b]
            assert not np.intersect1d(groups[t],groups[q]).size
            shuffled=centered.copy()
            for ids in (t,q):
                # Shuffle the whole centered vector among similarly scoring people.
                bins=np.digitize(base_z[ids],np.quantile(base_z[t],np.arange(1,7)/7))*7
                bins+=np.digitize(raw[ids],np.quantile(raw[t],np.arange(1,7)/7))
                rng=np.random.default_rng(487)
                for c in np.unique(bins):
                    seg=ids[bins==c]
                    shuffled[seg]=centered[rng.permutation(seg)]
            for k,z in (("scores",None),("attributes",centered),("shuffled",shuffled)):
                inputs=misc if z is None else np.c_[misc,z, np.abs(z)*sign[:,None]]
                scale=inputs[t].std(0).clip(.2)
                fit=LogisticRegression(C=.18,max_iter=500).fit(inputs[t]/scale,y[t])
                stats.setdefault(k,np.empty(len(y)))[q]=fit.predict_proba(inputs[q]/scale)[:,1]
    return stats


def matched_parts(base,raw,parts,repeat,y):
    # Compare persistent errors with predictions that were right this time,
    # matching both the direction and score range of the benchmark and the tree.
    def bucket(score):
        return np.digitize(score,[.1,.2,.3,.4,.5,.6,.7,.8,.9])
    bins=bucket(base)*10+bucket(expit(raw))
    reference=(base>=.5)==y
    orientation=np.where(base>=.5,1.,-1.)
    value=parts*orientation[:,None]
    out={}
    for direction in (False,True):
        diffs=[];weights=[]
        for c in np.unique(bins):
            mask=(bins==c)&((base>=.5)==direction)
            case=mask&repeat
            control=mask&reference
            if case.any() and control.any():
                diffs.append(value[case].mean(0)-value[control].mean(0))
                weights.append(int(case.sum()))
        out[str(direction)]=dict(matched=sum(weights),difference=np.average(diffs,axis=0,weights=weights).tolist())
    return out


def main():
    row,y,frame,g,_=load()
    masks=scopes(y,frame)
    history,refs,diagnostics={},[],{}
    fields={"bills":BILLS,"sleep":["CryoSleep"],"cabin":["Deck","Side","CabinNumber"],
            "registration":["HomePlanet","Destination","VIP"],"age_party":["Age","BookingSize"]}
    assert sorted(sum(fields.values(),[]))==sorted(frame.columns)
    dest=ART/"nov23_attribution_transfer"
    dest.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            base,fold=z["base"],z["fold"]
        phi,raw=tree_parts(frame,y,g,fold)
        part=np.column_stack([phi[:,frame.columns.get_indexer(cols)].sum(1) for cols in fields.values()])
        np.testing.assert_allclose(part.sum(1),phi.sum(1),atol=1e-8)
        diag=matched_parts(base,raw,part,masks["repeat"],y)
        out=compose(g,fold,y,base,raw,phi)
        out["tree"]=expit(raw)
        ref=(base>=.5)==y
        refs.append(ref)
        for k,p in out.items():
            assert np.isfinite(p).all()
            history.setdefault(k,[]).append((p>=.5)==y)
        np.savez_compressed(dest/f"{seed}.npz",row=row,y=y,base=base,fold=fold,phi=phi,**out)
        print(seed,[(k,int((h[-1]&~ref).sum()),int((~h[-1]&ref).sum())) for k,h in history.items()],flush=True)
        diagnostics[str(seed)]=diag
    write(REPORT/"tree_attributions.json",dict(domains=list(fields),matched=diagnostics,
        results={k:summarize(np.stack(h),np.stack(refs),masks) for k,h in history.items()}))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
