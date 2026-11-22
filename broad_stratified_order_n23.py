"""Measure how much observable passenger context relaxes conflicts between model rankings."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from broad_ordered_ensemble_n23 import ART,OUT,REPORT,SEEDS,load,historical,monotone_rule,apply_rule,interval,write
from broad_geographic_transfer_n23 import BILLS


def contexts(frame):
    spend=frame[BILLS].to_numpy()
    state=np.where(np.any(spend>0,axis=1),"positive",np.where(np.isnan(spend).any(1),"unknown","zero"))
    coarse=pd.MultiIndex.from_arrays([frame.HomePlanet,state])
    fine=pd.MultiIndex.from_arrays([frame.HomePlanet,state,frame.CryoSleep,frame.Side,
                                     np.where(frame.Age.isna(),"missing",np.where(frame.Age<13,"child","older"))])
    rng=np.random.default_rng(449)
    perm=rng.permutation(len(frame))
    result={}
    for name,c in (("origin_spend",coarse),("origin_spend_sleep_side_age",fine)):
        codes=pd.factorize(c)[0]
        result[name]=codes
        result["shuffled_"+name]=codes[perm]
    return result


def stratified(x,y,category,t,q):
    pred=x[q,0]>=.5
    min_errors=0
    for key in np.unique(category[t]):
        fit=t[category[t]==key]
        subset=category[q]==key
        pos,neg,n=monotone_rule(x[fit],y[fit])
        min_errors+=n
        if subset.any():
            pred[subset]=apply_rule(pos,neg,x[q[subset]])
    return pred,min_errors


def run():
    row,y,frame,groups,_=load()
    orig,target,masks=historical()
    np.testing.assert_array_equal(orig,row)
    np.testing.assert_array_equal(target,y)
    strata=contexts(frame)
    history,records={},{}
    for name in [*map(str,SEEDS),"v131"]:
        cols=[]
        for folder,field in (("broad_passenger_differences/balanced","native_pair"),
                             ("broad_count_supervision","native_similar"),("broad_latent_scale","native_real")):
            with np.load(ART/f"nov23_{folder}/{name}.npz") as z:
                for k,v in (("row",row),("y",y)):
                    np.testing.assert_array_equal(z[k],v)
                if cols:
                    for k,v in (("fold",fold),("ref",ref),("base",base)):
                        np.testing.assert_array_equal(z[k],v)
                fold,ref,base=z["fold"],z["ref"],z["base"]
                cols.append(z[field])
        x=np.column_stack([base,*cols])
        locked=ref!=(base>=.5)
        live=np.flatnonzero(~locked)
        results={}
        bank={}
        for key,category in strata.items():
            oracle,n=stratified(x,y,category,live,live)
            assert (oracle!=y[live]).sum()==n
            n+=int((locked&(ref!=y)).sum())
            pred=ref.copy()
            for f in np.unique(fold):
                idx=np.flatnonzero((fold==f)&~locked)
                for a,b in GroupKFold(3,shuffle=True,random_state=823).split(idx,groups=groups[idx]):
                    assert not np.intersect1d(groups[idx[a]],groups[idx[b]]).size
                    pred[idx[b]],_=stratified(x,y,category,idx[a],idx[b])
            hit=pred==y
            gain=hit.astype(int)-(ref==y)
            results[key]=dict(oracle_accuracy=1-n/len(y),minimum_errors=n,accuracy=float(hit.mean()),
                              cells=int(len(np.unique(category))),fixed=int((gain>0).sum()),harmed=int((gain<0).sum()))
            bank[key]=pred
            if name!="v131":
                history.setdefault(key,[]).append((hit,gain))
        records[name]=results
        np.savez_compressed(OUT/f"strata_{name}.npz",row=row,y=y,base=base,ref=ref,fold=fold,**bank)
        print(name,[(s,v["minimum_errors"],v["fixed"]-v["harmed"]) for s,v in results.items()],flush=True)
    stable={}
    for k,values in history.items():
        ok=np.logical_and.reduce([h for h,_ in values])
        stable[k]={s:int((ok&m).sum()) for s,m in masks.items()}
        stable[k]["paired_gain_ci95_pp"]=interval(np.mean([d for _,d in values],0),groups)
    write(REPORT/"stratified.json",dict(plans=records,stable=stable))


if __name__=="__main__":
    run()
