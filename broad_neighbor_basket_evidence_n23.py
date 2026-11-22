"""Look for outcome evidence in the service choices of other, unlabeled passengers."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_service_assignment_n23 import (ART,ROOT,SEEDS,BILLS,encode,subset,fit_choice,likelihood,
                                         historical,load,strata,evaluate,write,calibrated)
from exaone_validation_data import compact_features

REPORT=ROOT/"reports/nov23_broad_basket_balance"


def neighborhood(frame,party,targets,pool):
    # Group members never serve as geographic neighbors of each other.
    by_party=pd.Series(pool).groupby(party[pool]).apply(np.asarray).to_dict()
    party_edges=[]
    geographic=[]
    number=frame.CabinNumber.to_numpy(float)
    valid=np.isfinite(number)&(frame.Side!="__MISSING__").to_numpy()&(frame.Deck!="__MISSING__").to_numpy()
    region=(frame.Deck.astype(str)+"/"+frame.Side.astype(str)).to_numpy()
    locations={}
    for key in np.unique(region[pool]):
        ix=pool[valid[pool]&(region[pool]==key)]
        locations[key]=ix[np.argsort(number[ix],kind="stable")]
    for i in targets:
        same=by_party[party[i]]
        party_edges.append(same[same!=i])
        ids=locations[region[i]]
        near=np.empty(0,int)
        if valid[i] and len(ids):
            pos=np.searchsorted(number[ids],number[i])
            cand=ids[max(0,pos-48):pos+48]
            cand=cand[party[cand]!=party[i]]
            if len(cand):
                distance=abs(number[cand]-number[i])
                radius=np.partition(distance,min(15,len(cand)-1))[min(15,len(cand)-1)]
                near=cand[distance<=radius]
        geographic.append(near)
    return party_edges,geographic


def neighbors(values,edges):
    out=np.zeros((len(edges[0]),4))
    for k,pools in enumerate(edges):
        for j,ids in enumerate(pools):
            if len(ids):
                out[j,2*k:2*k+2]=values[ids].mean(),np.log1p(len(ids))
    return out


def run():
    row,y,local,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda n:n not in ("Name","Transported"))
                   for s in ("train","test")],ignore_index=True)
    group=raw.PassengerId.str.split("_").str[0].astype(int)
    frame=compact_features(raw)
    frame["BookingSize"]=group.map(group.value_counts())
    party=group.to_numpy()
    np.testing.assert_array_equal(g,party[row])
    assert frame.iloc[row].reset_index(drop=True).equals(local)
    _,parts=encode(frame)
    paid=strata(local)==2
    scopes=dict(**masks,repeat_paid=masks["repeat"]&paid&y)
    # Restrict randomization within planet, deck, payment status and booking size.
    codes=pd.factorize(pd.MultiIndex.from_arrays([frame.HomePlanet,frame.Deck,
           strata(frame),np.minimum(frame.BookingSize,3)]))[0]
    records,histories={},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_service_assignment/{name}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            fold,base,ref,core=[z[k] for k in ("fold","base","ref","unordered")]
            local_ratio=z["real"]
        entries={k:np.zeros((len(y),4)) for k in ("real","fake")}
        own=np.zeros(len(y))
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            pool=np.flatnonzero(~np.isin(party,g[t]))
            assert np.isin(row[q],pool).all()
            events=subset(parts,pool)
            ll=[likelihood(events,fit_choice(parts,row[t[y[t]==label]]),len(frame)) for label in (False,True)]
            signal=ll[1]-ll[0]
            own[q]=signal[row[q]]
            edges=neighborhood(frame,party,row[q],pool)
            entries["real"][q]=neighbors(signal,edges)
            sham=signal.copy()
            rng=np.random.default_rng(219+int(f))
            for cell in np.unique(codes[pool]):
                ix=pool[codes[pool]==cell]
                sham[ix]=rng.permutation(signal[ix])
            entries["fake"][q]=neighbors(sham,edges)
        # Different BLAS rows/extra all-zero schema columns can move the optimizer slightly.
        drift=float(np.max(abs(own-local_ratio)))
        anchor=np.column_stack([logit(p.clip(.001,.999)) for p in (base,core)]+[own])
        scores={}
        for k,v in (("own",np.empty((len(y),0))),("size",entries["real"][:,[1,3]]),
                    ("real",entries["real"]),("fake",entries["fake"])):
            ds=np.column_stack([anchor,v])
            p=base.copy()
            for f in np.unique(fold):
                ids=np.flatnonzero((fold==f)&paid)
                for a,b in GroupKFold(3,shuffle=True,random_state=412).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    p[q]=calibrated(ds[t],ds[q],y,t)
            scores["update_"+k]=p
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",scopes)
        for k,(hit,diff) in hits.items():
            result[k]["paid_fn_fixed"]=int((paid&y&(diff>0)).sum())
            result[k]["paid_fp_added"]=int((paid&~y&(diff<0)).sum())
            if name!="v131":
                histories.setdefault(k,[]).append(hit)
        records[name]=dict(models=result,repeat_paid_with_companions=int((scopes["repeat_paid"]&(entries["real"][:,1]>0)).sum()),
                           repeat_paid_with_geographic=int((scopes["repeat_paid"]&(entries["real"][:,3]>0)).sum()),
                           own_likelihood_max_difference=drift)
        print(name,drift,result["update_real"]["accuracy"],flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in scopes.items()} for k,v in histories.items()}
    write(REPORT/"neighbors.json",dict(plans=records,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
