"""Generate full class-conditional passengers by sequential distributional forest sampling."""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART,ROOT,SEEDS
from booking_mixup_n23 import train
from exaone_validation_data import compact_features,CATEGORICAL
from grouped_five_way_coverage_n23 import verify
from nested_manifest_expressions_n23 import BILLS
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write

ANCHOR=["HomePlanet","Deck","CabinNumber","Side","PartySize"]
PROFILE=["Age","CryoSleep","Destination","VIP"]


def numeric(train,query):
    joined=pd.concat([train,query],ignore_index=True)
    cats=[c for c in train if c in CATEGORICAL]
    values=pd.get_dummies(joined,columns=cats).astype(float)
    values=values.fillna(-1)
    return values.iloc[:len(train)].to_numpy(),values.iloc[len(train):].to_numpy()


def draw(features,synthetic,response,rng):
    fs,qs=numeric(features,synthetic)
    target=pd.get_dummies(response,columns=[c for c in response if c in CATEGORICAL]).astype(float).fillna(-1)
    # Joint output normalization makes both rare categorical attributes and continuous
    # targets visible to the tree splits. Each drawn block is a real donor observation.
    z=target.to_numpy()
    z=(z-z.mean(0))/z.std(0).clip(.15)
    forest=ExtraTreesRegressor(n_estimators=24,max_depth=7,min_samples_leaf=12,max_features=.85,
                               n_jobs=2,random_state=int(rng.integers(2**30))).fit(fs,z)
    assignments=rng.integers(len(forest.estimators_),size=len(qs))
    donors=np.empty(len(qs),int)
    for i,est in enumerate(forest.estimators_):
        ids=np.flatnonzero(assignments==i)
        if not len(ids):continue
        train_leaf,query_leaf=est.apply(fs),est.apply(qs[ids])
        for key in np.unique(query_leaf):
            q=ids[query_leaf==key]
            candidates=np.flatnonzero(train_leaf==key)
            donors[q]=rng.choice(candidates,len(q),replace=True)
    return donors


def generate(x,y,seed):
    rng=np.random.default_rng(seed)
    generated,shuffled=x.copy(),x.copy()
    total_transfers=0
    for label in [False,True]:
        idx=np.flatnonzero(y==label)
        local=x.iloc[idx].reset_index(drop=True)
        # Keep category frequencies and priors exact with one example per original
        # anchor. Attributes supplied to the sampler contain no identifiers or labels.
        cases=local.copy()
        profile=draw(local[ANCHOR],cases[ANCHOR],local[PROFILE],rng)
        cases[PROFILE]=local.iloc[profile][PROFILE].to_numpy()
        expenses=local[BILLS].to_numpy(float)
        state=np.where(np.isnan(expenses),2,(expenses>0).astype(int))
        outputs=pd.DataFrame(np.column_stack([state==v for v in range(3)]).astype(float))
        outputs["total"]=np.log1p(np.nansum(expenses,axis=1))
        use=draw(local[ANCHOR+PROFILE],cases[ANCHOR+PROFILE],outputs,rng)
        amount=expenses[use].copy()
        # Sample a budget from nearby amounts within the identical multivariate
        # service state, preserving zeroes, missingness, and the drawn composition.
        budget=np.nansum(expenses,axis=1)
        code=state@(3**np.arange(len(BILLS)))
        for pattern in np.unique(code[use]):
            available=np.flatnonzero((code==pattern)&(budget>0))
            if len(available)<2:continue
            candidates=available[np.argsort(budget[available])]
            q=np.flatnonzero(code[use]==pattern)
            pos=np.searchsorted(budget[candidates],budget[use[q]])
            chosen=rng.integers(np.maximum(0,pos-4),np.minimum(len(candidates),pos+5))
            scale=budget[candidates[chosen]]/budget[use[q]]
            amount[q]=np.where(amount[q]>0,np.maximum(1,np.rint(amount[q]*scale[:,None])),amount[q])
            total_transfers+=int((scale!=1).sum())
        np.testing.assert_array_equal(np.isnan(amount),np.isnan(expenses[use]))
        np.testing.assert_array_equal(amount>0,expenses[use]>0)
        cases[BILLS]=amount
        generated.iloc[idx]=cases.to_numpy()
        for c in local:
            shuffled.loc[idx,c]=local[c].to_numpy()[rng.permutation(len(idx))]
    for frame in [generated,shuffled]:
        for c in frame:
            if c not in CATEGORICAL:frame[c]=pd.to_numeric(frame[c])
    return generated,shuffled,total_transfers


def run():
    y,base_hits,challengers=read_panel()
    old=~base_hits.any(0)
    hard=old&~challengers.any(axis=(0,1))
    assert (old.sum(),hard.sum())==(1070,466)
    dev=pd.read_csv(ART/'nov23_global_error_inventory.csv',usecols=['row']).row.to_numpy()
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ['Transported','Name'])
                   for s in ['train','test']],ignore_index=True)
    party=raw.PassengerId.str.split('_').str[0]
    x=compact_features(raw).assign(PartySize=party.map(party.value_counts())).iloc[dev].reset_index(drop=True)
    group=party.iloc[dev].to_numpy()
    out=ART/'nov23_passenger_sampler'
    out.mkdir(exist_ok=True)
    result={}
    for seed in SEEDS:
        with np.load(ART/f'nov23_group_kernel/{seed}.npz') as z:
            fold,base=z['fold'],z['base']
            verify(z,dev,y,fold)
        probs={n:np.full(len(y),np.nan) for n in ['duplicate','permuted','sequential']}
        transferred=[]
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(group[t],group[q]).size
            xt=x.iloc[t].reset_index(drop=True)
            shared,weak,count=generate(xt,y[t],seed+int(k))
            transferred.append(count)
            for name,examples in [('duplicate',xt),('permuted',weak),('sequential',shared)]:
                probs[name][q]=train(xt,y[t].astype(float),examples,y[t].astype(float),x.iloc[q])
        summary={}
        for n,p in probs.items():
            assert np.isfinite(p).all()
            for name,score in [(n,p),(n+'_blend',.75*base+.25*p)]:
                ok=(score>=.5)==y
                wrong=(base>=.5)!=y
                summary[name]=dict(accuracy=float(ok.mean()),fixed=int((wrong&ok).sum()),harmed=int((~wrong&~ok).sum()),
                                   recovered1070=int((ok&old).sum()),recovered466=int((ok&hard).sum()))
        np.savez_compressed(out/f'{seed}.npz',row=dev,y=y,fold=fold,base=base,**probs)
        result[str(seed)]=dict(metrics=summary,changed_budget_per_fold=transferred)
        write(ROOT/'reports/nov23_passenger_sampler/result.json',result)
        print(seed,summary,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):
        run()
