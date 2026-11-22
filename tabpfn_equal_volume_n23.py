"""At equal switch volume, compare exception context to shifts and random ties."""
import numpy as np
import pandas as pd

from archive_error_recovery_n23 import ART, ROOT, SEEDS
from earth_bill_order_n23 import BILLS
from relational_error_corrector import write


def run():
    stacks={key:[] for key in ("base","exceptions","balanced","base_rate","balanced_rate","base_bill_rate","net_prior","constant_negative","constant_positive")}
    results={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_earth_tabpfn_context/{seed}.npz") as z:
            base,exc,ordinary,fold,row,y=[z[k] for k in ("base","exceptions","balanced","fold","row","y")]
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            for key,values in [("base",base),("fold",fold),("row",row),("y",y)]:
                np.testing.assert_array_equal(z[key],values)
        raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["HomePlanet","PassengerId",*BILLS]).iloc[row]
        earth=raw.HomePlanet.eq("Earth").to_numpy()
        booking=raw.PassengerId.str.split("_").str[0].to_numpy()
        paid=raw[BILLS].sum(axis=1).to_numpy()
        missing=raw[BILLS].isna().any(axis=1).to_numpy()
        baseline=base>=.5
        exceptions=exc>=.5
        errors=baseline!=y
        decisions={k:baseline.copy() for k in ("base_rate","balanced_rate","base_bill_rate")}
        net_prior=baseline.copy()
        random_expected=np.zeros(len(y))
        bill_band=np.full(len(y),"none",dtype="U12")
        for f in np.unique(fold):
            train=earth&(fold!=f)
            query=earth&(fold==f)
            assert not np.intersect1d(booking[train],booking[query]).size
            cut=np.median(paid[train & ~missing & (paid>0)])
            bill_band[query]=np.select([missing[query],paid[query]==0,paid[query]<=cut],
                                       ["missing","zero","low"],default="high")
            ids=np.flatnonzero(query)
            net_prior[ids]=False
            chosen=ids[np.lexsort((row[ids],-base[ids]))[:int(exceptions[ids].sum())]]
            net_prior[chosen]=True
            for direction in (False,True):
                ids=np.flatnonzero(query&(baseline==direction))
                flip=exceptions!=baseline
                k=int(flip[ids].sum())
                random_expected[ids]=k/len(ids)
                for key,score in (("base_rate",base),("balanced_rate",ordinary),("base_bill_rate",base)):
                    for part in ([ids] if key!="base_bill_rate" else
                                 [ids[bill_band[ids]==val] for val in np.unique(bill_band[ids])]):
                        kk=int(flip[part].sum())
                        risk=1-score[part] if direction else score[part]
                        take=part[np.lexsort((row[part],-risk))[:kk]]
                        decisions[key][take]=~baseline[take]
                assert all(np.sum(v[ids]!=baseline[ids])==k for v in decisions.values())
        pred=dict(base=baseline,exceptions=exceptions,balanced=ordinary>=.5,net_prior=net_prior,constant_negative=baseline & ~earth,
                  constant_positive=baseline | earth,**decisions)
        for key,values in pred.items():
            stacks[key].append(values)
        out={}
        for name,part in [("all",earth),("baseline_positive",earth&baseline),("baseline_negative",earth&~baseline),
                          *[(band,earth&(bill_band==band)) for band in ("missing","zero","low","high")]]:
            t={}
            for key,values in pred.items():
                flip=(values!=baseline)&part
                t[key]=dict(switched=int(flip.sum()),fixed=int((flip&errors).sum()),
                            harmed=int((flip&~errors).sum()),overall_accuracy=float((values==y).mean()))
            t["constant_score_random_ties"]=dict(expected_fixed=float(random_expected[part&errors].sum()),
                expected_harmed=float(random_expected[part&~errors].sum()))
            out[name]=t
        results[seed]=out
    persist=earth&np.all(np.array(stacks["base"])!=y,axis=0)
    stable={k:persist&np.all(np.array(stack)==y,axis=0) for k,stack in stacks.items()}
    original=stable["exceptions"]
    assert persist.sum()==804 and original.sum()==115
    retained={k:dict(stable=int(mask.sum()),same_as_115=int((mask&original).sum())) for k,mask in stable.items()}
    checks={}
    for seed in SEEDS:
        v=results[seed]
        checks[seed]=[(k,v[k]["exceptions"]["fixed"],v[k]["base_bill_rate"]["fixed"]) for k in ("zero","low","high")]
    print(retained,checks)
    write(ROOT/"reports/nov23_tabpfn_equal_volume/result.json",dict(retention=retained,results=results))


if __name__=="__main__":
    run()
