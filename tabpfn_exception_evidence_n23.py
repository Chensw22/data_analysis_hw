"""Test the extra information in changing a frozen classifier's example population."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from archive_error_recovery_n23 import ART, ROOT, SEEDS, panel, predictions
from earth_bill_order_n23 import BILLS
from fresh_family_rescue_n23 import NEW
from relational_error_corrector import write


def run():
    pack={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_earth_tabpfn_context/{seed}.npz") as z:
            pack[seed] = {k:z[k] for k in ("row","y","base","fold","balanced","exceptions")}
    first = pack[SEEDS[0]]
    row,y = first["row"],first["y"]
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId","HomePlanet","CryoSleep",*BILLS]).iloc[row]
    earth=raw.HomePlanet.eq("Earth").to_numpy()
    sleeping=raw.CryoSleep.eq(True).to_numpy()
    spending=raw[BILLS].gt(0).any(axis=1).to_numpy()
    group=raw.PassengerId.str.split("_").str[0].to_numpy()
    repeat=earth&np.logical_and.reduce([(v["base"]>=.5)!=y for v in pack.values()])
    with np.load(ART/"nov23_archive_recovery/masks.npz") as prior:
        np.testing.assert_array_equal(prior["row"],row)
        stubborn=earth&prior["still_unresolved"]
    out=ART/"nov23_tabpfn_exception_evidence"
    out.mkdir(exist_ok=True)
    result,all_hit={},{}
    for seed,v in pack.items():
        np.testing.assert_array_equal(row,v["row"])
        np.testing.assert_array_equal(y,v["y"])
        fold,base=v["fold"],v["base"]
        b,regular,special = [logit(v[k].clip(.001,.999)) for k in ("base","balanced","exceptions")]
        delta=special-regular
        inputs={"control":[b,regular], "with_delta":[b,regular,delta],
                "conditional_delta":[b,regular,delta,abs(delta),b*delta,delta*sleeping,delta*spending]}
        estimates={name:base.copy() for name in inputs}
        for f in np.unique(fold):
            held=np.flatnonzero(earth&(fold==f))
            for a,c in GroupKFold(3,shuffle=True,random_state=620).split(held,groups=group[held]):
                t,q=held[a],held[c]
                assert not np.intersect1d(group[t],group[q]).size
                for name,columns in inputs.items():
                    x=np.column_stack(columns)
                    clf=LogisticRegression(C=.3,max_iter=400).fit(x[t],y[t])
                    estimates[name][q]=clf.predict_proba(x[q])[:,1]
        estimates.update(balanced=v["balanced"],exceptions=v["exceptions"])
        scores={}
        ref=(base>=.5)==y
        for name,p in estimates.items():
            hit=(p>=.5)==y
            fixed=hit&~ref
            harm=~hit&ref
            scores[name]=dict(accuracy=float(hit.mean()),fixed=int(fixed.sum()),harmed=int(harm.sum()),
                              fixed_misses=int((fixed&y).sum()),fixed_false_alarms=int((fixed&~y).sum()),
                              previous804=int((hit&repeat).sum()),previous179=int((hit&stubborn).sum()))
            all_hit.setdefault(name,[]).append(hit)
        result[seed]=scores
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**estimates)
    stable={name:dict(previous804=int((repeat&np.logical_and.reduce(h)).sum()),
                      previous179=int((stubborn&np.logical_and.reduce(h)).sum())) for name,h in all_hit.items()}
    aligned,labels,_,archived,_,_,_=panel()
    np.testing.assert_array_equal(aligned,row)
    np.testing.assert_array_equal(labels,y)
    recent=np.stack([predictions(row,y,seed,NEW)[1] for seed in SEEDS])
    previous=(np.concatenate([archived,recent],axis=2)>=.5)==y[None,:,None]
    rescued=repeat&np.logical_and.reduce(all_hit["exceptions"])
    had_stable=previous.all(0).any(1)
    had_any=previous.any(axis=(0,2))
    overlap=dict(previous_candidates=previous.shape[2],previously_stable=int((rescued&had_stable).sum()),
                 previously_intermittent=int((rescued&~had_stable&had_any).sum()),
                 never_previously_right=int((rescued&~had_any).sum()))
    write(ROOT/"reports/nov23_tabpfn_exception_evidence/result.json",dict(results=result,stable=stable,overlap=overlap))


if __name__=="__main__":
    run()
