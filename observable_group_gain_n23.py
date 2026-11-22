"""Bound and cross-fit expert selection in progressively finer observable passenger strata."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from archive_error_recovery_n23 import ART,ROOT,SEEDS,panel
from population_error_accounting_n23 import BILLS
from relational_error_corrector import write


def choose(train, query, code, hits):
    gain=hits.astype(int)-hits[:,:1].astype(int)
    use=np.zeros(len(query),int)
    for cell in np.unique(code[query]):
        seen=train[code[train]==cell]
        if not len(seen):
            continue
        use[code[query]==cell]=gain[seen].sum(0).argmax()
    return use


def totals(pred, baseline, y, weak, unresolved):
    good=pred==y
    was=baseline==y
    return dict(accuracy=float(good.mean()),fixed=int((good&~was).sum()),harmed=int((~good&was).sum()),
                weak335=int(good[weak].sum()),unresolved252=int(good[unresolved].sum()))


def run():
    rows,y,base,p,_,names,_=panel()
    hit=(p>=.5)==y[None,:,None]
    ref=(base>=.5)==y
    same=(hit.mean(1)>=ref.mean(1)[:,None]-.01).all(0)
    weak=~ref.any(0)&hit.all(0).any(1)&~hit.all(0)[:,same].any(1)
    unresolved=~ref.any(0)&~hit.any(axis=(0,2))
    assert [weak.sum(),unresolved.sum()]==[335,252]
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId","HomePlanet","CryoSleep",*BILLS])
    book=raw.PassengerId.str.split("_").str[0]
    size=book.map(book.value_counts()).to_numpy()[rows]
    group=book.to_numpy()[rows]
    raw=raw.iloc[rows].reset_index(drop=True)
    bill=raw[BILLS]
    frame=pd.DataFrame(dict(origin=raw.HomePlanet.fillna("?"),sleep=raw.CryoSleep.astype("string").fillna("?"),
                 purchases=np.minimum((bill>0).sum(axis=1),2),multiple=size>1,missing=bill.isna().any(axis=1)))
    # Only first-stage experts whose outer validation scores have no fitting on
    # any groups in that same outer fold may enter the cross-fitted selector.
    direct=["base","registration_only","budget_prior","feature_projections","field_factors",
            "bayesian_forest","tensor_train","passenger_relation","manifest_likelihood","boarding_sequence"]
    idx=[["base",*names].index(s) for s in direct]
    result={}
    for n in (2,3,4,5,6):
        summary={}
        for si,seed in enumerate(SEEDS):
            f=frame.copy()
            f["direction"]=base[si]>=.5
            cols=["origin","sleep","purchases","multiple","missing","direction"][:n]
            groups=f.groupby(cols,sort=True).ngroup().to_numpy()
            count=np.bincount(groups)
            preds=np.column_stack([base[si],p[si]])>=.5
            h=preds==y[:,None]
            allrows=np.arange(len(y))
            # Exact in-sample maximum for this policy class: maximize correct
            # count separately in each label-blind cell, including the baseline.
            oracle=choose(allrows,allrows,groups,h)
            by_cell=np.stack([np.bincount(groups,weights=member) for member in h.T])
            assert int(by_cell.max(axis=0).sum()) == int(h[allrows,oracle].sum())
            with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
                folds=z["fold"]
            safe=np.full(len(y),-1,int)
            for fold in np.unique(folds):
                held=np.flatnonzero(folds==fold)
                for a,b in GroupKFold(3,shuffle=True,random_state=209).split(held,groups=group[held]):
                    t,q=held[a],held[b]
                    assert not np.intersect1d(group[t],group[q]).size
                    safe[q]=np.asarray(idx)[choose(t,q,groups,h[:,idx])]
            assert (safe>=0).all()
            oracle_hit=preds[allrows,oracle]
            safe_hit=preds[allrows,safe]
            summary[seed]=dict(cells=len(count),largest=int(count.max()),
                               oracle=totals(oracle_hit,preds[:,0],y,weak,unresolved),
                               held_out=totals(safe_hit,preds[:,0],y,weak,unresolved))
            if n==3:
                by_cell={}
                for key,part in f.groupby(cols,sort=True).indices.items():
                    if len(part)>=100:
                        by_cell["/".join(map(str,key))]=dict(rows=len(part),weak335=int(weak[part].sum()),
                          unresolved252=int(unresolved[part].sum()),net=int((safe_hit[part]==y[part]).sum()-ref[si,part].sum()))
                summary[seed]["cells_with_at_least_100"]=by_cell
        result[n]=summary
        print(n,[(round(m["oracle"]["accuracy"],4),round(m["held_out"]["accuracy"],4)) for m in summary.values()])
    write(ROOT/"reports/nov23_observable_group_gain/result.json",dict(order=["origin","sleep","purchases","multiple","missing","direction"],
          oracle_members=["base",*names],selection_members=direct,results=result))


if __name__=="__main__":
    run()
