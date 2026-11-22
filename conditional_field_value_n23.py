"""Learn the value of omitting each block of fields using other training bookings."""
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS, panel
from conditional_view_density_n23 import fit_score
from exaone_validation_data import compact_features
from grouped_five_way_coverage_n23 import verify
from relational_error_corrector import write
from spend_simplex_likelihood_n23 import BILLS


def field_blocks(raw, rows):
    x = compact_features(raw).iloc[rows].reset_index(drop=True)
    g = raw.PassengerId.str.split("_").str[0]
    bills = x[BILLS].to_numpy()
    blocks = {
        "bill_amounts": np.log1p(bills),
        "bill_states": np.column_stack((bills>0, np.isnan(bills))),
        "origin_destination": pd.get_dummies(x[["HomePlanet","Destination"]],dtype=float).to_numpy(),
        "age_sleep_vip": np.column_stack((x.Age, pd.get_dummies(x[["CryoSleep","VIP"]],dtype=float))),
        "cabin_booking": np.column_stack((x.CabinNumber,pd.get_dummies(x[["Deck","Side"]],dtype=float),
                                         g.map(g.value_counts()).to_numpy()[rows])),
    }
    return blocks, g.to_numpy()[rows]


def independent_views(x, indices, labels, t, q):
    # Removing all financial information is distinct from hiding precise amounts
    # while retaining the observed zero/positive/missing states.
    omissions = [(), ("bill_amounts",), ("bill_amounts","bill_states"),
                 ("origin_destination",), ("age_sleep_vip",), ("cabin_booking",)]
    return np.column_stack([expit(fit_score(x[t][:,keep],labels[t],x[q][:,keep],None))
                            for keep in [~np.isin(indices,block) for block in omissions]])


def fit_policies(x, train_scores, labels, t, shuffled=False):
    models=[]
    full=train_scores[:,0]
    for j in range(1,train_scores.shape[1]):
        alt=train_scores[:,j]
        observed=(labels[t]-full)**2-(labels[t]-alt)**2
        if shuffled:
            rng=np.random.default_rng(247+j)
            bins=np.minimum((full*10).astype(int),9)
            for v in np.unique(bins):
                i=np.flatnonzero(bins==v)
                observed[i]=rng.permutation(observed[i])
        model=HistGradientBoostingRegressor(max_iter=70, max_leaf_nodes=7,
                min_samples_leaf=75,l2_regularization=8,learning_rate=.06,
                early_stopping=False,random_state=41)
        model.fit(np.column_stack((x[t],full,alt,alt-full)),observed)
        models.append(model)
    return models


def predict_gains(models, x, scores, q):
    full=scores[:,0]
    gains=[np.zeros(len(q))]
    for model,alt in zip(models,scores[:,1:].T):
        gains.append(model.predict(np.column_stack((x[q],full,alt,alt-full))))
    return np.column_stack(gains)


def decisions(scores, gains):
    # Full-field prediction has zero estimated improvement by definition.
    pick=np.argmax(gains,axis=1)
    return scores[np.arange(len(pick)),pick],pick


def fold_predictions(x,block_fields,y,group,t,q,seed):
    counts=np.zeros(6,int)
    inner=np.full((len(t),6),np.nan)
    independent=[]
    for a,b in GroupKFold(3,shuffle=True,random_state=seed).split(t,groups=group[t]):
        assert not np.intersect1d(group[t[a]],group[t[b]]).size
        view=independent_views(x,block_fields,y,t[a],np.r_[t[b],q])
        inner[b]=view[:len(b)]
        independent.append(view[len(b):])
    assert np.isfinite(inner).all()
    fixed=np.argmin(((inner-y[t,None])**2).mean(0))
    modes={k:[] for k in ("full","fixed_omit","adaptive","shuffled")}
    policies=fit_policies(x,inner,y,t)
    placebo=fit_policies(x,inner,y,t,shuffled=True)
    for query in independent:
        good=predict_gains(policies,x,query,q)
        null=predict_gains(placebo,x,query,q)
        estimate,pick=decisions(query,good)
        counts+=np.bincount(pick,minlength=query.shape[1])
        modes["full"].append(query[:,0])
        modes["fixed_omit"].append(query[:,fixed])
        modes["adaptive"].append(estimate)
        modes["shuffled"].append(decisions(query,null)[0])
    return ({key:np.mean(vals,axis=0) for key,vals in modes.items()}, np.mean(independent,axis=0), counts)


def run():
    rows,y,bref,bank,hard,names,_=panel()
    before=(bref>=.5)==y
    hit=(bank>=.5)==y[None,:,None]
    qualified=(hit.mean(1)>=before.mean(1)[:,None]-.01).all(0)
    weak=~before.any(0)&hit.all(0).any(1)&~hit.all(0)[:,qualified].any(1)
    deep=~hit.any(axis=(0,2)) & ~before.any(0)
    assert weak.sum()==335 and deep.sum()==252
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in {"Name","Transported"})
                   for s in ("train","test")],ignore_index=True)
    blocks,group=field_blocks(raw,rows)
    x=np.column_stack(list(blocks.values()))
    block_fields=np.concatenate([np.repeat(k,v.shape[1]) for k,v in blocks.items()])
    results={}
    out=ART/"nov23_conditional_field_value"
    out.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,rows,y,fold)
        scores={k:np.full(len(y),np.nan) for k in ("full","fixed_omit","adaptive","shuffled")}
        counts=np.zeros(6,int)
        cache_views=np.empty((len(y),6))
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(group[t],group[q]).size
            modes,cache_views[q],choices=fold_predictions(x,block_fields,y,group,t,q,seed+int(f))
            counts+=choices
            for mode,values in modes.items():
                scores[mode][q]=values
            print(seed,"fold",f,flush=True)
        summary={}
        for mode,p in scores.items():
            assert np.isfinite(p).all()
            for name,s in ((mode,p),(mode+"_quarter",.75*base+.25*p)):
                correct=(s>=.5)==y
                ref=(base>=.5)==y
                summary[name]=dict(accuracy=float(correct.mean()),fixed=int((correct&~ref).sum()),
                                   harmed=int((~correct&ref).sum()), weak335=int(correct[weak].sum()),
                                   hard252=int(correct[deep].sum()), hard365=int(correct[hard].sum()))
        zhits=(cache_views>=.5)==y[:,None]
        oracle=dict(weak335_any=int(zhits[weak].any(axis=1).sum()),hard252_any=int(zhits[deep].any(axis=1).sum()),
                    all_any=int(zhits.any(axis=1).sum()))
        results[seed]=dict(metrics=summary,expected_gain_choices=counts.tolist(),retrospective=oracle)
        np.savez_compressed(out/f"{seed}.npz",row=rows,y=y,fold=fold,base=base,views=cache_views,**scores)
        write(ROOT/"reports/nov23_conditional_field_value/result.json",dict(results=results))
        print(seed,summary["adaptive"],summary["adaptive_quarter"],oracle,flush=True)


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
