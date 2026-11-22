"""Replay a new complementary rotation forest with the deployed v131 decision policy."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_dispute_leaf_evidence_n23 import ART,ROOT,SEEDS,REPORT,cohorts
from broad_rotation_forest_n23 import rotate_and_fit
from broad_shared_passenger_modes_n23 import axes,expanded_design,peer_pairs
from broad_matched_companions_n23 import fit
from exaone_validation_data import compact_features
from class_analogue_experts_n23 import design
from relational_error_corrector import write


def run():
    with np.load(ART/"nov23_incumbent_budget_transfer/oof.npz") as z:
        row,y,fold,inc,base,unseen=[z[n] for n in ("row","y","fold","v131","v108","unseen")]
    history=pd.read_csv(ART/"nov21_case_checks/v136_final_oof.csv")
    for n,v in [("row",row),("true",y),("fold",fold),("v131",inc)]:
        np.testing.assert_array_equal(history[n],v)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ("Name","Transported"))
                   for s in ("train","test")],ignore_index=True)
    np.testing.assert_array_equal(raw.PassengerId.iloc[row],history.PassengerId)
    groups=raw.PassengerId.str.split("_").str[0].to_numpy()
    gs=groups[row]
    df=compact_features(raw)
    original=df.iloc[row]
    logits={}
    for k in np.unique(fold):
        q=np.flatnonzero(fold==k)
        held=np.isin(groups,gs[q])
        t=np.flatnonzero((fold!=k)&~held[row])
        assert not np.intersect1d(gs[t],gs[q]).size
        np.testing.assert_array_equal(unseen[q],~np.isin(gs[q],gs[fold!=k]))
        x,_=design(original,t)
        x=StandardScaler().fit(x[t]).transform(x)
        rotation=rotate_and_fit(x,y,t,q,1821+int(k))
        unsup=np.flatnonzero(~held)
        _,z=expanded_design(df,row[t],unsup)
        a,b,_,w=peer_pairs(groups,unsup)
        social=z[row]@axes(z,a,b,w)
        shared=fit(social[t],y[t]).predict_proba(social[q])[:,1]
        p=np.column_stack([base[q],rotation,shared])
        x=logit(p.clip(.002,.998))
        logits.setdefault("quarter",np.full(len(y),np.nan))[q]=.75*base[q]+.25*rotation
        for a,b in GroupKFold(3,shuffle=True,random_state=513).split(q,groups=gs[q]):
            assert not np.intersect1d(gs[q[a]],gs[q[b]]).size
            for name,n in (("score_only",1),("rotation",2),("both",3)):
                model=LogisticRegression(C=.1,max_iter=300).fit(x[a,:n],y[q[a]])
                logits.setdefault(name,np.full(len(y),np.nan))[q[b]]=model.predict_proba(x[b,:n])[:,1]
    masks=cohorts(row,y)
    stable_rotation=np.ones(len(y),bool)
    for seed in SEEDS:
        with np.load(ART/f"nov23_broad_rotation_forest/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            stable_rotation &= (z["rotation"]>=.5)==y
    override=inc!=(base>=.5)
    result={}
    for name,p in logits.items():
        assert np.isfinite(p).all()
        hit=np.where(override,inc,p>=.5)==y
        previous=inc==y
        result[name]={}
        for title,m in {"all":np.ones(len(y),bool),"unseen":unseen,**masks}.items():
            fixed=hit&~previous&m
            harmed=~hit&previous&m
            result[name][title]=dict(rows=int(m.sum()),accuracy=float(hit[m].mean()),fixed=int(fixed.sum()),
                        harmed=int(harmed.sum()),net_per_fold=[int(fixed[fold==k].sum()-harmed[fold==k].sum()) for k in np.unique(fold)])
    overlap=int((stable_rotation&masks["new50"]).sum())
    write(REPORT/"rotation_incumbent.json",dict(reference=float((inc==y).mean()),overrides=int(override.sum()),
             rotation_stable_in_previous_new50=overlap,policies=result))
    print({n:v["all"] for n,v in result.items()},overlap,flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
