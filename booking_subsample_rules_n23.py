"""Use agreement across independently trained booking subsets as observable correction evidence."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ROOT,ART,SEEDS
from exaone_validation_data import compact_features,CATEGORICAL
from grouped_five_way_coverage_n23 import verify
from grouped_sparse_rules_n23 import remaining,rule_basis
from relational_error_corrector import write


def run():
    records=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    row,y=records.row.to_numpy(),records.target.to_numpy(bool)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ["Name","Transported"])
                   for s in ["train","test"]],ignore_index=True)
    party=raw.PassengerId.str.split("_").str[0]
    groups=party.iloc[row].to_numpy()
    frame=compact_features(raw).iloc[row].copy()
    frame["BookingSize"]=party.map(party.value_counts()).iloc[row].to_numpy()
    x=pd.get_dummies(frame,columns=CATEGORICAL).fillna(-1).to_numpy(np.float32)
    hard=remaining(row,y)
    out=ART/"nov23_rule_stability"
    out.mkdir(exist_ok=True)
    summary={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_sparse_rules/{seed}.npz") as z:
            base,fold,original=z["base"],z["fold"],z["rules"]
            verify(z,row,y,fold)
        members=np.full((len(y),5),np.nan)
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            parties=np.unique(groups[t])
            assert not set(groups[t])&set(groups[q])
            for i in range(5):
                rng=np.random.default_rng(seed+100*int(k)+i)
                subset=rng.choice(parties,int(.8*len(parties)),replace=False)
                s=t[np.isin(groups[t],subset)]
                tree=GradientBoostingClassifier(n_estimators=120,learning_rate=.045,max_depth=3,min_samples_leaf=20,
                                                max_features=.8,subsample=.78,random_state=312+i)
                tree.fit(x[s],y[s])
                _,features=rule_basis(tree,x)
                fit=LogisticRegression(solver="liblinear",l1_ratio=1,C=.25,max_iter=2500,random_state=312+i)
                fit.fit(features[s],y[s])
                if fit.n_iter_.max()>=fit.max_iter:
                    raise RuntimeError("Rule vote coefficients did not converge")
                members[q,i]=fit.predict_proba(features[q])[:,1]
        assert np.isfinite(members).all()
        avg=members.mean(1)
        agreement=((members>=.5)!=(base>=.5)[:,None]).sum(1)
        combined=.75*base+.25*avg
        variants={"all_rules":avg,"one_rule_blend":.75*base+.25*original,
                  "subsample_blend":combined,"consistent_blend":np.where(agreement>=4,combined,base)}
        metrics={}
        for name,p in variants.items():
            ok=(p>=.5)==y
            bad=(base>=.5)!=y
            metrics[name]=dict(accuracy=float(ok.mean()),fixed=int((bad&ok).sum()),harmed=int((~bad&~ok).sum()),
                               hard356_correct=int(ok[hard].sum()))
        by_votes={}
        for i in range(6):
            sel=agreement==i
            by_votes[str(i)]=dict(passengers=int(sel.sum()),baseline_errors=int(((base>=.5)[sel]!=y[sel]).sum()),
                                  hard356=int((sel&hard).sum()))
        summary[str(seed)]=dict(metrics=metrics,opposing_models=by_votes)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,members=members,**variants)
        print(seed,metrics,flush=True)
    write(ROOT/"reports/nov23_rule_stability/result.json",summary)


if __name__=="__main__":
    with threadpool_limits(1):
        run()
