"""Replay the two training-error corrections on the original v131 score and booking partitions."""
import numpy as np
import pandas as pd
from scipy.special import logit,expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_train_error_margins_n23 import ART,ROOT,REPORT,features,teacher_misses,optimize
from broad_dispute_leaf_evidence_n23 import cohorts
from exaone_validation_data import compact_features
from relational_error_corrector import write


def run():
    with np.load(ART/"nov23_incumbent_budget_transfer/oof.npz") as z:
        row,y,fold,inc,base,unseen=[z[n] for n in ("row","y","fold","v131","v108","unseen")]
    audit=pd.read_csv(ART/"nov21_case_checks/v136_final_oof.csv")
    for name,expected in (("row",row),("true",y),("fold",fold),("v131",inc)):
        np.testing.assert_array_equal(audit[name],expected)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ("Name","Transported"))
                   for s in ("train","test")],ignore_index=True)
    np.testing.assert_array_equal(raw.PassengerId.iloc[row],audit.PassengerId)
    group=raw.PassengerId.str.split("_").str[0]
    size=group.map(group.value_counts()).to_numpy()[row]
    gs=group.to_numpy()[row]
    frame=compact_features(raw.iloc[row])
    outcomes={key:np.full(len(y),np.nan) for key in ("ordinary","real")}
    for k in np.unique(fold):
        q=np.flatnonzero(fold==k)
        t=np.flatnonzero((fold!=k)&~np.isin(gs,gs[q]))
        assert not np.intersect1d(gs[t],gs[q]).size
        np.testing.assert_array_equal(unseen[q],~np.isin(gs[q],gs[fold!=k]))
        teacher_x,x,z=features(frame,size,t,q)
        errs=teacher_misses(frame,teacher_x,y,gs,t)
        def scores(key):
            beta,_=optimize(x,y[t],errs,key)
            return logit(expit(z@beta).clip(.002,.998))
        ordinary=scores("ordinary")
        delta=np.column_stack([scores("robust")-ordinary,scores("margin")-scores("constant")])
        control=np.column_stack([logit(base[q].clip(.002,.998)),ordinary])
        designs={"ordinary":control,"real":np.column_stack([control,delta,delta*np.where(base[q]>=.5,1,-1)[:,None]])}
        for a,b in GroupKFold(3,shuffle=True,random_state=421).split(q,groups=gs[q]):
            assert not np.intersect1d(gs[q[a]],gs[q[b]]).size
            for key,v in designs.items():
                model=LogisticRegression(C=.3,max_iter=250).fit(v[a],y[q[a]])
                outcomes[key][q[b]]=model.predict_proba(v[b])[:,1]
        print("v131 fold",int(k),"complete",flush=True)
    before=inc==y
    override=inc!=(base>=.5)
    masks=cohorts(row,y)
    summary={}
    for key,p in outcomes.items():
        assert np.isfinite(p).all()
        hit=np.where(override,inc,p>=.5)==y
        summary[key]={}
        for name,m in {"all":np.ones(len(y),bool),"unseen_booking":unseen,**masks}.items():
            fixed=hit&~before&m
            harmed=~hit&before&m
            summary[key][name]=dict(rows=int(m.sum()),accuracy=float(hit[m].mean()),fixed=int(fixed.sum()),harmed=int(harmed.sum()),
                     net_by_fold=[int(fixed[fold==k].sum()-harmed[fold==k].sum()) for k in np.unique(fold)])
    write(REPORT/"incumbent.json",dict(baseline=float(before.mean()),overrides=int(override.sum()),results=summary))
    print(summary["real"]["all"])


if __name__=="__main__":
    with threadpool_limits(2):
        run()
