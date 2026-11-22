"""Test the extra shift from real hard-group training against an equal-cost placebo."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from broad_train_error_margins_n23 import ART,ROOT,SEEDS,REPORT
from broad_dispute_leaf_evidence_n23 import cohorts
from relational_error_corrector import write


def run():
    people=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.str.split("_").str[0].to_numpy()
    hits={}
    refs=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_broad_train_error_margins/{seed}.npz") as z:
            row,y,fold=[z[n] for n in ("row","y","fold")]
            def score(key):
                return logit(z[key].clip(.002,.998))
            base,ordinary=score("base"),score("ordinary")
            real=np.column_stack([score("robust")-ordinary,score("margin")-score("constant")])
            random=np.column_stack([score("random_robust")-ordinary,score("random_margin")-score("constant")])
        gs=people[row]
        refs.append((base>=0)==y)
        # The changes can rectify a passenger differently depending on the
        # direction of the original prediction. This is available at deployment.
        sign=np.where(base>=0,1,-1)
        control=np.column_stack([base,ordinary])
        designs={"ordinary":control,"real":np.column_stack([control,real,real*sign[:,None]]),
                 "random":np.column_stack([control,random,random*sign[:,None]])}
        out={n:np.full(len(y),np.nan) for n in designs}
        for k in np.unique(fold):
            q=np.flatnonzero(fold==k)
            for a,b in GroupKFold(3,shuffle=True,random_state=421).split(q,groups=gs[q]):
                t,c=q[a],q[b]
                assert not np.intersect1d(gs[t],gs[c]).size
                for name,x in designs.items():
                    model=LogisticRegression(C=.3,max_iter=250).fit(x[t],y[t])
                    out[name][c]=model.predict_proba(x[c])[:,1]
        for n,p in out.items():
            assert np.isfinite(p).all()
            hits.setdefault(n,[]).append((p>=.5)==y)
    before=np.stack(refs)
    masks=cohorts(row,y)
    result={}
    for n,v in hits.items():
        good=np.stack(v)
        result[n]=dict(accuracy=good.mean(1).tolist(),fixed=(good&~before).sum(1).tolist(),
                       harmed=(~good&before).sum(1).tolist(),stable={k:int((good.all(0)&m).sum()) for k,m in masks.items()})
    write(REPORT/"readout.json",result)
    print(result["real"])


if __name__=="__main__":
    run()
