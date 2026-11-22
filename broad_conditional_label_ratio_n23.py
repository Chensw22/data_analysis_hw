"""Learn the odds of real labels versus labels sampled from an out-of-booking prior."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART,ROOT,SEEDS
from broad_dispute_leaf_evidence_n23 import cohorts
from exaone_validation_data import compact_features,CATEGORICAL
from group_crossfit_residual_discovery_n23 import teachers
from relational_error_corrector import write

OUT=ART/"nov23_broad_label_ratio"
REPORT=ROOT/"reports/nov23_broad_conditional_raw_signal"


def ratio(frame,y,t,q,prior):
    # For each hypothetical class the real distribution puts mass 1 on a
    # matching observed label; the negative distribution contributes the
    # teacher probability to every passenger. Integrate over synthetic labels.
    evidence=[]
    for cls in (0,1):
        truth=(y[t]==cls).astype(float)
        fake=prior if cls else 1-prior
        mass=truth+fake
        clf=CatBoostClassifier(iterations=270,depth=5,learning_rate=.065,l2_leaf_reg=6,loss_function="CrossEntropy",
                               verbose=False,random_seed=402,thread_count=2,allow_writing_files=False)
        clf.fit(frame.iloc[t],truth/mass,sample_weight=mass,cat_features=CATEGORICAL)
        evidence.append(clf.predict(frame.iloc[q],prediction_type="RawFormulaVal"))
    return evidence[1]-evidence[0]


def fold_scores(frame,y,groups,t,q,base,seed):
    assert not np.intersect1d(groups[t],groups[q]).size
    p,qt=teachers(frame,y,groups,t,q,seed)
    p=p.clip(.002,.998)
    qt=qt.clip(.002,.998)
    perm=np.random.default_rng(seed).permutation(p)
    empirical=float(y[t].mean())
    real=ratio(frame,y,t,q,p)
    no_link=ratio(frame,y,t,q,perm)
    direct=ratio(frame,y,t,q,np.full(len(t),empirical))
    return dict(teacher=qt.mean(1),conditional_native=expit(logit(qt)+real[:,None]).mean(1),
                transfer=expit(logit(base[q].clip(.002,.998))+real),
                shuffled_transfer=expit(logit(base[q].clip(.002,.998))+no_link),
                unconditional=expit(logit(empirical)+direct),logratio=real)


def run():
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Transported","Name"))
    OUT.mkdir(exist_ok=True)
    refs=[]
    hits={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            row,y,fold,base=[z[n] for n in ("row","y","fold","base")]
        g=raw.PassengerId.str.split("_").str[0]
        groups=g.to_numpy()[row]
        frame=compact_features(raw.iloc[row])
        frame["BookingSize"]=g.map(g.value_counts()).to_numpy()[row]
        bank={}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            vals=fold_scores(frame,y,groups,t,q,base,seed+int(f))
            for name,v in vals.items():
                bank.setdefault(name,np.full(len(y),np.nan))[q]=v
        for name,p in bank.items():
            assert np.isfinite(p).all()
            if name!="logratio":
                hits.setdefault(name,[]).append((p>=.5)==y)
        refs.append((base>=.5)==y)
        np.savez_compressed(OUT/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**bank)
        print(seed,{k:round(float(((v>=.5)==y).mean()),4) for k,v in bank.items() if k!="logratio"},flush=True)
    ref=np.stack(refs)
    masks=cohorts(row,y)
    result={}
    for name,v in hits.items():
        ok=np.stack(v)
        result[name]=dict(accuracy=ok.mean(1).tolist(),fixed=(ok&~ref).sum(1).tolist(),
                          harmed=(~ok&ref).sum(1).tolist(),stable={k:int((m&ok.all(0)).sum()) for k,m in masks.items()})
    write(REPORT/"conditional_label_ratio.json",result)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
