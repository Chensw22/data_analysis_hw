"""Compare risk-matched foundation contexts with random, class-count-matched contexts."""
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

import mitra_candidate as mitra
from boarding_sequence_n23 import ROOT,ART,hard_rows
from exaone_validation_data import compact_features
from mitra_crossfit_teachers_n23 import predictions
from relational_error_corrector import write


def run(seed=20261112):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal unavailable")
    torch.set_num_threads(3)
    torch.mps.set_per_process_memory_fraction(.65)
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ["Name","Transported"])
    with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
        row,fold,y,base=[z[k] for k in ("row","fold","y","base")]
    frame=compact_features(raw.iloc[row]).reset_index(drop=True)
    group=raw.PassengerId.iloc[row].str.split("_").str[0].to_numpy()
    hard=hard_rows(row,y)
    out=ART/"nov23_mitra_risk_context"
    out.mkdir(exist_ok=True)
    model,_=mitra.load_initial_model(mitra.CHECKPOINT_DIR,"mps")
    outputs={k:np.full(len(y),np.nan) for k in ["risk_context","matched_random"]}
    seen=np.zeros(len(y),int)
    diagnostics=[]
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.intersect1d(group[t],group[q]).size
        # Support and query scores both come from one teacher trained without the
        # passenger's booking. Neither score is presented as an input to Mitra.
        train,query,_,_=predictions(frame,y,group,t,q,seed+int(k),match_single_teacher=True)
        clusters=KMeans(4,n_init=10,random_state=120).fit(train)
        allocation=clusters.predict(query)
        pre=mitra.MitraPreprocessor().fit(frame.iloc[t],y[t].astype(int),seed=seed+int(k))
        x=pre.transform(frame)
        for j,center in enumerate(clusters.cluster_centers_):
            target=q[allocation==j]
            if not len(target):continue
            distance=((train-center)**2).sum(axis=1)
            support=np.sort(t[np.argsort(distance,kind="stable")[:1536]])
            rng=np.random.default_rng(seed+int(k)*10+j)
            placebo=np.sort(np.concatenate([rng.choice(t[y[t]==c],int((y[support]==c).sum()),replace=False)
                                            for c in [False,True]]))
            assert np.array_equal(np.bincount(y[support].astype(int)),np.bincount(y[placebo].astype(int)))
            for key,ix in [("risk_context",support),("matched_random",placebo)]:
                a=torch.as_tensor(x[ix],device="mps")
                labels=torch.as_tensor(y[ix].astype(int),device="mps")
                b=torch.as_tensor(x[target],device="mps")
                outputs[key][target]=mitra.predict_arrays(model,a,labels,b,chunk=768)[:,1]
                del a,b,labels
                torch.mps.empty_cache()
            seen[target]+=1
            diagnostics.append(dict(fold=int(k),region=j,query=len(target),support=len(support),
                                    train_positives=int(y[support].sum())))
        np.savez_compressed(out/f"fold{k}.npz",row=row[q],fold=fold[q],y=y[q],**{key:p[q] for key,p in outputs.items()})
        print(int(k),{key:float(((p[q]>=.5)==y[q]).mean()) for key,p in outputs.items()},flush=True)
    assert (seen==1).all()
    results={}
    for key,p in outputs.items():
        assert np.isfinite(p).all()
        for name,prob in [(key,p),(key+"_blend",.75*base+.25*p)]:
            ok=(prob>=.5)==y
            bad=(base>=.5)!=y
            results[name]=dict(accuracy=float(ok.mean()),fixed=int((bad&ok).sum()),
                               harmed=int((~bad&~ok).sum()),hard447_correct=int(ok[hard].sum()))
    real=(outputs["risk_context"]>=.5)==y
    control=(outputs["matched_random"]>=.5)==y
    results["intervention_vs_control"]=dict(fixed=int((real&~control).sum()),harmed=int((~real&control).sum()))
    write(ROOT/"reports/nov23_mitra_risk_context/result.json",dict(seed=seed,results=results,contexts=diagnostics))
    np.savez_compressed(out/f"{seed}.npz",row=row,fold=fold,y=y,base=base,**outputs)
    print(results,flush=True)


if __name__=="__main__":
    run()
