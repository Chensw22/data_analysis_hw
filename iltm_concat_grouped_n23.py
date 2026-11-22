"""Evaluate one native CPU iLTM xgbrconcat member on 1,024 unsealed support passengers."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from iltm_resource_probe import COLUMNS, CATEGORICAL, FAIL_CLOSED, WEIGHT, VENDOR, probabilities, sha, write

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"artifacts/nov23_iltm_concat_grouped"
REPORT=ROOT/"reports/nov23_iltm_concat_grouped"


def worker(seed, fold):
    import numpy as np
    import pandas as pd
    import torch
    from threadpoolctl import threadpool_limits
    sys.path.insert(0,str(VENDOR))
    from iltm import iLTMClassifier
    from iltm import inference_interface
    assert Path(inference_interface.__file__).resolve()==VENDOR/"iltm/inference_interface.py"
    torch.set_num_threads(3)
    torch.set_num_interop_threads(2)
    start=time.monotonic()
    with np.load(ROOT/f"artifacts/nov23_group_kernel/{seed}.npz") as z:
        r,y,fold_id,base=(z[k] for k in ("row","y","fold","base"))
    roster=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"}).iloc[r].copy()
    g=roster.pop("PassengerId").str.split("_").str[0].to_numpy()
    train=np.flatnonzero(fold_id!=fold)
    q=np.flatnonzero(fold_id==fold)
    assert not np.intersect1d(g[train],g[q]).size
    support=np.sort(np.random.default_rng(seed+fold).choice(train,1024,replace=False))
    cab=roster.pop("Cabin").str.split("/",expand=True)
    roster["Deck"],roster["CabinNumber"],roster["Side"]=cab[0],pd.to_numeric(cab[1]),cab[2]
    for col in ["CryoSleep","VIP"]:
        roster[col]=roster[col].map({True:"yes",False:"no"})
    roster=roster[COLUMNS]
    with threadpool_limits(3):
        model=iLTMClassifier(device="cpu",n_ensemble=1,seed=7108,cat_features=CATEGORICAL,checkpoint=str(WEIGHT),**FAIL_CLOSED)
        params=model.get_params()
        assert params["tree_model"]=="XGBoost_hist" and params["concat_tree_with_orig_features"] and params["tree_embedding"]
        model.fit(roster.iloc[support],y[support].astype(int),eval_set=None,fit_max_time=None,return_partial_on_timeout=False)
        assert model._fit_succeeded and len(model.predictors_)==1 and not model.predictors_[0]["timed_out"]
        pp=np.concatenate([probabilities(model,roster.iloc[q[i:i+384]])[:,1] for i in range(0,len(q),384)])
    assert os.environ["KMP_DUPLICATE_LIB_OK"]=="FALSE"
    path=OUT/f"{seed}_{fold}.npz"
    np.savez_compressed(path,train=r[support],row=r[q],y=y[q],base=base[q],p=pp)
    rec=dict(rows=len(q),training_rows=len(support),correct=int(((pp>=.5)==y[q]).sum()),rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             seconds=time.monotonic()-start,sha256=sha(path))
    write(REPORT/f"{seed}_fold{fold}.json",rec)
    print(seed,fold,rec,flush=True)


def aggregate(seed):
    import numpy as np
    with np.load(ROOT/f"artifacts/nov23_group_kernel/{seed}.npz") as z:
        r,y,fold,base=(z[k] for k in ("row","y","fold","base"))
    with np.load(ROOT/"artifacts/nov23_archive_recovery/masks.npz") as z:
        np.testing.assert_array_equal(z["row"],r)
        hard=z["still_unresolved"]
    p=np.full(len(r),np.nan)
    for f in np.unique(fold):
        with np.load(OUT/f"{seed}_{f}.npz") as z:
            np.testing.assert_array_equal(z["row"],r[fold==f])
            np.testing.assert_array_equal(z["y"],y[fold==f])
            p[fold==f]=z["p"]
    assert np.isfinite(p).all()
    scores={}
    for name,v in (("direct",p),("quarter",.75*base+.25*p)):
        h=(v>=.5)==y
        b=(base>=.5)==y
        scores[name]=dict(accuracy=float(h.mean()),fixed=int((h&~b).sum()),harmed=int((~h&b).sum()),deep252=int(h[hard].sum()))
    np.savez_compressed(OUT/f"{seed}.npz",row=r,y=y,base=base,fold=fold,p=p)
    write(REPORT/f"{seed}.json",scores)
    print(seed,scores,flush=True)


def overlap():
    import numpy as np
    from archive_error_recovery_n23 import panel
    r,y,b,bank,hard,_,_=panel()
    base=(b>=.5)==y
    h=(bank>=.5)==y[None,:,None]
    comparable=(h.mean(1)>=base.mean(1)[:,None]-.01).all(0)
    weak=~base.any(0)&h.all(0).any(1)&~h.all(0)[:,comparable].any(1)
    deep=~h.any(axis=(0,2))&~base.any(0)
    assert weak.sum()==335 and deep.sum()==252
    preds=[]
    for seed in [20261112,20261115,20261116]:
        with np.load(OUT/f"{seed}.npz") as z:
            np.testing.assert_array_equal(r,z["row"])
            np.testing.assert_array_equal(y,z["y"])
            preds.append(z["p"])
    p=np.array(preds)
    out={}
    for name,pp in [("direct",p),("quarter",.75*b+.25*p)]:
        hit=(pp>=.5)==y
        out[name]={}
        for title,subset in [("common252",deep),("weak335",weak),("all1070",~base.any(0))]:
            a=hit[:,subset]
            out[name][title]=dict(hits=a.sum(1).tolist(),stable=int(a.all(0).sum()),ever=int(a.any(0).sum()))
    write(REPORT/"overlap.json",out)
    matched_controls(r,y,deep,weak)
    calibration(r,y,b,p,deep,weak)


def matched_controls(rows, y, deep, weak):
    """Attribute support-size loss with an ordinary tree fitted to identical rows."""
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier
    from exaone_validation_data import CATEGORICAL as cats,compact_features
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"})
    x=compact_features(raw.iloc[rows]).reset_index(drop=True)
    lookup={j:i for i,j in enumerate(rows)}
    report={}
    for seed in [20261112,20261115,20261116]:
        pred=np.full(len(y),np.nan)
        for f in range(3):
            with np.load(OUT/f"{seed}_{f}.npz") as z:
                t=np.array([lookup[j] for j in z["train"]])
                q=np.array([lookup[j] for j in z["row"]])
            clf=CatBoostClassifier(iterations=350,depth=5,learning_rate=.05,thread_count=2,
                                   random_seed=7108,cat_features=cats,verbose=False,allow_writing_files=False)
            clf.fit(x.iloc[t],y[t])
            pred[q]=clf.predict_proba(x.iloc[q])[:,1]
        assert np.isfinite(pred).all()
        with np.load(OUT/f"{seed}.npz") as z:
            candidate=(z["p"]>=.5)==y
        control=(pred>=.5)==y
        report[seed]=dict(control_accuracy=float(control.mean()),candidate_over_control_fixes=int((candidate&~control).sum()),
                          candidate_over_control_harms=int((~candidate&control).sum()),
                          control_weak335=int(control[weak].sum()),control_deep252=int(control[deep].sum()))
    write(REPORT/"matched_control.json",report)


def calibration(rows,y,base,scores,deep,weak):
    """Test whether different score ranges across support draws explain misses."""
    import numpy as np
    import pandas as pd
    from scipy.special import logit
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    groups=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.str.split("_").str[0].to_numpy()[rows]
    report={}
    for i,seed in enumerate([20261112,20261115,20261116]):
        p=scores[i]
        z=logit(p.clip(.001,.999)).reshape(-1,1)
        calibrated=np.full(len(y),np.nan)
        stats=[]
        with np.load(OUT/f"{seed}.npz") as cache:
            fold=cache["fold"]
        for f in np.unique(fold):
            section=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3).split(section,groups=groups[section]):
                t,q=section[a],section[b]
                assert not np.intersect1d(groups[t],groups[q]).size
                fit=LogisticRegression(C=1.).fit(z[t],y[t])
                calibrated[q]=fit.predict_proba(z[q])[:,1]
            stats.append(dict(mean_probability=float(p[section].mean()),positive_fraction=float(y[section].mean()),
                              predicted_positive_fraction=float((p[section]>=.5).mean()),
                              auc=float(roc_auc_score(y[section],p[section]))))
        assert np.isfinite(calibrated).all()
        also={}
        for tag,v in [("calibrated",calibrated),("calibrated_quarter",.75*base[i]+.25*calibrated)]:
            hit=(v>=.5)==y
            ref=(base[i]>=.5)==y
            also[tag]=dict(accuracy=float(hit.mean()),fixed=int((hit&~ref).sum()),harmed=int((~hit&ref).sum()),
                           deep252=int(hit[deep].sum()),weak335=int(hit[weak].sum()))
        report[seed]=dict(folds=stats,metrics=also)
    write(REPORT/"calibration.json",report)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--worker",nargs=2,type=int)
    parser.add_argument("--overlap",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    if args.overlap:
        overlap()
        return
    if args.worker:
        worker(*args.worker)
        return
    setup=json.loads((ROOT/"reports/iltm_setup.json").read_text())
    assert sha(WEIGHT)==setup["weights"]["sha256"]
    assert sha(VENDOR/"LICENSE")==setup["source"]["license_sha256"]
    assert sha(WEIGHT.parent/"MODEL_CARD.md")==setup["weights"]["model_card_sha256"]
    assert sha(VENDOR/"SOURCE.json")==setup["source"]["source_manifest_sha256"]
    manifest=json.loads((VENDOR/"SOURCE.json").read_text())
    for path,digest in manifest["source_files_sha256"].items():
        assert sha(VENDOR/path)==digest
    env=dict(os.environ,KMP_DUPLICATE_LIB_OK="FALSE",DYLD_FALLBACK_LIBRARY_PATH=str(Path(importlib.util.find_spec("torch").origin).parent/"lib"))
    for key in ["OMP_NUM_THREADS","OMP_THREAD_LIMIT","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
        env[key]="3"
    for seed in [20261112,20261115,20261116]:
        for f in range(3):
            rec=REPORT/f"{seed}_fold{f}.json"
            if rec.exists() and (OUT/f"{seed}_{f}.npz").exists() and sha(OUT/f"{seed}_{f}.npz")==json.loads(rec.read_text())["sha256"]:
                continue
            subprocess.run([sys.executable,__file__,"--worker",str(seed),str(f)],check=True,env=env)
        aggregate(seed)


if __name__=="__main__":
    main()
