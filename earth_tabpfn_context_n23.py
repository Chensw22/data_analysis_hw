"""Built with PriorLabs-TabPFN. Compare uniform and exception-enriched Earth contexts."""
import hashlib
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TABPFN_NO_BROWSER"] = "1"
os.environ["TABPFN_MODEL_VERSION"] = "v2"
os.environ["TABPFN_MODEL_CACHE_SIZE"] = "0"

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from threadpoolctl import threadpool_limits

from earth_order_exceptions_n23 import matches
from earth_bill_order_n23 import BILLS
from exaone_validation_data import compact_features, CATEGORICAL
from archive_error_recovery_n23 import ART,ROOT,SEEDS
from relational_error_corrector import write


def balanced(pool, y, n, rng):
    return np.concatenate([rng.choice(pool[y[pool]==label],n,replace=False) for label in (False,True)])


def run():
    weight = ART/"pretrained/tabpfn-v2-classifier-v2_default.ckpt"
    assert hashlib.sha256(weight.read_bytes()).hexdigest()=="cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"
    with np.load(ART/f"nov23_group_kernel/{SEEDS[0]}.npz") as z:
        row,y = z["row"],z["y"]
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"}).iloc[row].reset_index(drop=True)
    frame=compact_features(raw)
    group=raw.PassengerId.str.split("_").str[0].to_numpy()
    earth=raw.HomePlanet.eq("Earth").to_numpy()
    bills=raw[BILLS].fillna(0).to_numpy()*[1,1,-1,-1,-1]
    mask=raw[BILLS].isna().to_numpy() @ (2**np.arange(5))
    out=ART/"nov23_earth_tabpfn_context"
    out.mkdir(exist_ok=True)
    data={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            data[seed]=z["base"],z["fold"]
    repeat=earth&np.logical_and.reduce([(b>=.5)!=y for b,_ in data.values()])
    results={}
    for seed,(base,fold) in data.items():
        predictions={key:base.copy() for key in ("balanced","exceptions")}
        for f in np.unique(fold):
            t=np.flatnonzero(earth&(fold!=f))
            q=np.flatnonzero(earth&(fold==f))
            rng=np.random.default_rng(seed+int(f))
            shared=balanced(t,y,256,rng)
            rest=np.setdiff1d(t,shared)
            pairs=matches(rest,y,bills,mask,group,rng)
            a,b=pairs[rng.choice(len(pairs),256,replace=False)].T
            contexts={"balanced":balanced(rest,y,256,rng),"exceptions":np.concatenate([a,b])}
            for name,extra in contexts.items():
                support=np.sort(np.concatenate([shared,extra]))
                assert len(np.unique(support))==1024 and y[support].sum()==512
                assert not np.intersect1d(group[support],group[q]).size
                model=TabPFNClassifier.create_default_for_version(
                    ModelVersion.V2, model_path=weight, n_estimators=1, device="cpu",
                    random_state=291, ignore_pretraining_limits=True, n_preprocessing_jobs=1,
                    inference_precision=torch.float32,
                    categorical_features_indices=[frame.columns.get_loc(c) for c in CATEGORICAL],
                    inference_config={"ENABLE_GPU_PREPROCESSING":False, "TRANSFORM_TEXT":False,
                                      "TRANSFORM_DATES":False, "SUBSAMPLE_SAMPLES":None})
                model.fit(frame.iloc[support], y[support].astype(int))
                chunks=[model.predict_proba(frame.iloc[part])[:,1] for part in np.array_split(q,4)]
                p=np.concatenate(chunks)
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                predictions[name][q]=p
                np.savez_compressed(out/f"{seed}_{f}_{name}.npz",support=row[support],query=row[q],p=p)
                print(seed,f,name,len(q),float(((p>=.5)==y[q]).mean()),flush=True)
        for name,p in list(predictions.items()):
            predictions[name+"_quarter"]=.75*base+.25*p
        scores={}
        for name,p in predictions.items():
            ok=(p>=.5)==y
            before=(base>=.5)==y
            scores[name]=dict(accuracy=float(ok.mean()),fixed=int((ok&~before).sum()),
                              harmed=int((~ok&before).sum()),earth_repeat=int((ok&repeat).sum()))
        results[seed]=scores
        write(ROOT/"reports/nov23_earth_tabpfn_context/result.json",results)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**predictions)


if __name__=="__main__":
    torch.set_num_threads(3)
    with threadpool_limits(3):
        run()
