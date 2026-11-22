"""Built with PriorLabs-TabPFN. Test whether learned training usefulness transfers across architectures."""
import os
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["TABPFN_NO_BROWSER"]="1"
os.environ["TABPFN_MODEL_VERSION"]="v2"
os.environ["TABPFN_MODEL_CACHE_SIZE"]="0"

import hashlib
import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART,ROOT,SEEDS
from earth_bill_order_n23 import BILLS
from exaone_validation_data import CATEGORICAL,compact_features
from relational_error_corrector import write


def select(t, cells, weights, seed):
    rng=np.random.default_rng(seed)
    keys=rng.exponential(size=len(t))
    result={k:[] for k in ("uniform","learned","shuffled")}
    count=np.bincount(cells[t])
    target=1536*count/len(t)
    quota=np.floor(target).astype(int)
    remainder=1536-quota.sum()
    if remainder:
        quota[np.argsort(-(target-quota),kind="stable")[:remainder]]+=1
    for key in np.flatnonzero(quota):
        ix=np.flatnonzero(cells[t]==key)
        w=weights[ix]
        for name,values in [("uniform",np.ones_like(w)),("learned",w),("shuffled",rng.permutation(w))]:
            chosen=ix[np.argsort(keys[ix]/values,kind="stable")[:quota[key]]]
            result[name].extend(t[chosen])
    for name,ids in result.items():
        result[name]=np.sort(ids)
        assert len(set(ids))==1536
        np.testing.assert_array_equal(np.bincount(cells[ids],minlength=len(count)),quota)
    return result


def run():
    weight=ART/"pretrained/tabpfn-v2-classifier-v2_default.ckpt"
    assert hashlib.sha256(weight.read_bytes()).hexdigest()=="cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"
    data={}
    for s in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{s}.npz") as z:
            data[s]={k:z[k] for k in ("row","y","base","fold")}
    ref=data[SEEDS[0]];row,y=ref["row"],ref["y"]
    full=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"})
    gr=full.PassengerId.str.split("_").str[0]
    multiple=gr.map(gr.value_counts()).to_numpy()[row]>1
    group=gr.to_numpy()[row]
    raw=full.iloc[row].reset_index(drop=True)
    earth=raw.HomePlanet.eq("Earth").to_numpy()
    cells=pd.DataFrame(dict(planet=raw.HomePlanet.fillna("?"),paid=raw[BILLS].gt(0).any(axis=1),multiple=multiple,y=y))
    cells=cells.groupby(list(cells),sort=True).ngroup().to_numpy()
    frame=compact_features(raw)
    repeat=np.logical_and.reduce([(z["base"]>=.5)!=y for z in data.values()])
    out=ART/"nov23_weighted_tabpfn_manifest"
    out.mkdir(exist_ok=True)
    results={}
    for seed,v in data.items():
        np.testing.assert_array_equal(v["row"],row);np.testing.assert_array_equal(v["y"],y)
        fold,base=v["fold"],v["base"]
        pred={k:np.full(len(row),np.nan) for k in ("uniform","learned","shuffled")}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            with np.load(ART/f"nov23_booking_meta_weights/{seed}_{f}_weights.npz") as z:
                np.testing.assert_array_equal(z["train"],row[t])
                alternatives=select(t,cells,z["aligned"],seed+int(f))
            for name,ids in alternatives.items():
                assert not np.intersect1d(group[ids],group[q]).size
                path=out/f"{seed}_{f}_{name}.npz"
                if path.exists():
                    with np.load(path) as cached:
                        np.testing.assert_array_equal(cached["train"],row[ids])
                        np.testing.assert_array_equal(cached["query"],row[q])
                        p=cached["p"]
                else:
                    model=TabPFNClassifier.create_default_for_version(ModelVersion.V2,model_path=weight,
                        n_estimators=1,device="cpu",random_state=329,n_preprocessing_jobs=1,
                        ignore_pretraining_limits=True,inference_precision=torch.float32,
                        categorical_features_indices=[frame.columns.get_loc(c) for c in CATEGORICAL],
                        inference_config={"ENABLE_GPU_PREPROCESSING":False,"TRANSFORM_TEXT":False,
                                          "TRANSFORM_DATES":False,"SUBSAMPLE_SAMPLES":None})
                    model.fit(frame.iloc[ids],y[ids].astype(int))
                    p=np.concatenate([model.predict_proba(frame.iloc[part])[:,1] for part in np.array_split(q,6)])
                    temp=path.with_suffix(".tmp")
                    with temp.open("wb") as stream:
                        np.savez_compressed(stream,train=row[ids],query=row[q],p=p)
                    temp.replace(path)
                assert p.shape==(len(q),) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                pred[name][q]=p
                print(seed,f,name,round(float(((p>=.5)==y[q]).mean()),4),flush=True)
        for k,p in list(pred.items()):
            pred[k+"_quarter"]=.75*base+.25*p
        with np.load(ART/f"nov23_identical_bills/{seed}.npz") as old:
            np.testing.assert_array_equal(old["row"],row)
            hard=earth&repeat&old["eligible"]
        before=(base>=.5)==y
        scores={}
        for k,p in pred.items():
            ok=(p>=.5)==y
            scores[k]=dict(accuracy=float(ok.mean()),fixed=int((ok&~before).sum()),harmed=int((~ok&before).sum()),
                           repeat=int((ok&repeat).sum()),earth430=int((ok&hard).sum()))
        results[seed]=scores
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**pred)
        write(ROOT/"reports/nov23_weighted_tabpfn_manifest/result.json",results)


if __name__=="__main__":
    torch.set_num_threads(3)
    with threadpool_limits(3):
        run()
