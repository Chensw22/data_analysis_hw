"""Score the pinned native Orion checkpoint on unseen passenger bookings using CPU."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"
OUT=ART/"nov23_pretrained_recheck"
REPORT=ROOT/"reports/nov23_pretrained_recheck"


def evaluate(prob, base, y, hard):
    hit=(prob>=.5)==y
    before=(base>=.5)==y
    return dict(rows=len(y),accuracy=float(hit.mean()),correct=int(hit.sum()),
                fixed=int((hit&~before).sum()),harmed=int((~hit&before).sum()),
                in252_correct=int(hit[hard].sum()))


def run(seed):
    import torch
    from orion_bix_probe import CHECKPOINT, MANIFEST_SHA, VENDOR, WEIGHT_SHA, sha
    torch.set_num_threads(3)
    assert sha(VENDOR/"SOURCE.json")==MANIFEST_SHA and sha(CHECKPOINT)==WEIGHT_SHA
    manifest=json.loads((VENDOR/"SOURCE.json").read_text())
    for name,checksum in manifest["vendor_files_sha256"].items():
        assert sha(VENDOR/name)==checksum
    sys.path.insert(0,str(VENDOR/"src"))
    from orion_bix.sklearn.classifier import OrionBixClassifier
    import orion_bix
    assert Path(orion_bix.__file__).resolve().is_relative_to(VENDOR/"src")
    ref=np.load(ART/f"nov23_group_kernel/{seed}.npz")
    row,y,fold,base=(ref[k] for k in ("row","y","fold","base"))
    inventory=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    np.testing.assert_array_equal(row,inventory.row)
    np.testing.assert_array_equal(y,inventory.target)
    with np.load(ART/"nov23_archive_recovery/masks.npz") as masks:
        np.testing.assert_array_equal(row,masks["row"])
        hard=masks["still_unresolved"]
    df=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Transported","Name"}).iloc[row].copy()
    group=df.pop("PassengerId").str.split("_").str[0].to_numpy()
    cabin=df.pop("Cabin").str.split("/",expand=True)
    df["Deck"],df["CabinNumber"],df["Side"]=cabin[0],pd.to_numeric(cabin[1]),cabin[2]
    for name in ["CryoSleep","VIP"]:
        df[name]=df[name].map({True:"yes",False:"no"})
    OUT.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    p=np.full(len(row),np.nan); evidence=[]
    for f in np.unique(fold):
        t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
        assert not np.intersect1d(group[t],group[q]).size
        support=np.sort(np.random.default_rng(seed+int(f)).choice(t,1536,replace=False))
        model=OrionBixClassifier(n_estimators=2,batch_size=1,use_amp=False,device="cpu",n_jobs=3,
            model_path=str(CHECKPOINT),allow_auto_download=False,checkpoint_version=CHECKPOINT.name,random_state=121)
        model.fit(df.iloc[support],y[support].astype(int))
        assert list(model.classes_)==[0,1] and model.model_.row_attention_type=="standard"
        for part in np.array_split(q,6):
            pr=model.predict_proba(df.iloc[part])
            assert pr.shape==(len(part),2) and np.isfinite(pr).all() and np.allclose(pr.sum(1),1,atol=1e-5)
            p[part]=pr[:,1]
        path=OUT/f"{seed}_fold{f}.npz"
        np.savez_compressed(path,training=row[support],query=row[q],p=p[q],base=base[q],y=y[q])
        stats=evaluate(p[q],base[q],y[q],hard[q])
        evidence.append(dict(fold=int(f),sha256=sha(path),metrics=stats,train_rows=len(support)))
        print(seed,f,stats,flush=True)
    assert np.isfinite(p).all()
    np.savez_compressed(OUT/f"{seed}.npz",row=row,y=y,fold=fold,base=base,orion=p)
    results=dict(checkpoint=WEIGHT_SHA,code=manifest["commit"],folds=evidence,metrics={k:evaluate(v,base,y,hard) for k,v in
                 [("direct",p),("quarter",.75*base+.25*p),("base",base)]})
    (REPORT/f"{seed}.json").write_text(json.dumps(results,indent=2)+"\n")
    print(results["metrics"],flush=True)


def summarize():
    # Use the main analysis environment. Membership of the error cohorts is
    # calculated only after the checkpoint emitted every passenger probability.
    from archive_error_recovery_n23 import panel
    rows,y,b,bank,hard,names,_=panel()
    base_hit=(b>=.5)==y
    hit=(bank>=.5)==y[None,:,None]
    qualify=(hit.mean(1)>=base_hit.mean(1)[:,None]-.01).all(0)
    weak=~base_hit.any(0)&hit.all(0).any(1)&~hit.all(0)[:,qualify].any(1)
    deep=~base_hit.any(0)&~hit.any(axis=(0,2))
    assert (weak.sum(),deep.sum())==(335,252)
    data=[]
    for seed in (20261112,20261115,20261116):
        with np.load(OUT/f"{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],rows)
            np.testing.assert_array_equal(z["y"],y)
            data.append(z["orion"])
    direct=np.array(data)
    report={}
    for name,score in [("direct",direct),("quarter",.75*b+.25*direct)]:
        correct=(score>=.5)==y
        report[name]={}
        for title,mask in [("baseline_always_wrong",~base_hit.any(0)),("weak335",weak),("hard252",deep),("hard365",hard)]:
            out=correct[:,mask]
            report[name][title]=dict(rows=int(mask.sum()),hits=out.sum(1).tolist(),
                                     stable=int(out.all(0).sum()),ever=int(out.any(0).sum()))
    (REPORT/"overlap.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("seeds",nargs="*",type=int,default=[20261112,20261115,20261116])
    parser.add_argument("--summarize",action="store_true")
    args=parser.parse_args()
    if args.summarize:
        summarize()
    else:
        for selected in args.seeds:
            run(selected)
