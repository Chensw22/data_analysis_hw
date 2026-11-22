"""Evaluate the unchanged v93 fine-tuning recipe on a frozen whole-booking split."""
import fcntl
import hashlib
import json
import os
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch

import mitra_candidate as base
import mitra_support3072 as core
from exaone_validation_data import load_data
from grouped_sparse_rules_n23 import remaining

ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"
OUT=ART/"nov23_group_support3072"
REPORT=ROOT/"reports/nov23_group_support3072"
SEED=20261112


def write(path,obj):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2)+"\n")
    os.replace(tmp,path)


def now():
    return datetime.now(timezone.utc).isoformat()


def evaluate():
    source=load_data()
    rows=source["dev"]
    labels=source["y"].astype(bool)
    groups=np.asarray(source["groups"]).astype(int)
    reference=ART/f"top5_group_phase/refs/v44/group_{SEED}.npz"
    foundation=ART/f"nov23_group_kernel/{SEED}.npz"
    rules=ART/f"nov23_sparse_rules/{SEED}.npz"
    with np.load(reference) as z:
        np.testing.assert_array_equal(z["dev_ids"],rows)
        np.testing.assert_array_equal(z["group_ids"].astype(int),groups)
        fold,tree=z["fold_id"],z["prob"]
    with np.load(foundation) as z, np.load(rules) as r:
        for data in [z,r]:
            for k,v in [("row",rows),("y",labels),("fold",fold)]:
                np.testing.assert_array_equal(data[k],v)
        anchor=z["base"]
        sparse_rule=r["rules"]
    previous=json.loads((ROOT/"reports/run_056/protocol.json").read_text())
    assert core.SETTINGS==previous["settings"]
    for name,digest in core.source_hashes().items():
        assert digest=={**previous["frozen"]["resource"]["sources"],**previous["frozen"]["sources"]}[name]
    spec=dict(seed=SEED,rows=base.ids_hash(rows),target=hashlib.sha256(labels.tobytes()).hexdigest(),
              model=core.SETTINGS,source_hashes=core.source_hashes(),checkpoint_sha256=base.WEIGHT_SHA,
              references={str(p.relative_to(ROOT)):base.sha256(p) for p in [reference,foundation,rules]},
              assessments=["v93","equal_v93_v44","three_quarters_equal_pair_one_quarter_rules"],
              partitions="Whole bookings separate in outer folds. The unchanged v93 inner validation selects steps on a random split within outer training.")
    spec_id=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    protocol=REPORT/"protocol.json"
    if protocol.exists() and json.loads(protocol.read_text())!=spec:
        raise ValueError("Existing protocol differs from this run")
    write(protocol,spec)
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal unavailable for the unchanged v93 recipe")
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.65)
    p=np.full(len(rows),np.nan)
    for k in np.unique(fold):
        k=int(k)
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.intersect1d(groups[t],groups[q]).size
        path=OUT/f"fold{k}.npz"
        record=REPORT/f"fold{k}.json"
        if record.exists():
            saved=json.loads(record.read_text())
            if saved["protocol"]!=spec_id or base.sha256(path)!=saved["sha256"]:
                raise ValueError("Completed fold integrity check failed")
        else:
            log=OUT/f"fold{k}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
            write(OUT/"status.json",dict(pid=os.getpid(),fold=k,started=now(),log=log.name,protocol=spec_id))
            print(f"Starting whole-booking fold {k}; {len(t)} training, {len(q)} validation",flush=True)
            model=core.fit_support3072(source["features"].iloc[rows[t]],labels[t].astype(int),rows[t],
                                       SEED+k,device="mps",log_path=log,max_steps=50)
            try:
                assert len(model.support)==len(t) and np.array_equal(model.labels,labels[t])
                values=model.predict_proba(source["features"].iloc[rows[q]])[:,1]
                if values.shape!=(len(q),) or not np.isfinite(values).all() or not ((values>=0)&(values<=1)).all():
                    raise ValueError("Incomplete fold probabilities")
                with path.with_suffix(".tmp").open("wb") as f:
                    np.savez_compressed(f,row=rows[q],train=rows[t],p=values)
                path.with_suffix(".tmp").replace(path)
                write(record,dict(protocol=spec_id,fold=k,sha256=base.sha256(path),training_rows=len(t),query_rows=len(q),
                                group_overlap=0,selected_steps=int(model.metadata["selected_steps"]),
                                accuracy=float(((values>=.5)==labels[q]).mean()),completed=now()))
            finally:
                model.release()
                del model
                base.release_memory()
        with np.load(path) as z:
            np.testing.assert_array_equal(z["row"],rows[q])
            np.testing.assert_array_equal(z["train"],rows[t])
            p[q]=z["p"]
        print("Completed fold",k,json.loads(record.read_text())["accuracy"],flush=True)
    assert np.isfinite(p).all()
    hard=remaining(rows,labels)
    pair=.5*(tree+p)
    scores={}
    for name,prob in zip(spec["assessments"],[p,pair,.75*pair+.25*sparse_rule]):
        ok=(prob>=.5)==labels
        bad=(anchor>=.5)!=labels
        scores[name]=dict(accuracy=float(ok.mean()),fixed=int((ok&bad).sum()),harmed=int((~ok&~bad).sum()),
                          hard356_correct=int(ok[hard].sum()),fold_accuracy=[float(ok[fold==k].mean()) for k in np.unique(fold)])
    np.savez_compressed(OUT/"oof.npz",row=rows,y=labels,fold=fold,v93=p,pair=pair,rule_blend=.75*pair+.25*sparse_rule)
    write(REPORT/"result.json",dict(protocol=spec_id,scores=scores,completed=now(),sha256=base.sha256(OUT/"oof.npz")))
    write(OUT/"status.json",dict(state="completed",protocol=spec_id,completed=now()))
    print(scores,flush=True)


if __name__=="__main__":
    for folder in [OUT,REPORT]:
        folder.mkdir(exist_ok=True)
    with (OUT/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        evaluate()
