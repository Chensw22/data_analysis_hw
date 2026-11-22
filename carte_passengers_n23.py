"""Run the native single-member CPU CARTE classifier using compact exact fastText vectors."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from types import ModuleType
import warnings

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"artifacts/nov23_carte"
REPORT=ROOT/"reports/nov23_carte"
sys.path.insert(0,str(OUT/"packages"))

# Import only the required official modules. CARTE package initializers also
# import unrelated benchmark estimators and visualization dependencies.
for name,folder in [("carte_ai",""),("carte_ai.src","src"),("carte_ai.configs","configs")]:
    namespace=ModuleType(name)
    namespace.__path__=[str(OUT/"packages/carte_ai"/folder)]
    sys.modules[name]=namespace

import numpy as np
import pandas as pd
import torch
from carte_ai.src.carte_estimator import CARTEClassifier
from carte_ai.src.carte_table_to_graph import Table2GraphTransformer


class ObservedClassifier(CARTEClassifier):
    def _eval(self, model, dataset):
        loss=super()._eval(model,dataset)
        self.loss_history.append(float(loss))
        if self.capture_prefix and len(self.loss_history)<=14 and (len(self.loss_history)==1 or loss<min(self.loss_history[:-1])):
            self.prefix_model=copy.deepcopy(model).eval()
        return loss


def compare_training_lengths(seed, row, y, base, fold, repeat, earth):
    from archive_error_recovery_n23 import panel
    archive_row, archive_y, archive_base, archive_bank, *_ = panel()
    np.testing.assert_array_equal(archive_row, row)
    np.testing.assert_array_equal(archive_y, y)
    bank_hit = (archive_bank >= .5) == y[None, :, None]
    hard = ~bank_hit.any(axis=(0, 2)) & ~((archive_base >= .5) == y).any(axis=0)
    assert hard.sum() == 252
    prefix = np.empty(len(y))
    full = np.empty(len(y))
    folds = []
    for f in range(3):
        select = fold == f
        trajectory = json.loads((REPORT/f"{seed}_{f}_extended_trajectory.json").read_text())
        losses = trajectory["negative_auroc"]
        with np.load(OUT/f"{seed}_{f}_extended.npz") as z:
            np.testing.assert_array_equal(z["query"], row[select])
            prefix[select] = z["matched14"]
            full[select] = z["p"]
            elapsed = float(z["seconds"])
        folds.append(dict(fold=f, epochs=len(losses), best_epoch=int(np.argmin(losses))+1,
                          prefix_best_epoch=int(np.argmin(losses[:14]))+1,
                          validation_auroc10=-losses[9], validation_auroc14=-losses[13],
                          best_validation_auroc=-min(losses), seconds=elapsed))

    metrics = {}
    before = (base >= .5) == y
    for name, weight in [("direct", 1.), ("quarter", .25)]:
        def correct(p):
            return (weight*p + (1.-weight)*base >= .5) == y
        short, long = correct(prefix), correct(full)
        counts = {}
        for label, result in [("prefix14", short), ("extended", long)]:
            counts[label] = dict(correct=int(result.sum()), accuracy=float(result.mean()),
                                 fixed=int((result & ~before).sum()), harmed=int((~result & before).sum()),
                                 repeat1070=int(result[repeat].sum()), earth430=int(result[earth].sum()),
                                 archive252=int(result[hard].sum()))
        counts["full_vs_prefix"] = dict(fixed=int((long & ~short).sum()), harmed=int((~long & short).sum()))
        metrics[name] = counts
    report = dict(seed=seed, rows=len(row), folds=folds, metrics=metrics)
    (REPORT/f"{seed}_extended_comparison.json").write_text(json.dumps(report,indent=2)+"\n")


def run(seeds, scratch=False, trace=False, extended=False):
    suffix="_scratch" if scratch else "_extended" if extended else "_trace" if trace else ""
    torch.set_num_threads(3)
    warnings.filterwarnings("ignore",message=".*CUDA.*")
    warnings.filterwarnings("ignore",category=FutureWarning,module="carte_ai")
    vec=OUT/"manifest_words.bin"
    assert hashlib.sha256(vec.read_bytes()).hexdigest()==json.loads((REPORT/"words.json").read_text())["compact_sha256"]
    ckpt=OUT/"packages/carte_ai/data/etc/kg_pretrained.pt"
    ckhash=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert ckhash==json.loads((REPORT/"weights.json").read_text())["weights_sha256"]
    idx=pd.read_csv(ROOT/"artifacts/nov23_global_error_inventory.csv")
    row,y=idx.row.to_numpy(),idx.target.to_numpy(bool)
    df=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Transported","Name"}).iloc[row].reset_index(drop=True)
    group=df.pop("PassengerId").str.split("_").str[0].to_numpy()
    cabin=df.pop("Cabin").str.split("/",expand=True)
    df["Deck"],df["CabinNumber"],df["Side"]=cabin[0],pd.to_numeric(cabin[1]),cabin[2]
    for name in ["CryoSleep","VIP"]:df[name]=df[name].map({True:"true",False:"false"})
    repeat=np.ones(len(y),bool)
    for seed in (20261112,20261115,20261116):
        with np.load(ROOT/f"artifacts/nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(row,z["row"])
            np.testing.assert_array_equal(y,z["y"])
            repeat &= (z["base"]>=.5)!=y
    for seed in seeds:
        with np.load(ROOT/f"artifacts/nov23_identical_bills/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            mask=z["eligible"] & df.HomePlanet.eq("Earth").to_numpy() & repeat
            base,fold=z["baseline"],z["fold"]
        assert mask.sum()==430 and repeat.sum()==1070
        prob=np.full(len(y),np.nan)
        for f in range(3):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(group[t],group[q]).size
            dest=OUT/f"{seed}_{f}{suffix}.npz"
            if not dest.exists():
                start=time.monotonic()
                prep=Table2GraphTransformer(fasttext_model_path=str(vec))
                train=prep.fit_transform(df.iloc[t],y[t].astype(np.float32))
                query=prep.transform(df.iloc[q])
                print("graphs ready",seed,f,round(time.monotonic()-start,1),flush=True)
                clf=ObservedClassifier(load_pretrain=not scratch,num_model=1,max_epoch=500 if extended else 14,early_stopping_patience=40 if extended else 3,batch_size=96,
                                    n_jobs=1,device="cpu",random_state=804+f,disable_pbar=False)
                assert clf._estimator_type=="classifier" and clf.load_pretrain== (not scratch)
                clf.loss_history=[]
                clf.capture_prefix=extended
                clf.fit(train,y[t].astype(int))
                (REPORT/f"{seed}_{f}{suffix}_trajectory.json").write_text(json.dumps(dict(negative_auroc=clf.loss_history,best_epoch=1+int(np.argmin(clf.loss_history))))+"\n")
                p=np.concatenate([clf.predict_proba(query[k:k+96]) for k in range(0,len(q),96)])
                assert p.shape==(len(q),) and np.isfinite(p).all() and np.unique(p).size>30 and ((p>=0)&(p<=1)).all()
                prefix={}
                if extended:
                    clf.model_list_=[clf.prefix_model]
                    prefix["matched14"]=np.concatenate([clf.predict_proba(query[k:k+96]) for k in range(0,len(q),96)])
                np.savez_compressed(dest,train=row[t],query=row[q],p=p,weight=ckhash,seconds=time.monotonic()-start,**prefix)
                del prep,train,query,clf
                gc.collect()
            with np.load(dest) as z:
                np.testing.assert_array_equal(z["train"],row[t])
                np.testing.assert_array_equal(z["query"],row[q])
                assert z["weight"]==ckhash
                prob[q]=z["p"]
                print(seed,f,len(q),round(((prob[q]>=.5)==y[q]).mean(),4),round(float(z["seconds"]),1),flush=True)
        out={}
        before=(base>=.5)==y
        for tag,p in [("direct",prob),("quarter",.75*base+.25*prob)]:
            hit=(p>=.5)==y
            out[tag]=dict(accuracy=float(hit.mean()),fixed=int((hit&~before).sum()),harmed=int((~hit&before).sum()),
                        repeat1070=int(hit[repeat].sum()),earth430=int(hit[mask].sum()))
        np.savez_compressed(OUT/f"{seed}{suffix}.npz",row=row,y=y,fold=fold,base=base,p=prob)
        (REPORT/f"{seed}{suffix}.json").write_text(json.dumps(out,indent=2)+"\n")
        if extended:
            compare_training_lengths(seed,row,y,base,fold,repeat,mask)
        print(seed,out,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("seeds",nargs="*",type=int,default=[20261112,20261115,20261116])
    parser.add_argument("--scratch",action="store_true")
    parser.add_argument("--trace",action="store_true")
    parser.add_argument("--extended",action="store_true")
    args=parser.parse_args()
    run(args.seeds,args.scratch,args.trace,args.extended)
