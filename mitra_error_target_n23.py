"""Use a foundation learner to predict out-of-booking errors of an inner-fit teacher."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import mitra_candidate as mitra
from boarding_sequence_n23 import hard_rows
from exaone_validation_data import compact_features
from mitra_crossfit_teachers_n23 import predictions
from relational_error_corrector import write

ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"


def run(seed=20261112):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal device unavailable")
    torch.set_num_threads(3)
    torch.mps.set_per_process_memory_fraction(.65)
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ["Name","Transported"])
    with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
        dev,y,fold,base=[z[k] for k in ("row","y","fold","base")]
    x=compact_features(raw.iloc[dev]).reset_index(drop=True)
    group=raw.PassengerId.iloc[dev].str.split("_").str[0].to_numpy()
    hard=hard_rows(dev,y)
    model,_=mitra.load_initial_model(mitra.CHECKPOINT_DIR,"mps")
    estimates={name:np.full(len(y),np.nan) for name in ["teacher","matched_outcome","error_target"]}
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.intersect1d(group[t],group[q]).size
        train_score,query_score,tp,qp=predictions(x,y,group,t,q,seed+int(k),match_single_teacher=True)
        wrong=((tp>=.5)!=y[t]).astype(int)
        prep=mitra.MitraPreprocessor().fit(x.iloc[t],y[t].astype(int),seed=seed+int(k))
        # Make the teacher's hard decision explicit so that an output of "wrong"
        # has exactly the same interpretation during support and query inference.
        s=np.column_stack([prep.transform(x.iloc[t]),train_score,(tp>=.5)])
        v=np.column_stack([prep.transform(x.iloc[q]),query_score,(qp>=.5)])
        s,v=[torch.as_tensor(a.astype(np.float32),device="mps") for a in [s,v]]
        estimates["matched_outcome"][q]=mitra.predict_arrays(model,s,torch.as_tensor(y[t].astype(int),device="mps"),v,chunk=768)[:,1]
        chance=mitra.predict_arrays(model,s,torch.as_tensor(wrong,device="mps"),v,chunk=768)[:,1]
        estimates["teacher"][q]=qp
        estimates["error_target"][q]=np.where(qp>=.5,1-chance,chance)
        del s,v
        torch.mps.empty_cache()
        print(int(k),float(((estimates["error_target"][q]>=.5)==y[q]).mean()),flush=True)
    output=ART/"nov23_mitra_errors"
    output.mkdir(exist_ok=True)
    results={}
    for name,p in estimates.items():
        assert np.isfinite(p).all()
        for label,score in [(name,p),(name+"_blend",(.75*base+.25*p))]:
            ok=(score>=.5)==y
            wrong=(base>=.5)!=y
            results[label]=dict(accuracy=float(ok.mean()),fixed=int((wrong&ok).sum()),
                                harmed=int((~wrong&~ok).sum()),hard447_correct=int(ok[hard].sum()),
                                fold_accuracy=[float(ok[fold==k].mean()) for k in np.unique(fold)])
    np.savez_compressed(output/f"{seed}.npz",row=dev,y=y,fold=fold,**estimates)
    direct=(estimates["matched_outcome"]>=.5)==y
    err=(estimates["error_target"]>=.5)==y
    results["identical_inputs_error_vs_outcome"]=dict(fixed=int((err&~direct).sum()),harmed=int((~err&direct).sum()),
        hard447_fixed=int((hard&err&~direct).sum()),hard447_harmed=int((hard&~err&direct).sum()))
    write(ROOT/f"reports/nov23_mitra_errors/{seed}.json",results)
    print(results,flush=True)


if __name__=="__main__":
    run()
