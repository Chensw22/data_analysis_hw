"""Give a frozen foundation learner two honest out-of-booking predictions as input fields."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.model_selection import GroupKFold

import mitra_candidate as mitra
from boarding_sequence_n23 import hard_rows
from exaone_validation_data import compact_features
from relational_error_corrector import write

ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"


def predictions(frame,y,group,t,q,seed,match_single_teacher=False):
    views=[list(frame.columns),["RoomService","FoodCourt","Spa","VRDeck","ShoppingMall","Age","CryoSleep"]]
    train=np.full((len(t),len(views)),np.nan)
    query=np.zeros((len(q),len(views)))
    splits=GroupKFold(3,shuffle=True,random_state=seed).split(t,groups=group[t])
    query_groups,group_index=np.unique(group[q],return_inverse=True)
    assigned=np.random.default_rng(seed).permutation(len(query_groups))[group_index] % 3
    for inner,(a,b) in enumerate(splits):
        assert not np.intersect1d(group[t[a]],group[t[b]]).size
        for j,columns in enumerate(views):
            fit=CatBoostClassifier(iterations=420,depth=5,learning_rate=.055,thread_count=3,
                                   verbose=False,allow_writing_files=False,random_seed=235)
            cats=[c for c in columns if c in mitra.CATEGORICAL]
            fit.fit(frame.iloc[t[a]][columns],y[t[a]],cat_features=cats)
            train[b,j]=fit.predict_proba(frame.iloc[t[b]][columns])[:,1]
            query_prediction=fit.predict_proba(frame.iloc[q][columns])[:,1]
            if match_single_teacher:
                take=assigned==inner
                query[take,j]=query_prediction[take]
            else:
                query[:,j]+=query_prediction/3
    assert np.isfinite(train).all()
    values=[logit(np.clip(v,.005,.995)) for v in (train,query)]
    standardized=[(v-values[0].mean(0))/values[0].std(0).clip(.01) for v in values]
    return (*standardized,train[:,0],query[:,0])


def run(seed):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal device unavailable for the foundation comparison")
    torch.set_num_threads(3)
    torch.mps.set_per_process_memory_fraction(.65)
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ["Name","Transported"])
    with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
        dev,y,fold,base=[z[k] for k in ("row","y","fold","base")]
    with np.load(ART/f"top5_group_phase/refs/v44/group_{seed}.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"],dev)
        np.testing.assert_array_equal(z["fold_id"],fold)
        tree=z["prob"]
    x=compact_features(raw.iloc[dev]).reset_index(drop=True)
    group=raw.PassengerId.iloc[dev].str.split("_").str[0].to_numpy()
    hard=hard_rows(dev,y)
    model,_=mitra.load_initial_model(mitra.CHECKPOINT_DIR,"mps")
    out=ART/"nov23_mitra_teachers"
    out.mkdir(exist_ok=True)
    outputs={key:np.full(len(dev),np.nan) for key in ["plain","teachers"]}
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.intersect1d(group[t],group[q]).size
        train_signal,query_signal,_,_=predictions(x,y,group,t,q,seed+int(k))
        prep=mitra.MitraPreprocessor().fit(x.iloc[t],y[t].astype(int),seed=seed+int(k))
        support,query=prep.transform(x.iloc[t]),prep.transform(x.iloc[q])
        for name in outputs:
            s,v=support,query
            if name=="teachers":
                s=np.column_stack([s,train_signal])
                v=np.column_stack([v,query_signal])
            s,v=[torch.as_tensor(a.astype(np.float32),device="mps") for a in (s,v)]
            targets=torch.as_tensor(y[t].astype(np.int64),device="mps")
            outputs[name][q]=mitra.predict_arrays(model,s,targets,v,chunk=768)[:,1]
            del s,v,targets
            torch.mps.empty_cache()
            print(seed,int(k),name,round(float(((outputs[name][q]>=.5)==y[q]).mean()),6),flush=True)
        np.savez_compressed(out/f"{seed}_fold{k}.npz",row=dev[q],y=y[q],**{n:v[q] for n,v in outputs.items()})
    results={}
    for name,p in outputs.items():
        assert np.isfinite(p).all()
        for label,score in [(name,p),(name+"_with_v44",.5*(p+tree))]:
            ok=(score>=.5)==y
            bad=(base>=.5)!=y
            results[label]=dict(accuracy=float(ok.mean()),fixed=int((bad&ok).sum()),harmed=int((~bad&~ok).sum()),
                                hard447_correct=int(ok[hard].sum()),fold_accuracy=[float(ok[fold==k].mean()) for k in np.unique(fold)])
    np.savez_compressed(out/f"{seed}.npz",row=dev,y=y,fold=fold,base=base,v44=tree,**outputs)
    write(ROOT/f"reports/nov23_mitra_teachers/{seed}.json",dict(results=results,checkpoint_sha256=mitra.WEIGHT_SHA,
          note="Inner teacher folds exclude entire bookings. The two additional fields are trained only inside the outer training partition."))
    print(results,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=20261112,choices=[20261112,20261115,20261116])
    run(parser.parse_args().seed)
