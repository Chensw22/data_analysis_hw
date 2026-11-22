"""Evaluate the pinned row-attention TabSwift model on every development booking."""
import hashlib
from pathlib import Path
import sys

# The upstream release uses BaseEstimator._validate_data, retained by sklearn 1.6.
# Install the compatible wheel in isolation; other experiments keep their own sklearn.
COMPAT=Path(__file__).resolve().parent/"artifacts/nov23_tabswift/sklearn16"
sys.path.insert(0,str(COMPAT))
import sklearn
if sklearn.__version__!="1.6.1":
    raise RuntimeError("Install scikit-learn==1.6.1 in the TabSwift sklearn16 target")
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from setup_tabswift_n23 import OUT, SOURCE, WEIGHTS, DIGEST
from exaone_validation_data import compact_features, CATEGORICAL
from population_rescue_map_n23 import ART, ROOT, read_panel
from grouped_five_way_coverage_n23 import verify
from relational_error_corrector import write


def prepare(feat,t):
    # Match upstream's numeric input convention: fit category codes and fill
    # continuous missing cells using only the current training bookings.
    cat=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)
    cat.fit(feat.iloc[t][CATEGORICAL])
    numeric=feat.drop(columns=CATEGORICAL)
    num=numeric.fillna(numeric.iloc[t].mean()).to_numpy()
    x=np.column_stack([num,cat.transform(feat[CATEGORICAL])])
    assert np.isfinite(x).all()
    return x


def run(seed, estimators):
    from tabswift.classifier import TabSwiftClassifier
    checkpoint=OUT/"swift.ckpt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    xraw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ["Name","Transported"])
    rows=pd.read_csv(ART/"nov23_global_error_inventory.csv",usecols=["row"]).row.to_numpy()
    feat=compact_features(xraw).iloc[rows].reset_index(drop=True)
    groups=xraw.PassengerId.iloc[rows].str.split("_").str[0].to_numpy()
    y,hits,others=read_panel()
    with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
        fold,base=ref["fold"],ref["base"]
        verify(ref,rows,y,fold)
    hard=~hits.any(0)&~others.any(axis=(0,1))
    probability=np.full(len(y),np.nan)
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.intersect1d(groups[t],groups[q]).size
        model=TabSwiftClassifier(n_estimators=estimators,device="cpu",batch_size=1,model_path=checkpoint,
                allow_auto_download=False,use_amp=False,random_state=43+int(k))
        print("training support",seed,int(k),len(t),"queries",len(q),flush=True)
        x=prepare(feat,t)
        model.fit(x[t],y[t])
        probability[q]=model.predict_proba(x[q])[:,1]
        print("completed fold",int(k),float(np.mean((probability[q]>=.5)==y[q])),flush=True)
    assert np.isfinite(probability).all()
    old=(base>=.5)==y
    metrics={}
    for name,p in [("tabswift",probability),("equal",(base+probability)/2),("quarter",.75*base+.25*probability)]:
        ok=(p>=.5)==y
        metrics[name]=dict(accuracy=float(ok.mean()),fixed=int((ok&~old).sum()),harmed=int((~ok&old).sum()),
                           persistent1070=int(ok[~hits.any(0)].sum()),unresolved466=int(ok[hard].sum()))
    key=f"{seed}_views{estimators}"
    np.savez_compressed(OUT/f"{key}.npz",row=rows,y=y,fold=fold,base=base,tabswift=probability)
    write(ROOT/f"reports/nov23_tabswift/{key}.json",dict(seed=seed,ensemble_views=estimators,
          source_revision=SOURCE,weights_revision=WEIGHTS,checkpoint_sha256=DIGEST,metrics=metrics))
    print(metrics,flush=True)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=20261112)
    parser.add_argument("--views",type=int,default=4)
    args=parser.parse_args()
    sys.path.insert(0,str(OUT/SOURCE))
    with threadpool_limits(2):
        run(args.seed,args.views)
