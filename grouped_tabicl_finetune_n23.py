"""Fine-tune a second pretrained tabular network, selecting epochs on disjoint bookings."""
import gc
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from tabicl import FinetunedTabICLClassifier, TabICLClassifier
from threadpoolctl import threadpool_limits

from tabicl_routed_context_n23 import ART, ROOT, WEIGHT, DIGEST, CAT


class TracedTabICL(FinetunedTabICLClassifier):
    def _validate_current_model(self, *args, **kwargs):
        metrics = super()._validate_current_model(*args, **kwargs)
        self.validation.append(float(metrics.primary))
        print(f"fold {self.outer_fold}, evaluation {len(self.validation)-1}, inner log loss {-metrics.primary:.5f}",flush=True)
        return metrics


def predict(weight, frame, y, t, q, seed):
    net = TabICLClassifier(n_estimators=2, model_path=weight, allow_auto_download=False,
                           device="cpu", n_jobs=2, batch_size=1, random_state=seed, use_fa3=False,
                           kv_cache=False, use_amp=False)
    net.fit(frame.iloc[t], y[t])
    return net.predict_proba(frame.iloc[q])[:,1]


def run():
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    seed = 20261112
    with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
        row, y, fold, base = (z[k] for k in ["row","y","fold","base"])
    raw = pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"}).iloc[row].reset_index(drop=True)
    group = raw.PassengerId.str.split("_").str[0].to_numpy()
    frame = raw.drop(columns=["Cabin","PassengerId"]).copy()
    frame[["Deck","CabinNumber","Side"]] = raw.Cabin.str.split("/",expand=True)
    frame.CabinNumber = pd.to_numeric(frame.CabinNumber)
    frame[CAT] = frame[CAT].fillna("__MISSING__").astype(str)
    native, adapted = np.full(len(y),np.nan), np.full(len(y),np.nan)
    out = ART/"nov23_tabicl_finetune"
    report = ROOT/"reports/nov23_tabicl_finetune"
    for folder in [out,report]:
        folder.mkdir(exist_ok=True)
    trace = []
    for k in np.unique(fold):
        t, q = np.flatnonzero(fold!=k), np.flatnonzero(fold==k)
        split = GroupShuffleSplit(n_splits=1,test_size=.15,random_state=917)
        train, check = next(split.split(t,groups=group[t]))
        a, b = t[train],t[check]
        assert not (set(group[a])&set(group[b]) or set(group[t])&set(group[q]))
        torch.manual_seed(43+int(k))
        net = TracedTabICL(epochs=4, patience=3, max_data_size=512, n_estimators_finetune=1,
                n_estimators_validation=2,n_estimators_inference=2, freeze_col=True, freeze_row=True,
                freeze_icl=False, learning_rate=1e-5, amp=False, eval_metric="log_loss", device="cpu",
                save_interval=10,model_path=WEIGHT,allow_auto_download=False,random_state=43+int(k),
                extra_classifier_kwargs=dict(batch_size=1,n_jobs=2,kv_cache=False,use_amp=False,use_fa3=False))
        net.validation = []
        net.outer_fold = int(k)
        with TemporaryDirectory(dir=out) as tmp:
            net.fit(frame.iloc[a],y[a],X_val=frame.iloc[b],y_val=y[b],output_dir=Path(tmp))
            trace.append(dict(fold=int(k),inner_fit=len(a),inner_validation=len(b),inference_support=len(t),
                              inner_log_loss=(-np.asarray(net.validation)).tolist()))
            del net
            gc.collect()
            native[q] = predict(WEIGHT,frame,y,t,q,43+int(k))
            adapted[q] = predict(Path(tmp)/"best.ckpt",frame,y,t,q,43+int(k))
        print("completed outer fold",int(k),"adapted accuracy",np.mean((adapted[q]>=.5)==y[q]),flush=True)
        np.savez_compressed(out/f"fold{int(k)}.npz",row=row[q],y=y[q],native=native[q],adapted=adapted[q])
    with np.load(ART/f"nov23_tabicl_context/{seed}.npz") as z:
        np.testing.assert_array_equal(z["row"],row)
        np.testing.assert_array_equal(z["fold"],fold)
        np.testing.assert_allclose(native,z["global_context"],atol=1e-5)
    np.savez_compressed(out/"oof.npz",row=row,y=y,fold=fold,base=base,native=native,adapted=adapted)
    old=(base>=.5)==y
    metrics={}
    for name,p in [("native",native),("adapted",adapted),("native_blend",(native+base)/2),("adapted_blend",(adapted+base)/2)]:
        ok=(p>=.5)==y
        assert np.isfinite(p).all()
        metrics[name]=dict(accuracy=float(ok.mean()),fixed=int((ok&~old).sum()),harmed=int((~ok&old).sum()))
    (report/"result.json").write_text(json.dumps(dict(seed=seed,checkpoint_sha256=DIGEST,metrics=metrics,folds=trace),indent=2)+"\n")
    print(metrics,flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
