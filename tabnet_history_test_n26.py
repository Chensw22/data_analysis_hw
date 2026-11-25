"""Build out-of-order historical forecasts and weight-reloaded test forecasts for TabNet."""
import argparse
import hashlib
import json
import os
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd

from attentive_tabnet_n26 import MODELS,OUT,fit_recipes,inputs
from full_roster_booking_validation_n23 import ROOT,data

PANEL = ROOT/"artifacts/nov26_tabicl_consensus_review/historical.npz"
SUMMARY = ROOT/"reports/nov23_matched_learning_curve/tabnet_history_test.json"
sys.path.insert(0,str(OUT/"package"))


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()


def historical_indices(raw, labels, groups, k):
    with np.load(PANEL) as z:
        ids,fold=z["ids"],z["fold"]
        np.testing.assert_array_equal(z["passenger"],raw.PassengerId.iloc[ids])
        np.testing.assert_array_equal(z["wrong"],z["anchor"]!=labels[ids])
    q=ids[fold==k]
    ordinary=ids[fold!=k]
    t=ordinary[~np.isin(groups[ordinary],groups[q])]
    assert len(np.unique(q))==len(q)==2318 and not np.isin(groups[t],groups[q]).any()
    return t,q,ordinary


def save(path,t,q,labels,passengers,names,cache):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f".{path.stem}.{os.getpid()}.npz")
    np.savez_compressed(temporary,t=t,q=q,train_y=labels[t],passenger=passengers,names=names,**cache)
    temporary.replace(path)


def train_history(k):
    raw,y,groups,_=data()
    t,q,_=historical_indices(raw,y,groups,k)
    names,cache=fit_recipes(raw,y,groups,t,q,f"historical={k}",query_labels=y[q])
    save(OUT/"history"/f"fold{k}.npz",t,q,y,raw.PassengerId.iloc[q].to_numpy(str),names,cache)


def production():
    raw,y,groups,_=data()
    test=pd.read_csv(ROOT/"data/raw/test.csv",usecols=raw.columns).loc[:,raw.columns]
    assert len(raw)==8693 and len(test)==4277
    test_groups=test.PassengerId.str.split("_").str[0].to_numpy()
    assert not np.isin(groups,test_groups).any()
    combined=pd.concat([raw,test],ignore_index=True)
    return combined,y,np.r_[groups,test_groups],np.arange(len(y)),np.arange(len(test))


def train_test():
    raw,y,groups,t,q=production()
    row=len(t)+q
    names,cache=fit_recipes(raw,y,groups,t,row,"competition_test",weights_directory=OUT/"weights")
    save(OUT/"test.npz",t,q,y,raw.PassengerId.iloc[row].to_numpy(str),names,cache)


def validate(path,t,q,y,passengers):
    with np.load(path) as z:
        for key,array in (("t",t),("q",q),("train_y",y[t]),("passenger",passengers)):
            np.testing.assert_array_equal(z[key],array)
        for mode in MODELS:
            p,a=z[mode],z[mode+"_importance"]
            assert p.shape==(len(q),) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            assert a.shape==(len(q),len(z["names"])) and (a>=0).all() and np.isfinite(a).all()
            totals=a.sum(axis=1)
            assert np.all((totals==0) | (np.abs(totals-1)<1e-5))
        return dict(sha256=sha(path),train=len(t),held=len(q),fields=z["names"].tolist(),
                    all_zero_attention={n:int((z[n+"_importance"].sum(axis=1)==0).sum()) for n in MODELS})


def reload_test():
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    torch.set_num_threads(2)
    raw,y,groups,t,q=production()
    x,names,_=inputs(raw,groups,t)
    with np.load(OUT/"test.npz") as cached:
        np.testing.assert_array_equal(names,cached["names"])
        for name in MODELS:
            restored=TabNetClassifier(device_name="cpu",verbose=0)
            restored.load_model(str(OUT/"weights"/(name+".zip")))
            p=restored.predict_proba(x[len(t)+q])[:,1]
            a,_=restored.explain(x[len(t)+q],normalize=False)
            a=a/np.maximum(a.sum(axis=1,keepdims=True),1e-9)
            np.testing.assert_allclose(p,cached[name],rtol=0.,atol=1e-6)
            np.testing.assert_allclose(a,cached[name+"_importance"],rtol=0.,atol=1e-6)
    print("Independently reloaded all three archives and matched all 4277 test predictions and attention matrices.")


def report():
    raw,y,groups,_=data()
    history={}
    all_queries=[]
    with np.load(PANEL) as panel:
        history_ids,historical_fold=panel["ids"],panel["fold"]
    collected={}
    expected_names=None
    for k in range(3):
        t,q,ordinary=historical_indices(raw,y,groups,k)
        path=OUT/"history"/f"fold{k}.npz"
        item=validate(path,t,q,y,raw.PassengerId.iloc[q].to_numpy(str))
        item.update(purged_people=int(len(ordinary)-len(t)),purged_orders=int(len(set(groups[ordinary])&set(groups[q]))),
                    training_orders=len(set(groups[t])),held_orders=len(set(groups[q])))
        with np.load(path) as z:
            item["accuracy"]={n:float(np.mean((z[n]>=.5)==y[q])) for n in MODELS}
            if expected_names is None:
                expected_names=z["names"]
            np.testing.assert_array_equal(z["names"],expected_names)
            np.testing.assert_array_equal(q,history_ids[historical_fold==k])
            for name in MODELS:
                for field in (name,name+"_importance"):
                    shape=(len(history_ids),*z[field].shape[1:])
                    collected.setdefault(field,np.full(shape,np.nan,dtype=z[field].dtype))[historical_fold==k]=z[field]
        history[str(k)]=item
        all_queries.extend(q)
    assert len(set(all_queries))==len(all_queries)==6954
    assert all(np.isfinite(v).all() for v in collected.values())
    combined=OUT/"history"/"combined.npz"
    temporary=combined.with_suffix(".building.npz")
    np.savez_compressed(temporary,ids=history_ids,fold=historical_fold,passenger=raw.PassengerId.iloc[history_ids].to_numpy(str),
                        names=expected_names,**collected)
    temporary.replace(combined)
    hist_truth = y[history_ids]
    history_accuracy = {n:float(np.mean((collected[n]>=.5)==hist_truth)) for n in MODELS}
    history_accuracy["three_model_mean"] = float(np.mean(
        (np.mean([collected[n] for n in MODELS],axis=0)>=.5)==hist_truth))
    raw,y,groups,t,q=production()
    submission=validate(OUT/"test.npz",t,q,y,raw.PassengerId.iloc[len(t)+q].to_numpy(str))
    submission.update(weights_sha256={n:sha(OUT/"weights"/(n+".zip")) for n in MODELS},
                      train_test_shared_orders=0, reload_tolerance=1e-6)
    summary=dict(package_version=version("pytorch-tabnet"),torch_version=version("torch"),
                 historical_panel_sha256=sha(PANEL),training_csv_sha256=sha(ROOT/"data/raw/train.csv"),
                 test_csv_sha256=sha(ROOT/"data/raw/test.csv"),history=history,test=submission,
                 historical_combined_sha256=sha(combined),historical_accuracy=history_accuracy,
                 architecture_script_sha256=sha(ROOT/"attentive_tabnet_n26.py"),
                 caching="Historical q is the train.csv row index. Test q is zero-based within test.csv. "
                         "Training t is always indexed within train.csv; train_y and held passenger IDs are retained. "
                         "MODELS entries contain P(Transported=True). Each _importance matrix is ordered by names.",
                 training="Same two scratch seeds and masked-pretraining recipe as attentive_tabnet_n26. Historical "
                          "support starts with the other historical folds only, then removes every order occurring "
                          "in the target fold. Booking sizes use publicly observed nonlabel roster fields. "
                          "Predicting the test roster uses all 8693 observed labels and no test labels.",
                 reload="Three upstream TabNetClassifier zip files in artifacts/nov26_tabnet/weights. For test "
                        "features call production() then inputs(raw,groups,t); infer on x[len(t)+q]. "
                        "train_test independently reloaded each classifier and matched probabilities and "
                        "normalized explain() matrices elementwise before publishing the receipt.")
    SUMMARY.write_text(json.dumps(summary,indent=2)+"\n")
    print({k:(v["train"],v["held"],v["accuracy"]) for k,v in history.items()},flush=True)
    print("test",submission["sha256"],submission["held"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    choice=parser.add_mutually_exclusive_group()
    choice.add_argument("--history",type=int,choices=range(3))
    choice.add_argument("--test",action="store_true")
    choice.add_argument("--reload",action="store_true")
    args=parser.parse_args()
    if args.reload:
        reload_test()
    elif args.test:
        train_test()
    elif args.history is not None:
        train_history(args.history)
    else:
        report()
