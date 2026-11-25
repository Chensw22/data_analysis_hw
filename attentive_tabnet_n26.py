"""Train TabNet from random weights on whole-booking folds and audit its common-error coverage.

Install the upstream CPU-compatible package alongside the project's existing Torch runtime:
    uv pip install --python .venv-tabpfn/bin/python --no-deps --target artifacts/nov26_tabnet/package pytorch-tabnet==4.1.0
Run --fold 0 through --fold 7 with that environment; run without --fold in the main project
environment to compare the cached predictions to the existing release panel.
"""
import argparse
from importlib.metadata import version
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from full_roster_booking_validation_n23 import ROOT, data

OUT = ROOT / "artifacts/nov26_tabnet"
REPORT = ROOT / "reports/nov23_matched_learning_curve/attentive_tabnet.json"
BILLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORIES = ["HomePlanet", "Destination", "CryoSleep", "VIP", "deck", "side"]
MODELS = ["scratch_seed_431", "scratch_seed_731", "masked_pretraining"]


def inputs(raw, groups, t):
    frame = raw.drop(columns=["PassengerId", "Cabin"]).copy()
    frame[["deck", "berth", "side"]] = raw.Cabin.str.split("/", expand=True)
    frame.berth = pd.to_numeric(frame.berth)
    frame["order_size"] = raw.groupby(groups).PassengerId.transform("size")
    names, columns, cardinality = [], [], []
    for field in CATEGORIES:
        values = frame[field].astype("string").fillna("__missing__")
        levels = sorted(values.iloc[t].unique())
        ids = pd.Categorical(values, categories=levels).codes + 1
        names.append(field)
        columns.append(ids)
        cardinality.append(len(levels) + 1)
    numerical = frame[["Age", "berth", "order_size", *BILLS]].copy()
    numerical["bill_total"] = numerical[BILLS].sum(axis=1, min_count=1)
    numerical["paid_services"] = numerical[BILLS].gt(0).sum(axis=1)
    numerical[[*BILLS, "bill_total"]] = np.log1p(numerical[[*BILLS,"bill_total"]])
    for field in numerical:
        arr = numerical[field].to_numpy(float)
        center = np.nanmedian(arr[t])
        scale = np.nanstd(arr[t])
        if not np.isfinite(scale) or scale<1e-8:
            scale = 1.
        columns.append(np.nan_to_num((arr-center)/scale).clip(-8,8))
        names.append(field)
        if np.isnan(arr[t]).any():
            columns.append(np.isnan(arr).astype(float))
            names.append(field+"_missing")
    x = np.column_stack(columns).astype(np.float32)
    assert np.isfinite(x).all()
    return x, names, cardinality


def train_fold(k):
    import torch
    from pytorch_tabnet.pretraining import TabNetPretrainer
    from pytorch_tabnet.tab_model import TabNetClassifier
    assert version("pytorch-tabnet")=="4.1.0"
    torch.set_num_threads(2)
    raw,y,groups,fold = data()
    t,q = np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
    assert not np.isin(groups[t],groups[q]).any()
    x,names,cardinality = inputs(raw,groups,t)
    constructor = dict(n_d=16,n_a=16,n_steps=3,gamma=1.5,n_independent=2,n_shared=2,
                       cat_idxs=list(range(len(CATEGORIES))),cat_dims=cardinality,cat_emb_dim=3,
                       mask_type="sparsemax",optimizer_params={"lr":.012},
                       scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
                       device_name="cpu",verbose=0)
    fit = dict(batch_size=512,virtual_batch_size=128,num_workers=0,drop_last=False,pin_memory=False,patience=0)
    archive = {}
    for mode in MODELS:
        seed = 731 if mode=="scratch_seed_731" else 431
        unsupervised = None
        if mode=="masked_pretraining":
            unsupervised = TabNetPretrainer(**constructor,seed=seed,scheduler_params={"T_max":18,"eta_min":.001})
            unsupervised.fit(x[t],max_epochs=18,pretraining_ratio=.35,**fit)
        model = TabNetClassifier(**constructor,seed=seed,scheduler_params={"T_max":55,"eta_min":.0005})
        model.fit(x[t],y[t].astype(int),max_epochs=55,from_unsupervised=unsupervised,compute_importance=False,**fit)
        p = model.predict_proba(x[q])[:,1]
        contribution,_ = model.explain(x[q],normalize=False)
        attribution = contribution/np.maximum(contribution.sum(axis=1,keepdims=True),1e-9)
        assert np.isfinite(p).all() and np.isfinite(attribution).all()
        archive[mode] = p
        archive[mode+"_importance"] = np.asarray(attribution,dtype=np.float32)
        archive[mode+"_loss"] = np.asarray(model.history["loss"])
        print(f"fold={k} recipe={mode} accuracy={np.mean((p>=.5)==y[q]):.4f}",flush=True)
    OUT.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(OUT/f"fold{k}.npz",t=t,q=q,train_y=y[t],passenger=raw.PassengerId.iloc[q].to_numpy(str),
                        names=names,**archive)


def report():
    from frozen_queue_shared_errors_n26 import paired
    raw,y,groups,fold,known,versions = paired()
    base = versions[:,-1]
    common = known & (versions==base[:,None]).all(axis=1)
    wrong,correct = common&(base!=y),common&(base==y)
    assert wrong.sum()==1181
    sleeper = raw.CryoSleep.eq(True).to_numpy() & ~raw[BILLS].gt(0).any(axis=1).to_numpy() & base
    profiles = {"queue_unanimous":common, "sleeper_false_alarm_candidate":common&sleeper,
                "other_unanimous":common&~sleeper,"solo_unanimous":common&(raw.groupby(groups).PassengerId.transform("size").to_numpy()==1)}
    scores={n:np.full(len(y),np.nan) for n in MODELS}
    previous_agree = np.zeros(len(y),dtype=bool)
    explanations={n:[] for n in MODELS}
    train_loss={}
    for k in range(8):
        with np.load(OUT/f"fold{k}.npz") as z:
            t,q=z["t"],z["q"]
            np.testing.assert_array_equal(t,np.flatnonzero(fold!=k))
            np.testing.assert_array_equal(q,np.flatnonzero(fold==k))
            np.testing.assert_array_equal(z["train_y"],y[t])
            np.testing.assert_array_equal(z["passenger"],raw.PassengerId.iloc[q])
            assert not np.isin(groups[t],groups[q]).any()
            for name in MODELS:
                scores[name][q] = z[name]
                explanations[name].append(pd.DataFrame(z[name+"_importance"],columns=z["names"],index=q))
            train_loss[str(k)]={name:[float(z[name+"_loss"][0]),float(z[name+"_loss"][-1])] for name in MODELS}
        with np.load(ROOT/f"artifacts/nov26_original_tabstar/tuned{k}.npz") as stellar:
            np.testing.assert_array_equal(q,stellar["q"])
            np.testing.assert_array_equal(y[t],stellar["train_y"])
            previous_agree[q]=(stellar["finetuned"]>=.5)==base[q]
        with np.load(ROOT/f"artifacts/nov26_ordinal_factor_density/fold{k}.npz") as density:
            np.testing.assert_array_equal(q,density["q"])
            np.testing.assert_array_equal(y[t],density["train_y"])
            previous_agree[q]&=(density["conditional_bills"]>=.5)==base[q]
    profiles["queue_and_tabstar_and_density"]=common & previous_agree
    scores["scratch_mean"] = (scores[MODELS[0]] + scores[MODELS[1]])/2
    scores["three_model_mean"] = np.mean([scores[name] for name in MODELS],axis=0)
    result={}
    for name,p in scores.items():
        assert np.isfinite(p).all()
        predicts=p>=.5
        support={}
        for label,scope in profiles.items():
            was_wrong=scope&wrong
            was_right=scope&correct
            support[label]=dict(errors=int(was_wrong.sum()),fixed=int((was_wrong&(predicts==y)).sum()),
                                correct=int(was_right.sum()),harmed=int((was_right&(predicts!=y)).sum()))
        info=dict(accuracy=accuracy_score(y,predicts),log_loss=log_loss(y,p),support=support,
                  fold_accuracy=[accuracy_score(y[fold==k],predicts[fold==k]) for k in range(8)])
        if name in explanations:
            weights=pd.concat(explanations[name]).sort_index()
            np.testing.assert_array_equal(weights.index,np.arange(len(y)))
            actions={"fixed_unanimous":wrong&(predicts==y),"harmed_unanimous":correct&(predicts!=y)}
            info["mean_attention"]={key:weights.loc[m].mean().sort_values(ascending=False).head(5).to_dict()
                                    for key,m in actions.items()}
        result[name]=info
    same_votes=np.column_stack([scores[n]>=.5 for n in MODELS])
    agree=(same_votes==same_votes[:,:1]).all(axis=1)
    strength=np.minimum.reduce([np.maximum(scores[n],1-scores[n]) for n in MODELS])
    changed=same_votes[:,0]!=base
    committees={}
    for level in (.5,.7,.85,.95):
        act=agree&changed&(strength>=level)
        committees[str(level)]=dict(fixed=int((act&wrong).sum()),harmed=int((act&correct).sum()))
    REPORT.write_text(json.dumps(dict(
        package="pytorch-tabnet==4.1.0 (DreamQuark PyTorch implementation, MIT)",
        implementation="https://github.com/dreamquark-ai/tabnet",paper="https://arxiv.org/abs/1908.07442",
        release="https://pypi.org/project/pytorch-tabnet/4.1.0/",
        package_python_source_fingerprint="0b7e094a8a63a7cc83e87d040377d97b20e14f7dfa69db47f7b1ad92e599b6de",
        protocol="8 GroupKFold splits, no external weights or data. Encoders, centering, reconstruction pretraining and supervised fitting use only nonheld bookings. "
                 "55 fixed supervised epochs, n_d=n_a=16, three sparsemax decision steps; second seed is a reproducibility control. "
                 "The third recipe adds 18 epochs of masked self-supervised reconstruction on the same training passengers. "
                 "No name or passenger ID inputs; known booking size, cabin parts and expense summaries included.",
        results=result,unanimous_three_seeds_or_recipes=committees,train_loss_first_last=train_loss),indent=2)+"\n")
    for name,info in result.items():
        print(name,info["accuracy"],info["support"]["queue_unanimous"],flush=True)
    print(committees)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--fold",type=int,choices=range(8))
    parsed=parser.parse_args()
    if parsed.fold is None:
        report()
    else:
        sys.path.insert(0,str(OUT/"package"))
        train_fold(parsed.fold)
