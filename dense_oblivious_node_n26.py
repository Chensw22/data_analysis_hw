"""Use the authors' MIT DenseBlock and ODST with no external weights.

Use --fetch to obtain the four upstream modules and license, then train --fold 0 to 7
with the existing Torch environment. Run without arguments in the main environment
to validate row alignment and compare the predictions with frozen releases.
"""
import argparse
from hashlib import sha256
import importlib
import json
import sys
import time
from types import ModuleType
from urllib.request import urlopen

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import QuantileTransformer

from attentive_tabnet_n26 import CATEGORIES,MODELS,OUT as TABNET,inputs
from full_roster_booking_validation_n23 import ROOT,data,OUT as ALIGNED

OUT=ROOT/"artifacts/nov26_node"
REVISION="3bae6a8a63f0205683270b6d566d9cfa659403e4"
UPSTREAM=OUT/"upstream"/f"node-{REVISION}"
REPORT=ROOT/"reports/nov23_matched_learning_curve/dense_oblivious_node.json"
CHECKSUMS={"lib/arch.py":"02f9d08cfadc96b3d77c04dbe6720be7a5e165892949944c366439c018e50541",
           "lib/odst.py":"878fa48cbf3c319dc9909400efe0ed601dae180754ae30997aaf14c94e59b81b",
           "lib/nn_utils.py":"91fcac1db376552fef6ec9555cbfefbe468036d8a7d1e1d836d8d5aca8d52a4a",
           "lib/utils.py":"fac08661980a8097f8e4e7e6fd3745be19bd91db17def866b6023c9d6127354c",
           "LICENSE.md":"60b6519a046458e6901cf233eb0d403ae56f499aa6fc008e95fceb75eea4edd4"}


def check_sources(fetch=False):
    for rel,digest in CHECKSUMS.items():
        p=UPSTREAM/rel
        if fetch and not p.is_file():
            url=f"https://raw.githubusercontent.com/Qwicen/node/{REVISION}/{rel}"
            with urlopen(url,timeout=90) as response:
                contents=response.read()
            if sha256(contents).hexdigest()!=digest:
                raise ValueError(f"Unexpected remote source for {rel}")
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(contents)
        assert sha256(p.read_bytes()).hexdigest()==digest


def original_modules():
    check_sources()
    # The repository package initializer also imports dataset downloaders and an unused
    # third-party optimizer. Import its untouched model modules under their own namespace.
    package=ModuleType("node_upstream")
    package.__path__=[str(UPSTREAM/"lib")]
    sys.modules[package.__name__]=package
    return importlib.import_module("node_upstream.arch"),importlib.import_module("node_upstream.nn_utils")


def design(raw,groups,t):
    x,_,card=inputs(raw,groups,t)
    features=[np.eye(width,dtype=np.float32)[x[:,i].astype(int)] for i,width in enumerate(card)]
    for j in range(len(CATEGORIES),x.shape[1]):
        v=x[:,j:j+1]
        if np.unique(v[t]).size>16:
            transform=QuantileTransformer(n_quantiles=128,output_distribution="normal",random_state=902)
            v=transform.fit(v[t]).transform(v).clip(-3,3)
        features.append(v)
    matrix=np.column_stack(features).astype(np.float32)
    assert np.isfinite(matrix).all()
    return matrix


def fit_fold(k,historical=False):
    import torch
    from threadpoolctl import threadpool_limits
    arch,functions=original_modules()
    raw,y,g,fold=data()
    if historical:
        from tabnet_history_test_n26 import historical_indices
        t,q,_=historical_indices(raw,y,g,k)
    else:
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
    assert not np.isin(g[t],g[q]).any()
    x=design(raw,g,t)
    torch.set_num_threads(2)
    np.random.seed(2911)
    torch.manual_seed(2911)
    network=arch.DenseBlock(input_dim=x.shape[1],layer_dim=32,num_layers=2,tree_dim=2,depth=4,
                            choice_function=functions.entmax15,bin_function=functions.entmoid15,
                            flatten_output=False)
    train=torch.from_numpy(x[t])
    query=torch.from_numpy(x[q])
    labels=torch.from_numpy(y[t].astype(np.int64))
    with torch.no_grad():
        network(train[torch.randperm(len(t))[:1200]])
    opt=torch.optim.AdamW(network.parameters(),lr=.008,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=.0007)
    losses=[]
    started=time.monotonic()
    for epoch in range(50):
        total=0.
        for ix in torch.randperm(len(t)).split(256):
            logits=network(train[ix]).mean(dim=1)
            loss=torch.nn.functional.cross_entropy(logits,labels[ix])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),5.)
            opt.step()
            total+=loss.item()*len(ix)
        schedule.step()
        losses.append(total/len(t))
        if epoch in (0,24,49):
            print(f"fold={k} epoch={epoch+1} training_loss={losses[-1]:.4f}",flush=True)
    network.eval()
    with torch.inference_mode():
        preds=[network(batch).mean(dim=1).softmax(dim=1)[:,1].numpy() for batch in query.split(256)]
    p=np.concatenate(preds)
    with threadpool_limits(2):
        tree=HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=23,min_samples_leaf=25,
                    learning_rate=.055,l2_regularization=1.5,early_stopping=False,random_state=2911).fit(x[t],y[t])
        p_tree=tree.predict_proba(x[q])[:,1]
    assert np.isfinite(p).all() and np.isfinite(p_tree).all()
    destination=OUT/"history" if historical else OUT
    destination.mkdir(exist_ok=True,parents=True)
    additional={}
    if historical:
        import joblib
        changed=raw.copy()
        changed.loc[q,"Age"]=changed.loc[q,"Age"]+1
        alternative=design(changed,g,t)
        np.testing.assert_array_equal(alternative[t],x[t])
        seen=raw.Age.iloc[q].notna().to_numpy()
        np.testing.assert_array_equal(alternative[q][~seen],x[q][~seen])
        with torch.inference_mode():
            changed_p=np.concatenate([network(batch).mean(dim=1).softmax(dim=1)[:,1].numpy()
                                      for batch in torch.from_numpy(alternative[q]).split(256)])
        additional["node_age_plus1_delta"]=changed_p-p
        with threadpool_limits(2):
            additional["ordinary_tree_age_plus1_delta"]=tree.predict_proba(alternative[q])[:,1]-p_tree
        additional["age_observed"]=seen
        for name in ("node_age_plus1_delta","ordinary_tree_age_plus1_delta"):
            assert np.isfinite(additional[name]).all()
            np.testing.assert_allclose(additional[name][~seen],0,atol=1e-6)
        weights=destination/"weights"
        weights.mkdir(exist_ok=True)
        torch.save(network.state_dict(),weights/f"node{k}.pt")
        joblib.dump(tree,weights/f"hgb{k}.joblib",compress=3)
    np.savez_compressed(destination/f"fold{k}.npz",t=t,q=q,train_y=y[t],passenger=raw.PassengerId.iloc[q].to_numpy(str),
                        p=p,ordinary_tree=p_tree,loss=losses,revision=REVISION,**additional)
    print(f"fold={k} held={len(q)} accuracy={np.mean((p>=.5)==y[q]):.4f} seconds={time.monotonic()-started:.1f}",flush=True)


def history_report():
    from tabnet_history_test_n26 import PANEL,historical_indices,sha
    raw,y,g,_=data()
    with np.load(PANEL) as z:
        ids,fold=z["ids"],z["fold"]
    location=OUT/"history"
    fields={"NODE":"p","ordinary_tree":"ordinary_tree",
            "node_age_plus1_delta":"node_age_plus1_delta","ordinary_tree_age_plus1_delta":"ordinary_tree_age_plus1_delta"}
    combined={name:np.full(len(ids),np.nan) for name in fields}
    observed=np.zeros(len(ids),dtype=bool)
    metrics={}
    for k in range(3):
        t,q,ordinary=historical_indices(raw,y,g,k)
        path=location/f"fold{k}.npz"
        mask=fold==k
        np.testing.assert_array_equal(ids[mask],q)
        assert not np.isin(g[t],g[q]).any()
        with np.load(path) as z:
            for name,expected in (("t",t),("q",q),("train_y",y[t]),("passenger",raw.PassengerId.iloc[q])):
                np.testing.assert_array_equal(z[name],expected)
            assert str(z["revision"])==REVISION
            observed[mask]=z["age_observed"]
            np.testing.assert_array_equal(observed[mask],raw.Age.iloc[q].notna())
            for key,source in fields.items():
                combined[key][mask]=z[source]
        metrics[str(k)]=dict(train=len(t),held=len(q),removed_same_order=len(ordinary)-len(t),
                             sha256=sha(path),weights_node_sha256=sha(location/"weights"/f"node{k}.pt"),
                             weights_hgb_sha256=sha(location/"weights"/f"hgb{k}.joblib"))
    assert np.unique(ids).size==len(ids)==6954
    for key,value in combined.items():
        assert np.isfinite(value).all()
        if key.endswith("_delta"):
            np.testing.assert_allclose(value[~observed],0,atol=1e-6)
        else:
            assert ((value>=0)&(value<=1)).all()
    combined_path=location/"combined.npz"
    np.savez_compressed(combined_path,ids=ids,fold=fold,passenger=raw.PassengerId.iloc[ids].to_numpy(str),
                        age_observed=observed,**combined)
    aggregate=dict(held=len(ids),folds=metrics,combined_sha256=sha(combined_path),source_panel_sha256=sha(PANEL),
                   accuracy={n:float(np.mean((combined[n]>=.5)==y[ids])) for n in ("NODE","ordinary_tree")},
                   log_loss={n:log_loss(y[ids],combined[n]) for n in ("NODE","ordinary_tree")},
                   age_observed=int(observed.sum()),mean_abs_age_delta={key:float(np.mean(np.abs(val[observed])))
                                  for key,val in combined.items() if key.endswith("_delta")},
                   finite_difference="Each held passenger with observed Age is made one year older, "
                      "then predicted with the same fitted network or HGB and training-only preprocessing. "
                      "Each stored delta is perturbed P(True) minus observed P(True). Missing ages stay missing.",
                   eight_fold_difference="Unavailable: the original eight folds retained predictions, not fitted weights.",
                   training="Each model uses the same eight-fold recipe and upstream commit, trained on the other "
                      "historical panel folds after removing every booking that occurs in the query fold.")
    (REPORT.parent/"dense_oblivious_node_history.json").write_text(json.dumps(aggregate,indent=2)+"\n")
    print(aggregate["accuracy"],aggregate["combined_sha256"],flush=True)


def report():
    from distinct_challenger_error_reach_n26 import population
    raw,y,g,fold,base,same,count,_,_,wrong,_,_=population()
    prob={name:np.full(len(y),np.nan) for name in ("NODE","ordinary_tree","TabNet_mean")}
    training={}
    for k in range(8):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        for path in (OUT/f"fold{k}.npz",TABNET/f"fold{k}.npz"):
            with np.load(path) as z:
                for key,expected in [("t",t),("q",q),("train_y",y[t]),("passenger",raw.PassengerId.iloc[q])]:
                    np.testing.assert_array_equal(z[key],expected)
                assert not np.isin(g[t],g[q]).any()
                if path.parent==OUT:
                    assert str(z["revision"])==REVISION
                    prob["NODE"][q],prob["ordinary_tree"][q]=z["p"],z["ordinary_tree"]
                    training[str(k)]=z["loss"].tolist()
                else:
                    prob["TabNet_mean"][q]=np.mean([z[n] for n in MODELS],axis=0)
    with np.load(ALIGNED/"cat.npz") as z:
        for key,expected in [("passenger",raw.PassengerId),("fold",fold),("y",y)]:
            np.testing.assert_array_equal(z[key],expected)
        prob["archived_catboost"]=z["p"]
    result={}
    assert (same&wrong).sum()==1181 and (same&wrong&(count==0)).sum()==172
    for name,p in prob.items():
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        counter=(p>=.5)!=base
        scopes={"all_ten_common":same,"existing_countervote":same&(count>0),
                "no_prior_countervote":same&(count==0),
                "ordinary_tree_and_tabnet_agree_with_release":same&((prob["ordinary_tree"]>=.5)==base)&((prob["TabNet_mean"]>=.5)==base)}
        result[name]={"accuracy":float(((p>=.5)==y).mean()),"log_loss":log_loss(y,p),
                      "per_fold":[float(((p[fold==k]>=.5)==y[fold==k]).mean()) for k in range(8)],"scopes":{}}
        for tag,scope in scopes.items():
            errors=scope&wrong
            correct=scope&~wrong
            result[name]["scopes"][tag]=dict(rows=int(scope.sum()),errors=int(errors.sum()),correct=int(correct.sum()),
                                              fixed=int((errors&counter).sum()),harmed=int((correct&counter).sum()))
    action=(prob["NODE"]>=.5)!=base
    support={}
    for label,mask in {"unfiltered":same,"NODE_confidence_at_least_80":same&(np.maximum(prob["NODE"],1-prob["NODE"])>=.8),
                       "NODE_confidence_at_least_95":same&(np.maximum(prob["NODE"],1-prob["NODE"])>=.95),
                       "also_opposed_by_both_tree_and_tabnet":same&((prob["ordinary_tree"]>=.5)!=base)&((prob["TabNet_mean"]>=.5)!=base)}.items():
        proposed=mask&action
        support[label]=dict(fixed=int((proposed&wrong).sum()),harmed=int((proposed&~wrong).sum()),
            net_by_fold=[int(np.sum(proposed*(fold==k)*np.where(wrong,1,-1))) for k in range(8)])
    newly_reached=[]
    for i in np.flatnonzero(same & (count==0) & wrong & action):
        newly_reached.append(dict(row=int(i),PassengerId=str(raw.PassengerId.iloc[i]),fold=int(fold[i]),
                                  truth=bool(y[i]),old=bool(base[i]),p_node=float(prob["NODE"][i]),
                                  p_ordinary_tree=float(prob["ordinary_tree"][i]),p_tabnet=float(prob["TabNet_mean"][i])))
    (OUT/"newly_reached_old_errors.json").write_text(json.dumps(newly_reached,indent=2)+"\n")
    examples=[{k:item[k] for k in ("fold","p_node","p_ordinary_tree","p_tabnet")} for item in newly_reached]
    np.savez_compressed(OUT/"combined.npz",ids=np.arange(len(y)),passenger=raw.PassengerId.to_numpy(str),fold=fold,**prob)
    REPORT.write_text(json.dumps(dict(
        paper="https://arxiv.org/abs/1909.06312",upstream=f"https://github.com/Qwicen/node/tree/{REVISION}",
        source_checksums=CHECKSUMS,protocol="Eight full-booking holdouts on train.csv; no test data, labels or pretrained weights. "
        "The untouched author DenseBlock stacks two densely connected layers of 32 ODST, depth 4, 2 response dimensions, "
        "entmax15 feature selectors and entmoid15 routing. Trained locally with a 50-epoch AdamW schedule; this compact CPU "
        "training recipe differs from the paper benchmarks. Raw unnamed fields, known order size and spend summaries; "
        "one-hot categories and training-only marginal quantiles. The ordinary asymmetric histogram-boosted tree uses "
        "the same features. Archive CatBoost has its own existing feature policy. The six existing challenger blocks "
        "include the three TabNet recipes. Prior-countervote counts use distinct_challenger_error_reach_n26.population.",
        train_loss=training,results=result,node_countervotes=support,
        newly_reached_examples=examples),indent=2)+"\n")
    print({n:(v["accuracy"],v["scopes"]) for n,v in result.items()},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--fold",type=int,choices=range(8))
    parser.add_argument("--historical",action="store_true")
    parser.add_argument("--fetch",action="store_true")
    args=parser.parse_args()
    if args.fetch:
        check_sources(True)
    elif args.historical and args.fold not in (0,1,2,None):
        parser.error("The historical panel has three folds.")
    elif args.historical and args.fold is None:
        history_report()
    elif args.fold is None:
        report()
    else:
        fit_fold(args.fold,historical=args.historical)
