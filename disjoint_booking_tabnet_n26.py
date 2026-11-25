"""Repeat the seed-431 TabNet training with two sets of nonoverlapping booking labels."""
import argparse
import json
import os
import sys

import numpy as np

from attentive_tabnet_n26 import CATEGORIES,OUT as ORIGINAL,inputs
from full_roster_booking_validation_n23 import ROOT,data
from tabnet_history_test_n26 import PANEL,historical_indices,sha

OUT=ROOT/"artifacts/nov26_disjoint_tabnet"
REPORT=ROOT/"reports/nov23_matched_learning_curve/disjoint_booking_tabnet.json"
SEED=421901
MODE="scratch_seed_431"


def partition(t,g,k):
    groups=np.unique(g[t])
    order=np.random.default_rng(SEED+k).permutation(groups)
    half=np.isin(g[t],order[:len(order)//2])
    a,b=t[half],t[~half]
    assert abs(len(set(g[a]))-len(set(g[b])))<=1
    assert set(g[a]).isdisjoint(g[b])
    np.testing.assert_array_equal(np.sort(np.r_[a,b]),np.sort(t))
    return a,b


def fit(raw,y,groups,t,q):
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    torch.set_num_threads(2)
    x,_,levels=inputs(raw,groups,t)
    # Exact public seed-431 recipe, without inference-time attribution or other recipes.
    model=TabNetClassifier(n_d=16,n_a=16,n_steps=3,gamma=1.5,n_independent=2,n_shared=2,
                           cat_idxs=list(range(len(CATEGORIES))),cat_dims=levels,cat_emb_dim=3,mask_type="sparsemax",
                           optimizer_params={"lr":.012},scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
                           scheduler_params={"T_max":55,"eta_min":.0005},device_name="cpu",verbose=0,seed=431)
    model.fit(x[t],y[t].astype(int),max_epochs=55,compute_importance=False,batch_size=512,virtual_batch_size=128,
              num_workers=0,drop_last=False,pin_memory=False,patience=0)
    return model.predict_proba(x[q])[:,1]


def train(tag,k):
    raw,y,g,fold=data()
    if tag=="full":
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        source=ORIGINAL/f"fold{k}.npz"
    else:
        t,q,_=historical_indices(raw,y,g,k)
        source=ORIGINAL/"history"/f"fold{k}.npz"
    with np.load(source) as z:
        for name,expected in [("t",t),("q",q),("train_y",y[t]),("passenger",raw.PassengerId.iloc[q])]:
            np.testing.assert_array_equal(z[name],expected)
        full=z[MODE]
    a,b=partition(t,g,k)
    assert not np.isin(g[np.r_[a,b]],g[q]).any()
    p0,p1=fit(raw,y,g,a,q),fit(raw,y,g,b,q)
    assert np.isfinite(p0).all() and np.isfinite(p1).all()
    proof={}
    if tag=="full" and k==0:
        repeated=fit(raw,y,g,t,q)
        np.testing.assert_allclose(repeated,full,atol=1e-6,rtol=0)
        proof["max_control_difference"]=np.max(np.abs(repeated-full))
    path=OUT/tag/f"fold{k}.npz"
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(f".{path.stem}.{os.getpid()}.npz")
    np.savez_compressed(temp,t0=a,t1=b,q=q,passenger=raw.PassengerId.iloc[q].to_numpy(str),train_y0=y[a],train_y1=y[b],
                        p0=p0,p1=p1,p_full=full,source_sha=sha(source),seed=SEED+k,**proof)
    temp.replace(path)
    print(tag,k,len(a),len(b),len(q),float(((p0>=.5)==y[q]).mean()),float(((p1>=.5)==y[q]).mean()),flush=True)


def summarize(tag,raw,y,g,fold,known,versions):
    if tag=="full":
        ids=np.arange(len(y))
        f=fold
    else:
        with np.load(PANEL) as z:
            ids,f=z["ids"],z["fold"]
    probabilities=np.full((len(ids),3),np.nan)
    folds={}
    for k in np.unique(f):
        t,q=(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) if tag=="full" else historical_indices(raw,y,g,int(k))[:2]
        a,b=partition(t,g,int(k))
        path=OUT/tag/f"fold{k}.npz"
        source=ORIGINAL/(f"fold{k}.npz" if tag=="full" else f"history/fold{k}.npz")
        with np.load(path) as z, np.load(source) as reference:
            np.testing.assert_array_equal(q,ids[f==k])
            for key,expected in [("t0",a),("t1",b),("q",q),("train_y0",y[a]),("train_y1",y[b]),
                                 ("passenger",raw.PassengerId.iloc[q]),("p_full",reference[MODE])]:
                np.testing.assert_array_equal(z[key],expected)
            assert str(z["source_sha"])==sha(source)
            assert not np.isin(g[a],g[b]).any() and not np.isin(g[np.r_[a,b]],g[q]).any()
            probabilities[f==k]=np.column_stack([z[n] for n in ("p0","p1","p_full")])
            folds[str(k)]=dict(people0=len(a),people1=len(b),orders0=len(set(g[a])),orders1=len(set(g[b])),
                               held_people=len(q),held_orders=len(set(g[q])),source_sha256=str(z["source_sha"]),
                               prediction_sha256=sha(path))
            if tag=="full" and k==0:
                folds[str(k)]["rerun_full_support_max_abs_difference"]=float(z["max_control_difference"])
    assert np.isfinite(probabilities).all() and ((probabilities>=0)&(probabilities<=1)).all()
    path=OUT/tag/"combined.npz"
    np.savez_compressed(path,ids=ids,fold=f,passenger=raw.PassengerId.iloc[ids].to_numpy(str),
                        p0=probabilities[:,0],p1=probabilities[:,1],p_full=probabilities[:,2])
    targets=y[ids]
    votes=probabilities>=.5
    result=dict(folds=folds,rows=len(ids),combined_sha256=sha(path),accuracy={
        "half0":float(np.mean(votes[:,0]==targets)),"half1":float(np.mean(votes[:,1]==targets)),
        "original_full_support":float(np.mean(votes[:,2]==targets)),
        "mean_2_halves":float(np.mean((probabilities[:,:2].mean(axis=1)>=.5)==targets))})
    agree_halves=votes[:,0]==votes[:,1]
    result["halves_agree"]=dict(rows=int(agree_halves.sum()),accuracy=float(np.mean(votes[agree_halves,0]==targets[agree_halves])))
    # The frozen ten releases' archive uses the eight-fold whole-order protocol.
    # The historical subset evaluation is a matched-passenger comparison of two protocols.
    old=versions[ids,-1]
    scope=known[ids] & (versions[ids]==old[:,None]).all(axis=1)
    count=(votes[:,:2]!=old[:,None]).sum(axis=1)
    full_disagrees=votes[:,2]!=old
    if tag=="full":
        assert int((scope&(old!=targets)).sum())==1181
    actions={"both_vs_release":count==2,"one_vs_release":count==1,
             "full_support_vs_release":full_disagrees,"both_and_full_support":(count==2)&full_disagrees,
             "both_when_full_support_agrees":(count==2)&~full_disagrees,
             "one_when_full_support_agrees":(count==1)&~full_disagrees}
    prefix="ten_versions" if tag=="full" else "overlapping_passengers_vs_eight_fold_ten_versions"
    wrong=scope&(old!=targets); right=scope&~wrong
    tally={"common_wrong":int(wrong.sum()),"common_correct":int(right.sum()),
           "eligible":int(scope.sum()),"baseline_accuracy_on_common":float(right.sum()/scope.sum())}
    for label,mask in (("all_wrong_overlap",wrong),("all_correct_overlap",right)):
        dissent=votes[mask,:2]!=old[mask,None]
        marginals=dissent.sum(axis=0)
        intersection=int(dissent.all(axis=1).sum())
        union=int(dissent.any(axis=1).sum())
        tally[label]=dict(half0=int(marginals[0]),half1=int(marginals[1]),
                          intersection=intersection,union=union,intersection_over_union=intersection/max(1,union),
                          expected_intersection_if_independent=float(marginals.prod()/mask.sum()))
    for name,condition in actions.items():
        acted=condition&scope
        fixed=int((acted&wrong).sum()); harmed=int((acted&right).sum())
        tally[name]=dict(fixed=fixed,harmed=harmed,flips=fixed+harmed,net=fixed-harmed,
                         fraction_correct_among_changes=fixed/max(1,fixed+harmed),
                         accuracy_after=float((right.sum()+fixed-harmed)/scope.sum()),
                         net_by_fold=[int(np.sum(np.where(wrong,1,-1)*acted*(f==k))) for k in np.unique(f)])
    result[prefix]=tally
    return result


def report():
    from frozen_queue_shared_errors_n26 import paired
    raw,y,g,f,known,versions=paired()
    result={tag:summarize(tag,raw,y,g,f,known,versions) for tag in ("full","historical")}
    REPORT.write_text(json.dumps(dict(
        protocol="Each fold shuffles sorted available whole order IDs with numpy default_rng(421901+fold). "
                 "The first floor(n_orders/2) order IDs form half 0; remaining orders form half 1. "
                 "Train labels and order IDs are disjoint. Both fits use the same seed-431 initialization and "
                 "55 supervised epochs. Each half fits all preprocessing on its own support. Original full-support "
                 "scores come from the same fold's verified archive. No filtering or learned selection.",
        comparison="both means both models oppose the frozen ten-version decision; one means exactly one does. "
                   "Full is an eight-fold same-protocol comparison. Historical accuracy is independently group-purged; "
                   "the additional comparison uses the same passengers' eight-fold frozen releases.",
        reports=result),indent=2)+"\n")
    for tag,v in result.items():
        print(tag,v["accuracy"],v["halves_agree"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--panel",choices=["full","historical"])
    parser.add_argument("--fold",type=int)
    arg=parser.parse_args()
    if arg.panel is None and arg.fold is None:
        report()
    elif arg.panel and arg.fold in range(8 if arg.panel=="full" else 3):
        sys.path.insert(0,str(ORIGINAL/"package"))
        train(arg.panel,arg.fold)
    else:
        parser.error("Choose a panel and one of its folds.")
