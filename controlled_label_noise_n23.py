"""Benchmark peer-selected training cases with genuine labels and known injected label changes."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS
from class_analogue_experts_n23 import design
from exaone_validation_data import compact_features
from relational_error_corrector import write


def selected(loss,y,drop):
    keep=torch.ones(len(y),dtype=torch.bool)
    for label in (0,1):
        ids=torch.where(y==label)[0]
        n=round(drop*len(ids))
        if n:
            keep[ids[torch.topk(loss[ids].detach(),n).indices]]=False
    return keep


def fit(x,target,t,q,flipped,mode):
    torch.manual_seed(318)
    members=[nn.Sequential(nn.Linear(x.shape[1],96),nn.SiLU(),nn.Dropout(.12),
                           nn.Linear(96,48),nn.SiLU(),nn.Linear(48,1)) for _ in range(2)]
    optimizers=[torch.optim.AdamW(model.parameters(),lr=.0024,weight_decay=.01) for model in members]
    rng=np.random.default_rng(877)
    audited=np.zeros(len(target)); removed=np.zeros(len(target))
    for epoch in range(24):
        for batch in np.array_split(rng.permutation(t),16):
            losses=[F.binary_cross_entropy_with_logits(m(x[batch]).flatten(),target[batch],reduction="none")
                    for m in members]
            if mode=="peer":
                masks=[selected(v,target[batch],.1*min(1,epoch/6)) for v in losses[::-1]]
            elif mode=="oracle":
                masks=[torch.from_numpy(~flipped[batch])]*2
            else:
                masks=[torch.ones(len(batch),dtype=torch.bool)]*2
            for optimizer,loss,mask in zip(optimizers,losses,masks):
                optimizer.zero_grad()
                loss[mask].mean().backward()
                optimizer.step()
                if epoch>=6:
                    audited[batch]+=1
                    removed[batch]+=~mask.numpy()
    with torch.no_grad():
        probs=[torch.sigmoid(m.eval()(x[q]).flatten()).numpy() for m in members]
    return np.mean(probs,axis=0),removed[t]/audited[t]


def run():
    data={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            data[seed]={k:z[k] for k in ("row","y","base","fold")}
    first=data[SEEDS[0]]
    row,y=first["row"],first["y"]
    full=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"})
    gr=full.PassengerId.str.split("_").str[0]
    group=gr.to_numpy()[row]
    size=gr.map(gr.value_counts()).to_numpy()[row]
    raw=full.iloc[row]
    earth=raw.HomePlanet.eq("Earth").to_numpy()
    frame=compact_features(raw)
    repeat=np.logical_and.reduce([(v["base"]>=.5)!=y for v in data.values()])
    out=ART/"nov23_controlled_label_noise"
    out.mkdir(exist_ok=True)
    results={}
    for seed,v in data.items():
        np.testing.assert_array_equal(row,v["row"]);np.testing.assert_array_equal(y,v["y"])
        fold,base=v["fold"],v["base"]
        pred={};diagnostics=[]
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(group[t],group[q]).size
            _,features=design(frame,t)
            x=torch.tensor(np.column_stack([features,np.log1p(size)]),dtype=torch.float32)
            injected=np.zeros(len(y),bool)
            rng=np.random.default_rng(seed+int(f))
            for label in (False,True):
                s=t[y[t]==label]
                injected[rng.choice(s,round(.10*len(s)),replace=False)]=True
            assert not injected[q].any()
            for source,changes in [("original",np.zeros(len(y),bool)),("injected",injected)]:
                target=torch.tensor(y^changes,dtype=torch.float32)
                modes=("all","peer","oracle") if source=="injected" else ("all","peer")
                for mode in modes:
                    p,omitted=fit(x,target,t,q,changes,mode)
                    name=source+"_"+mode
                    pred.setdefault(name,np.full(len(y),np.nan))[q]=p
                    if mode=="peer":
                        diag=dict(source=source,fold=int(f),injected=int(changes.sum()),discard_rate=float(omitted.mean()))
                        if source=="injected":
                            diag["known_flip_share_of_discards"]=float(omitted[changes[t]].sum()/omitted.sum())
                            diag["discard_rate_for_known_flips"]=float(omitted[changes[t]].mean())
                        diagnostics.append(diag)
        for name in ("original_all","original_peer"):
            pred[name+"_quarter"]=.75*base+.25*pred[name]
        with np.load(ART/f"nov23_identical_bills/{seed}.npz") as old:
            np.testing.assert_array_equal(row,old["row"])
            hard=repeat&earth&old["eligible"]
        before=(base>=.5)==y
        scores={}
        for name,p in pred.items():
            assert np.isfinite(p).all()
            ok=(p>=.5)==y
            scores[name]=dict(accuracy=float(ok.mean()),fixed=int((ok&~before).sum()),harmed=int((before&~ok).sum()),
                              repeat=int((repeat&ok).sum()),earth430=int((hard&ok).sum()))
        results[seed]=dict(scores=scores,discard=diagnostics)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**pred)
        write(ROOT/"reports/nov23_controlled_label_noise/result.json",results)
        print(seed,scores["original_peer"],flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
