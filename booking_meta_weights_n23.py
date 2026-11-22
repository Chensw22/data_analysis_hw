"""Learn which labelled examples help classifiers on other bookings during training."""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS
from class_analogue_experts_n23 import design
from exaone_validation_data import compact_features
from relational_error_corrector import write


def influence_weights(model, x, labels, validation, losses):
    epsilon=torch.zeros_like(losses,requires_grad=True)
    params=dict(model.named_parameters())
    grads=torch.autograd.grad((epsilon*losses).sum(),tuple(params.values()),create_graph=True)
    trial={k:p-.02*g for (k,p),g in zip(params.items(),grads)}
    logits=functional_call(model,trial,(x[validation],)).flatten()
    signs=2*labels[validation]-1
    # Smooth classification error targets changes of decision on separate bookings.
    objective=torch.sigmoid(-signs*logits/.55).mean()+.15*F.softplus(-signs*logits).mean()
    grad,=torch.autograd.grad(objective,epsilon)
    helpful=(-grad.detach()).clamp_min(0)
    average=helpful.mean()
    return torch.where(average>0, .33+.67*helpful/average.clamp_min(1e-12), torch.ones_like(helpful))


def train(x,y,fit,meta,query,mode):
    torch.manual_seed(371)
    model=nn.Sequential(nn.Linear(x.shape[1],80),nn.SiLU(),nn.Linear(80,40),nn.SiLU(),nn.Linear(40,1))
    opt=torch.optim.AdamW(model.parameters(),lr=.0023,weight_decay=.006)
    rng=np.random.default_rng(1051)
    weights_by_row=np.zeros(len(y))
    seen=np.zeros(len(y))
    for _ in range(24):
        for batch in np.array_split(rng.permutation(fit),12):
            validation=rng.choice(meta,128,replace=False)
            perm=rng.permutation(len(batch))
            logits=model(x[batch]).flatten()
            loss=F.binary_cross_entropy_with_logits(logits,y[batch],reduction="none")
            if mode=="uniform":
                w=torch.ones_like(loss)
            else:
                w=influence_weights(model,x,y,validation,loss)
                if mode=="shuffled":
                    w=w[perm]
            weights_by_row[batch]+=w.detach().numpy()
            seen[batch]+=1
            opt.zero_grad()
            # Re-evaluate the ordinary parameter gradient after the virtual update.
            current=F.binary_cross_entropy_with_logits(model(x[batch]).flatten(),y[batch],reduction="none")
            (current*w.detach()).mean().backward()
            opt.step()
    with torch.no_grad():
        return torch.sigmoid(model(x[query]).flatten()).numpy(),weights_by_row[fit]/seen[fit]


def run():
    data={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            data[seed]={k:z[k] for k in ("row","y","base","fold")}
    first=data[SEEDS[0]]
    row,y=first["row"],first["y"]
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"})
    group=raw.PassengerId.str.split("_").str[0]
    size=group.map(group.value_counts()).to_numpy()[row]
    group=group.to_numpy()[row]
    frame=compact_features(raw.iloc[row]).reset_index(drop=True)
    earth=raw.HomePlanet.eq("Earth").to_numpy()[row]
    repeat=np.logical_and.reduce([(v["base"]>=.5)!=y for v in data.values()])
    labels=torch.tensor(y,dtype=torch.float32)
    out=ART/"nov23_booking_meta_weights"
    out.mkdir(exist_ok=True)
    results={}
    for seed,v in data.items():
        np.testing.assert_array_equal(row,v["row"])
        np.testing.assert_array_equal(y,v["y"])
        base,fold=v["base"],v["fold"]
        pred={k:np.full(len(y),np.nan) for k in ("uniform","shuffled","aligned")}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(group[t],group[q]).size
            sub={k:[] for k in pred}
            row_weights={}
            for a,b in GroupKFold(2,shuffle=True,random_state=731).split(t,groups=group[t]):
                fit,meta=t[a],t[b]
                assert not np.intersect1d(group[fit],group[meta]).size
                _,phi=design(frame,fit)
                x=torch.tensor(np.column_stack([phi,np.log1p(size)]),dtype=torch.float32)
                for k in pred:
                    p,w=train(x,labels,fit,meta,q,k)
                    sub[k].append(p)
                    row_weights.setdefault(k,np.zeros(len(t)))[a]=w
            for k in pred:
                pred[k][q]=np.mean(sub[k],axis=0)
            np.savez_compressed(out/f"{seed}_{f}_weights.npz",train=row[t],**row_weights)
        for k,p in list(pred.items()):
            assert np.isfinite(p).all()
            pred[k+"_quarter"]=.75*base+.25*p
        with np.load(ART/f"nov23_identical_bills/{seed}.npz") as old:
            np.testing.assert_array_equal(old["row"],row)
            subset=earth&repeat&old["eligible"]
        scores={}
        for k,p in pred.items():
            ok=(p>=.5)==y
            before=(base>=.5)==y
            scores[k]=dict(accuracy=float(ok.mean()),fixed=int((ok&~before).sum()),harmed=int((~ok&before).sum()),
                           repeat=int((ok&repeat).sum()),earth430=int((ok&subset).sum()))
        results[seed]=scores
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**pred)
        write(ROOT/"reports/nov23_booking_meta_weights/result.json",results)
        print(seed,scores["aligned"],flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
