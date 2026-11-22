"""Fit neural residuals with explicit control of gradient conflicts on training mistakes."""
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from torch import nn

from archive_error_recovery_n23 import ART, ROOT, SEEDS, panel
from class_analogue_experts_n23 import design
from conditional_field_value_n23 import field_blocks
from conditional_view_density_n23 import fit_score
from exaone_validation_data import compact_features
from grouped_five_way_coverage_n23 import verify
from relational_error_corrector import write


def residual_fit(train_x,query_x,labels,offset,query_offsets,mode):
    rng=np.random.default_rng(476)
    torch.manual_seed(476)
    sizes=[len(train_x),len(query_x)]
    x=torch.tensor(np.vstack([train_x,query_x]),dtype=torch.float32)
    p=torch.tensor(offset,dtype=torch.float32)
    y=torch.from_numpy(labels.astype(np.float32))
    misleading=(offset>=0)!=labels
    if mode=="shuffled_projection":
        shuffle=np.random.default_rng(244)
        # Keep difficulty prevalence in each predicted direction.
        for side in [False,True]:
            idx=np.flatnonzero((offset>=0)==side)
            misleading[idx]=misleading[shuffle.permutation(idx)]
    hard=torch.from_numpy(misleading)
    network=nn.Sequential(nn.Linear(x.shape[1],48),nn.LayerNorm(48),nn.SiLU(),
                          nn.Linear(48,24),nn.SiLU(),nn.Linear(24,1))
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    params=tuple(network.parameters())
    opt=torch.optim.SGD(params,lr=.075)
    conflicts,steps=0,0
    predictions=[]
    for epoch in range(32):
        for ix in np.array_split(rng.permutation(sizes[0]),int(np.ceil(sizes[0]/320))):
            delta=network(x[ix]).ravel()
            loss=nn.functional.binary_cross_entropy_with_logits(p[ix]+delta,y[ix],reduction="none")+.03*delta.square()
            hard_ids=hard[ix]
            easy_ids=~hard_ids
            ge=torch.autograd.grad(loss[easy_ids].mean(),params,retain_graph=True)
            gh=torch.autograd.grad(loss[hard_ids].mean(),params)
            dot=sum((a*b).sum() for a,b in zip(ge,gh))
            norm=sum(a.square().sum() for a in ge).clamp(min=1e-12)
            fraction=float(hard_ids.float().mean()) if mode=="ordinary" else .35
            conflicts+=float(dot)<0
            steps+=1
            # SGD (without momentum) acts in precisely the gradient coordinates
            # used for projection. Remove only the conflicting component of the
            # difficult-example gradient, retaining its orthogonal information.
            correction=(dot/norm).clamp(max=0) if mode in ("projection","shuffled_projection") else 0.
            opt.zero_grad()
            for param,e,h in zip(params,ge,gh):
                param.grad=((1-fraction)*e+fraction*(h-correction*e)).detach()
            opt.step()
        if epoch in (19,25,31):
            with torch.no_grad():
                d=network(x[sizes[0]:]).ravel().numpy()
            predictions.append(expit(query_offsets+d[:,None]).mean(axis=1))
    return np.mean(predictions,axis=0), conflicts/steps


def run():
    torch.set_num_threads(2)
    rows,y,bref,bank,_,_,_=panel()
    correct=((bank>=.5)==y[None,:,None])
    strong=(correct.mean(1)>=((bref>=.5)==y).mean(1)[:,None]-.01).all(0)
    repeated=~((bref>=.5)==y).any(axis=0)
    weak=repeated & correct.all(0).any(1)&~correct.all(0)[:,strong].any(1)
    deep=repeated & ~correct.any(axis=(0,2))
    assert weak.sum()==335 and deep.sum()==252
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{part}.csv",usecols=lambda c:c not in ("Name","Transported"))
                   for part in ("train","test")],ignore_index=True)
    blocks,group=field_blocks(raw,rows)
    main=np.column_stack(list(blocks.values()))
    frame=compact_features(raw.iloc[rows]).reset_index(drop=True)
    size=raw.PassengerId.str.split("_").str[0].value_counts().loc[group].to_numpy()
    out=ART/"nov23_competing_gradients"
    out.mkdir(exist_ok=True)
    report={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,rows,y,fold)
        modes=["ordinary","weighted","projection","shuffled_projection"]
        estimates={k:np.full(len(y),np.nan) for k in ["teacher",*modes]}
        frequency={k:[] for k in modes}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(group[t],group[q]).size
            teacher=np.full(len(t),np.nan)
            query=[]
            for a,b in GroupKFold(3,shuffle=True,random_state=seed+int(f)).split(t,groups=group[t]):
                assert not np.intersect1d(group[t[a]],group[t[b]]).size
                logits=fit_score(main[t[a]],y[t[a]],main[np.r_[t[b],q]],None)
                teacher[b]=logits[:len(b)]
                query.append(logits[len(b):])
            assert np.isfinite(teacher).all()
            query=np.column_stack(query)
            estimates["teacher"][q]=expit(query).mean(1)
            _,features=design(frame,t)
            features=np.column_stack([features,np.log1p(size)])
            for mode in modes:
                output,rate=residual_fit(features[t],features[q],y[t],teacher,query,mode)
                estimates[mode][q]=output
                frequency[mode].append(rate)
            print(seed,"fold",f,flush=True)
        ref=(base>=.5)==y
        teacher_hits=(estimates["teacher"]>=.5)==y
        metrics={}
        for mode,p in estimates.items():
            assert np.isfinite(p).all()
            for name,s in [(mode,p),(mode+"_quarter",.75*base+.25*p)]:
                now=(s>=.5)==y
                metrics[name]=dict(accuracy=float(now.mean()),fixed=int((now&~ref).sum()),harmed=int((~now&ref).sum()),
                                   weak335=int(now[weak].sum()),deep252=int(now[deep].sum()),
                                   fixes_vs_teacher=int((now&~teacher_hits).sum()),harms_vs_teacher=int((~now&teacher_hits).sum()))
        np.savez_compressed(out/f"{seed}.npz",row=rows,y=y,fold=fold,base=base,**estimates)
        report[seed]=dict(metrics=metrics,negative_gradient_dot_by_fold=frequency)
        write(ROOT/"reports/nov23_competing_gradients/result.json",dict(results=report))
        print(seed,metrics["projection"],metrics["projection_quarter"],flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
