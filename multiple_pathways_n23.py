"""Compare alternative ways to combine full passenger, spending and registration evidence."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from exaone_validation_data import compact_features
from grouped_five_way_coverage_n23 import verify
from population_error_accounting_n23 import ART, BILLS, ROOT, SEEDS
from relational_error_corrector import write


def views(roster, rows):
    x=compact_features(roster).iloc[rows]
    b=x[BILLS].to_numpy()
    total=np.nansum(b,axis=1)
    bills=np.column_stack([np.log1p(np.nan_to_num(b))/6, np.isnan(b),b>0,
                           np.log1p(total)/7,np.nan_to_num(b)/(total[:,None]+1)])
    demographics=np.column_stack([pd.get_dummies(x[["HomePlanet","CryoSleep","Destination","VIP"]],dtype=float),
                                  x.Age.fillna(28)/30,x.Age.isna(),x.Age.lt(13)])
    group=roster.PassengerId.str.split("_").str[0]
    seats=np.column_stack([pd.get_dummies(x[["Deck","Side"]],dtype=float),
                           x.CabinNumber.fillna(700)/800,x.CabinNumber.isna(),
                           np.log1p(group.map(group.value_counts()).to_numpy()[rows])])
    blocks=[bills,demographics,seats,np.column_stack([bills,demographics,seats])]
    return [torch.from_numpy(v.astype(np.float32)) for v in blocks],group.to_numpy()[rows]


class PathModel(nn.Module):
    def __init__(self,widths):
        super().__init__()
        self.paths=nn.ModuleList(nn.Sequential(nn.Linear(w,36),nn.LayerNorm(36),nn.SiLU(),
                                               nn.Linear(36,16),nn.SiLU(),nn.Linear(16,1)) for w in widths)

    def forward(self,views,idx,rule):
        logits=torch.cat([path(v[idx]) for path,v in zip(self.paths,views)],dim=1)
        if rule=="additive":
            return logits.sum(1)/2
        # Product of four survival probabilities. Center the gates so zero input
        # gives 50% for all three heads at identical parameter initialization.
        gate_bias=np.log(np.expm1(np.log(2.)/len(views)))
        intensity=nn.functional.softplus(logits+gate_bias).sum(1)
        # log(exp(intensity)-1) = log P(any event) - log P(no event).
        any_logit=intensity+torch.log(-torch.expm1(-intensity))
        return any_logit if rule=="any_positive" else -any_logit


def fit(v,yt,t,q,rule):
    torch.manual_seed(241)
    model=PathModel([x.shape[1] for x in v])
    opt=torch.optim.AdamW(model.parameters(),lr=.0017,weight_decay=.025)
    rng=np.random.default_rng(1221)
    average=[]
    for epoch in range(34):
        for ix in np.array_split(rng.permutation(t),int(np.ceil(len(t)/256))):
            score=model(v,ix,rule)
            loss=nn.functional.binary_cross_entropy_with_logits(score,yt[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch in (21,27,33):
            with torch.no_grad():
                average.append(torch.sigmoid(model(v,q,rule)).numpy())
    return np.mean(average,axis=0)


def run():
    torch.set_num_threads(2)
    inventory=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    rows,y=inventory.row.to_numpy(),inventory.target.to_numpy(bool)
    yt=torch.from_numpy(y.astype(np.float32))
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in {"Name","Transported"})
                   for s in ("train","test")],ignore_index=True)
    x,g=views(raw,rows)
    with np.load(ART/"nov23_archive_recovery/masks.npz") as masks:
        np.testing.assert_array_equal(rows,masks["row"])
        hard,deep=masks["originally_unresolved"],masks["still_unresolved"]
    out=ART/"nov23_multiple_pathways"
    out.mkdir(exist_ok=True)
    report={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            fold,base=ref["fold"],ref["base"]
            verify(ref,rows,y,fold)
        scores={n:np.full(len(y),np.nan) for n in ("additive","any_positive","any_negative")}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(g[t],g[q]).size
            for rule,values in scores.items():
                values[q]=fit(x,yt,t,q,rule)
        scores["symmetric"]=(scores["any_positive"]+scores["any_negative"])/2
        ref=(base>=.5)==y
        result={}
        for name,p in scores.items():
            assert np.isfinite(p).all()
            for label,s in [(name,p),(name+"_quarter",.75*base+.25*p)]:
                ok=(s>=.5)==y
                result[label]=dict(accuracy=float(ok.mean()),fixed=int((ok&~ref).sum()),harmed=int((~ok&ref).sum()),
                                   correct_on_365=int(ok[hard].sum()),correct_on_252=int(ok[deep].sum()),
                                   hard_positive=int((ok&hard&y).sum()),hard_negative=int((ok&hard&~y).sum()))
        np.savez_compressed(out/f"{seed}.npz",row=rows,y=y,fold=fold,base=base,**scores)
        report[seed]=result
        write(ROOT/"reports/nov23_multiple_pathways/result.json",dict(results=report))
        print(seed,result["symmetric"],result["symmetric_quarter"],flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
