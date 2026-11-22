"""Low-rank ANOVA factors with at most one contribution per manifest field."""
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART, ROOT, SEEDS
from grouped_five_way_coverage_n23 import verify
from manifest_tensor_outcomes import hat
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write


def encode(frame):
    berth=frame.Cabin.str.split("/",expand=True)
    group=frame.PassengerId.str.split("_").str[0]
    categories=[frame.HomePlanet,frame.CryoSleep,frame.Destination,frame.VIP,berth[0],berth[2]]
    blocks=[pd.get_dummies(v.fillna("missing").astype(str)).to_numpy(float) for v in categories]
    blocks.extend([hat(frame.Age,[0,5,12,17,25,40,60,80]),hat(pd.to_numeric(berth[1]),[0,250,500,750,1000,1250,1500,2000]),
                   hat(group.map(group.value_counts()),[1,2,3,4,6,8])])
    # Each input is an original service bill; no combination is allowed to count the
    # same observation again via correlated engineered totals or percentage features.
    for name in ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]:
        blocks.append(hat(np.log1p(frame[name]),np.log1p([0,1,100,500,850,1400,3000,10000,40000])))
    width=max(v.shape[1] for v in blocks)
    tensor=np.stack([np.pad(v,((0,0),(0,width-v.shape[1]))) for v in blocks],axis=1)
    return torch.from_numpy(tensor.astype(np.float32))


def interactions(z,order):
    """Sum products for distinct feature fields via elementary symmetric polynomials."""
    s=z.sum(dim=1)
    s2=z.square().sum(dim=1)
    if order==2:
        return (s.square()-s2)/2
    return (s**3-3*s*s2+2*(z**3).sum(dim=1))/6


class FieldFactors(nn.Module):
    def __init__(self,fields,width,order,rank=12):
        super().__init__()
        self.linear=nn.Parameter(torch.zeros(fields,width))
        self.bias=nn.Parameter(torch.zeros(()))
        self.params=nn.ParameterList([nn.Parameter(torch.randn(fields,width,rank)*.06)
                                      for _ in range(order-1)])

    def forward(self,x):
        result=self.bias+torch.einsum("bfw,fw->b",x,self.linear)
        for order,param in enumerate(self.params,start=2):
            # Mix active neighboring basis bins first, then multiply distinct fields.
            z=torch.einsum("bfw,fwr->bfr",x,param)
            result=result+interactions(z,order).sum(1)/np.sqrt(param.shape[-1])
        return result


def fit(x,y,t,q,order):
    torch.manual_seed(733)
    model=FieldFactors(*x.shape[1:],order)
    opt=torch.optim.AdamW(model.parameters(),lr=.012,weight_decay=.03)
    rng=np.random.default_rng(733)
    predictions=[]
    for epoch in range(36):
        for batch in np.array_split(rng.permutation(t),int(np.ceil(len(t)/384))):
            loss=nn.functional.binary_cross_entropy_with_logits(model(x[batch]),y[batch])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2)
            opt.step()
        if epoch in [23,29,35]:
            with torch.no_grad():
                predictions.append(torch.cat([model(x[i]).sigmoid() for i in np.array_split(q,8)]).numpy())
    return np.mean(predictions,axis=0)


def run():
    torch.set_num_threads(2)
    torch.manual_seed(119)
    tiny=torch.randn(3,5,7)
    for order in (2,3):
        exact=sum(tiny[:,list(key)].prod(1) for key in combinations(range(5),order))
        torch.testing.assert_close(exact,interactions(tiny,order),atol=2e-5,rtol=1e-5)
    catalog=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    dev,y=catalog.row.to_numpy(copy=True),catalog.target.to_numpy(bool)
    labels,old,challenge=read_panel()
    np.testing.assert_array_equal(labels,y)
    already_wrong=~old.any(0)
    hard=already_wrong&~challenge.any(axis=(0,1))
    assert (already_wrong.sum(),hard.sum())==(1070,466)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ["Transported","Name"])
                   for s in ["train","test"]],ignore_index=True)
    group=raw.PassengerId.iloc[dev].str.split("_").str[0].to_numpy()
    x=encode(raw)[dev]
    target=torch.from_numpy(y.astype(np.float32))
    out=ART/"nov23_field_factors"
    out.mkdir(exist_ok=True)
    results={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            base,fold=z["base"],z["fold"]
            verify(z,dev,y,fold)
        candidates={}
        metrics={}
        for order in (1,2,3):
            pred=np.full(len(y),np.nan)
            for k in np.unique(fold):
                t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
                assert not np.intersect1d(group[t],group[q]).size
                pred[q]=fit(x,target,t,q,order)
            assert np.isfinite(pred).all()
            name=f"order{order}"
            candidates[name]=pred
            for label,score in [(name,pred),(name+"_blend",.75*base+.25*pred)]:
                good=(score>=.5)==y
                bad=(base>=.5)!=y
                metrics[label]=dict(accuracy=float(good.mean()),fixed=int((good&bad).sum()),harmed=int((~good&~bad).sum()),
                                    recovered1070=int(good[already_wrong].sum()),recovered466=int(good[hard].sum()))
        results[str(seed)]=metrics
        np.savez_compressed(out/f"{seed}.npz",row=dev,y=y,fold=fold,base=base,**candidates)
        write(ROOT/"reports/nov23_field_factors/result.json",results)
        print(seed,metrics,flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
