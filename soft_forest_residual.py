"""Learn oblique soft decision trees as a residual on out-of-fold model log odds."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="FALSE"
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer

import relational_error_corrector as ref
from exaone_validation_data import CATEGORICAL
from peer_bill_correction import BILLS,load_v96

ROOT=Path(__file__).resolve().parent


def encode(frame,train):
    cat=OneHotEncoder(handle_unknown="ignore",sparse_output=False).fit(frame.iloc[train][CATEGORICAL])
    num=frame[["Age","CabinNumber",*BILLS]]
    full=num.fillna(num.iloc[train].median())
    rank=QuantileTransformer(n_quantiles=256,subsample=None,random_state=20261118).fit(full.iloc[train])
    x=np.column_stack([cat.transform(frame[CATEGORICAL]),rank.transform(full),num.isna(),
                       frame[BILLS].gt(0),np.log1p(frame[BILLS].sum(axis=1).to_numpy())/8])
    return torch.tensor(2*x-1,dtype=torch.float32)


class ResidualForest(nn.Module):
    def __init__(self,x,seed,oblique):
        super().__init__()
        rng=np.random.default_rng(seed)
        trees,depth=32,4
        dimensions=x.shape[1]
        selectors=rng.integers(dimensions,size=(trees,depth))
        weights=np.eye(dimensions,dtype=np.float32)[selectors]
        init=x.numpy()[rng.integers(len(x),size=selectors.shape),selectors]
        self.directions=nn.Parameter(torch.from_numpy(weights),requires_grad=oblique)
        self.thresholds=nn.Parameter(torch.tensor(init,dtype=torch.float32))
        self.leaves=nn.Parameter(torch.zeros(trees,2**depth))
        self.register_buffer("signs",torch.tensor((np.arange(2**depth)[:,None]>>np.arange(depth))%2,dtype=torch.bool))
        self.register_buffer("initial",torch.from_numpy(weights.copy()))

    def forward(self,x):
        direction=nn.functional.normalize(self.directions,dim=-1)
        branch=torch.sigmoid(3*(torch.einsum("bf,tdf->btd",x,direction)-self.thresholds))
        mass=torch.where(self.signs,branch[:,:,None,:],1-branch[:,:,None,:]).prod(-1)
        return torch.sum(mass*self.leaves,dim=(1,2))/np.sqrt(len(self.leaves))


def fit_predict(x,labels,prior,q,seed,oblique):
    torch.manual_seed(seed)
    net=ResidualForest(x,seed,oblique)
    opt=torch.optim.Adam(net.parameters(),lr=.008)
    offsets=torch.as_tensor(ref.logit(prior),dtype=torch.float32)
    y=torch.as_tensor(labels,dtype=torch.float32)
    for _ in range(24):
        for idx in torch.randperm(len(y)).split(384):
            opt.zero_grad(set_to_none=True)
            logits=offsets[idx]+net(x[idx])
            loss=nn.functional.binary_cross_entropy_with_logits(logits,y[idx])
            loss=loss+.04*net.leaves.square().mean()+.075*(net.directions-net.initial).square().sum(-1).mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return torch.cat([net(chunk) for chunk in q.split(512)]).numpy()


def run():
    torch.set_num_threads(2)
    inp,splits,folds,bases,evidence=ref.overlap.load_authenticated()
    dev,y,raw=inp["dev"],inp["y"],bases["rawMitra"]
    frame=inp["features"].iloc[dev].reset_index(drop=True)
    shifts={name:np.zeros(len(y)) for name in ["axis","oblique"]}
    for fold,(a,b) in enumerate(splits):
        x=encode(frame,a)
        inner=ref.load_inner(fold,a,b,dev,y,evidence)
        for name in shifts:
            shifts[name][b]=fit_predict(x[a],y[a],inner,x[b],20261118+fold,name=="oblique")
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    result={}
    for name,base in [("raw",raw),("v96",load_v96(dev,folds))]:
        result[name]={}
        for kind,shift in shifts.items():
            p=expit(ref.logit(base)+shift)
            result[name][kind]=dict(metrics=ref.score(p,y),vs_base=ref.pair(p,base,y),hard546=ref.pair(p,base,y,hard),
                fold_net=[ref.pair(p[b],base[b],y[b])["net_correct"] for _,b in splits])
    ref.write(ROOT/"reports/soft_forest_residual/result.json",result)
    print(result)


if __name__=="__main__":
    run()
