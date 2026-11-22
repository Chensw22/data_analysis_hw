"""Learn outcome-specific densities for the discrete bill states and their active amounts."""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS
from earth_bill_order_n23 import BILLS
from exaone_validation_data import CATEGORICAL, compact_features
from relational_error_corrector import write


def network(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in,48),nn.SiLU(),nn.Linear(48,n_out))


class BillModel(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.context=network(width,24)
        self.prior=nn.Linear(24,1)
        self.states=nn.ModuleList([network(25+3*i,3) for i in range(5)])
        self.normal=network(40,10)
        self.couplings=nn.ModuleList([network(45,10) for _ in range(4)])
        for block in self.couplings:
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)

    def transform(self,z,active,cond):
        jacobian=z.new_zeros(len(z))
        for i,block in enumerate(self.couplings):
            keep=(torch.arange(5,device=z.device)%2==i%2)
            change=active*(~keep)
            shift,scale=block(torch.cat([cond,z*keep],dim=1)).chunk(2,dim=1)
            scale=.7*torch.tanh(scale)*change
            z=(z-shift*change)*torch.exp(-scale)
            jacobian-=scale.sum(1)
        return z,jacobian

    def evidence(self,reg,state,amount,cls):
        h=self.context(reg)
        log_prior=-F.binary_cross_entropy_with_logits(self.prior(h).flatten(),cls,reduction="none")
        ctx=torch.cat([h,cls[:,None]],dim=1)
        indicators=F.one_hot(state,num_classes=3).float().flatten(1)
        discrete=log_prior
        for i,net in enumerate(self.states):
            inputs=torch.cat([ctx,indicators[:,:3*i]],dim=1)
            discrete=discrete-F.cross_entropy(net(inputs),state[:,i],reduction="none")
        active=(state==2).float()
        cond=torch.cat([ctx,indicators],dim=1)
        mean,scale=self.normal(cond).chunk(2,dim=1)
        scale=1.7*torch.tanh(scale)
        z=(amount-mean)*torch.exp(-scale)*active
        constant=-(.5*np.log(2*np.pi)+scale)*active
        diagonal=constant.sum(1)-.5*z.square().sum(1)
        latent,jac=self.transform(z,active,cond)
        flow=constant.sum(1)-.5*latent.square().sum(1)+jac
        return torch.stack([discrete,discrete+diagonal,discrete+flow],dim=1)


def fit(reg,state,amount,y,t,q):
    torch.manual_seed(471)
    rng=np.random.default_rng(614)
    model=BillModel(reg.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=.0018,weight_decay=.008)
    for _ in range(26):
        for ix in np.array_split(rng.permutation(t),15):
            joint=model.evidence(reg[ix],state[ix],amount[ix],y[ix])
            # Both continuous variants train with exactly the same labelled rows.
            loss=-joint[:,1:].mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),4.)
            opt.step()
    with torch.no_grad():
        scores=[model.evidence(reg[q],state[q],amount[q],torch.full((len(q),),c)) for c in (0.,1.)]
        return torch.sigmoid(scores[1]-scores[0]).numpy()


def run():
    data={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            data[seed]={k:z[k] for k in ("row","y","base","fold")}
    first=data[SEEDS[0]]
    row,y=first["row"],first["y"]
    full=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"})
    booking=full.PassengerId.str.split("_").str[0]
    size=booking.map(booking.value_counts()).to_numpy()[row]
    group=booking.to_numpy()[row]
    raw=full.iloc[row]
    earth=raw.HomePlanet.eq("Earth").to_numpy()
    frame=compact_features(raw)
    bills=raw[BILLS].to_numpy()
    state=torch.tensor(np.where(np.isnan(bills),0,np.where(bills==0,1,2)),dtype=torch.long)
    normalized=(np.log1p(np.nan_to_num(bills))-5.)/2
    amount=torch.tensor(normalized,dtype=torch.float32)
    labels=torch.tensor(y,dtype=torch.float32)
    repeat=np.logical_and.reduce([(v["base"]>=.5)!=y for v in data.values()])
    out=ART/"nov23_conditional_bill_flow"
    out.mkdir(exist_ok=True)
    results={}
    for seed,v in data.items():
        np.testing.assert_array_equal(row,v["row"])
        np.testing.assert_array_equal(y,v["y"])
        base,fold=v["base"],v["fold"]
        p=np.full((len(y),3),np.nan)
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(group[t],group[q]).size
            enc=OneHotEncoder(sparse_output=False,handle_unknown="ignore").fit(frame.iloc[t][CATEGORICAL])
            reg=np.column_stack([enc.transform(frame[CATEGORICAL]),frame.Age.fillna(28)/35,frame.Age.isna(),
                  frame.CabinNumber.fillna(850)/1000,frame.CabinNumber.isna(),np.log1p(size)])
            p[q]=fit(torch.tensor(reg,dtype=torch.float32),state,amount,labels,t,q)
        assert np.isfinite(p).all()
        estimates=dict(zip(("states","diagonal","flow"),p.T))
        for name,s in list(estimates.items()):
            estimates[name+"_quarter"]=.75*base+.25*s
        scores={}
        for name,s in estimates.items():
            correct=(s>=.5)==y
            ref=(base>=.5)==y
            scores[name]=dict(accuracy=float(correct.mean()),fixed=int((correct&~ref).sum()),
                harmed=int((~correct&ref).sum()),repeat=int((correct&repeat).sum()),
                earth_repeat=int((correct&earth&repeat).sum()),
                logloss=float(-np.log(np.where(y,s,1-s).clip(1e-9)).mean()))
        results[seed]=scores
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,base=base,fold=fold,**estimates)
        write(ROOT/"reports/nov23_conditional_bill_flow/result.json",results)
        print(seed,scores["flow"],flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
