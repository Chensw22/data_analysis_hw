"""Infer passenger outcomes by learning whether two different bookings share an outcome."""
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from threadpoolctl import threadpool_limits

from class_analogue_experts_n23 import design
from exaone_validation_data import compact_features
from grouped_five_way_coverage_n23 import verify
from neighborhood_registration_rank_n23 import neighbors
from population_rescue_map_n23 import ART, ROOT, SEEDS, read_panel
from relational_error_corrector import write
from spend_simplex_likelihood_n23 import BILLS
from transported_subclasses_n23 import stats


class PairNet(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(width,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.direct=nn.Linear(32,1)
        self.compare=nn.Sequential(nn.Linear(96,48),nn.SiLU(),nn.Linear(48,1))

    def relation(self,a,b):
        return self.compare(torch.cat((a+b,(a-b).abs(),a*b),dim=-1)).squeeze(-1)


def fit(x,y,groups,t,near,mode):
    torch.manual_seed(823)
    model=PairNet(x.shape[1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.015)
    rng=np.random.default_rng(961)
    donor_targets=y.clone()
    if mode=="placebo":
        donor_targets[t]=y[np.random.default_rng(587).permutation(t)]
    for _ in range(26):
        for ix in np.array_split(rng.permutation(len(t)),int(np.ceil(len(t)/192))):
            rows=t[ix]
            local=near[ix[:,None],rng.integers(0,near.shape[1],size=(len(ix),3))]
            global_rows=rng.choice(t,size=(len(ix),3))
            overlap=groups[rows,None]==groups[global_rows]
            while overlap.any():
                global_rows[overlap]=rng.choice(t,size=int(overlap.sum()))
                overlap=groups[rows,None]==groups[global_rows]
            donors=np.column_stack([local,global_rows])
            a=model.embed(x[rows])
            loss=nn.functional.binary_cross_entropy_with_logits(model.direct(a).ravel(),y[rows])
            if mode!="direct":
                b=model.embed(x[donors])
                logit=model.relation(a[:,None,:],b)
                same=(y[rows,None]==donor_targets[donors]).float()
                loss=loss+nn.functional.binary_cross_entropy_with_logits(logit,same)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict(model,x,anchors,ids):
    h=model.embed(x)
    direct=model.direct(h[ids]).ravel().numpy()
    # Independent labeled template sets are balanced by outcome. No validation
    # passenger or related booking can become an anchor.
    signals=[]
    for bank in anchors:
        score=[]
        for sub in np.array_split(ids,int(np.ceil(len(ids)/192))):
            logits=model.relation(h[sub,None,:],h[bank][None,:,:])
            score.extend(logits.mean(1).numpy())
        signals.append(score)
    return direct,np.column_stack(signals)


def run():
    torch.set_num_threads(2)
    row=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"})
    booking=raw.PassengerId.str.split("_").str[0]
    size=booking.map(booking.value_counts()).to_numpy()[row]
    group=booking.to_numpy()[row]
    frame=compact_features(raw.iloc[row]).reset_index(drop=True)
    bills=frame[BILLS].to_numpy(float)
    spend=np.column_stack([np.log1p(np.nan_to_num(bills)),3*np.isnan(bills),3*(bills>0)])
    y,old,others=read_panel()
    ty=torch.from_numpy(y.astype(np.float32))
    persistent=~old.any(0)
    hard=persistent&~others.any(axis=(0,1))
    out=ART/"nov23_passenger_relation"
    out.mkdir(exist_ok=True)
    result={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,row,y,fold)
        pred={}
        for k in np.unique(fold):
            outer=np.flatnonzero(fold!=k)
            q=np.flatnonzero(fold==k)
            a,b=next(GroupShuffleSplit(n_splits=1,test_size=.20,random_state=885).split(outer,groups=group[outer]))
            t,c=outer[a],outer[b]
            assert not np.intersect1d(group[t],group[c]).size and not np.intersect1d(group[outer],group[q]).size
            mat,_=design(frame,t)
            xt=torch.from_numpy(np.column_stack([mat,np.log1p(size)]).astype(np.float32))
            near,_=neighbors(spend,group,t,t,seed+int(k))
            rng=np.random.default_rng(816)
            anchors=[rng.choice(t[y[t]==label],64,replace=False) for label in [False,True]]
            for mode in ["direct","relation","placebo"]:
                model=fit(xt,ty,group,t,near,mode)
                direct,rel=predict(model,xt,anchors,np.r_[c,q])
                signals={"direct":direct[:,None]}
                if mode!="direct":
                    signals["anchors"]=rel
                    signals["both"]=np.column_stack([direct,rel])
                for kind,features in signals.items():
                    name=mode+"_"+kind
                    net=LogisticRegression(C=.5,max_iter=250).fit(features[:len(c)],y[c])
                    pred.setdefault(name,np.full(len(y),np.nan))[q]=net.predict_proba(features[len(c):])[:,1]
            print(seed,"fold",k,flush=True)
        summary={}
        for key,p in pred.items():
            assert np.isfinite(p).all()
            for name,score in [(key,p),(key+"_blend",.75*base+.25*p)]:
                summary[name]=stats(score>=.5,base>=.5,y,persistent,hard,size==1)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**pred)
        result[seed]=summary
        write(ROOT/"reports/nov23_passenger_relation/result.json",dict(results=result))
        print(seed,[(k,round(v["accuracy"]*100,3)) for k,v in summary.items() if k.endswith("blend")],flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
