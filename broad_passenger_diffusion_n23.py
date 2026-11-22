"""Compare the same passenger's denoising loss under both hypothetical outcomes."""
import json

import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

from archive_error_recovery_n23 import ART,ROOT,SEEDS,panel,predictions
from broad_dispute_leaf_evidence_n23 import cohorts
from exaone_validation_data import compact_features,CATEGORICAL
from fresh_family_rescue_n23 import NEW
from peer_bill_correction import BILLS


def encode(frame,t):
    parts,fields=[],[]
    for j,col in enumerate([*CATEGORICAL,"Age","CabinNumber",*BILLS]):
        if col in CATEGORICAL:
            choices=np.unique(frame[col].to_numpy()[t])
            block=frame[col].to_numpy()[:,None]==choices
        else:
            raw=frame[col].to_numpy(dtype=float)
            numeric=np.log1p(raw) if col in BILLS else raw
            block=np.column_stack([np.nan_to_num(numeric,nan=np.nanmedian(numeric[t])),np.isnan(raw)])
            if col in BILLS:
                block=np.column_stack((block, raw>0))
        parts.append(block)
        fields.extend([j]*block.shape[1])
    x=np.column_stack(parts).astype(float)
    standard=StandardScaler().fit(x[t])
    x=((x-standard.mean_)/np.maximum(standard.scale_,.15)).clip(-7,7)
    return torch.tensor(x,dtype=torch.float32),torch.tensor(fields)


class Denoiser(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.input=nn.Linear(width,128)
        self.time=nn.Linear(5,128)
        self.condition=nn.Embedding(3,128)
        self.body=nn.Sequential(nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128))
        self.noise=nn.Sequential(nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,width))
        self.output=nn.Linear(128,1)

    def latent(self,x,a,c):
        times=torch.cat([a,1-a,torch.sqrt(a),torch.sqrt(1-a),torch.logit(a.clamp(.001,.999))/5],dim=1)
        h=self.input(x)+self.time(times)+self.condition(c)
        return h+self.body(h)


def fit(x, field, target, t, q, seed):
    torch.manual_seed(seed)
    net=Denoiser(x.shape[1])
    opt=torch.optim.AdamW(net.parameters(),lr=.0018,weight_decay=.015)
    y=torch.tensor(target.astype(int))
    group_map=F.one_hot(field).float()
    group_map/=group_map.sum(0)
    for _ in range(48):
        for batch in torch.randperm(len(t)).split(256):
            ix=t[batch.numpy()]
            clean=x[ix]
            a=torch.rand(len(ix),1)*.96+.02
            epsilon=torch.randn_like(clean)
            noisy=torch.sqrt(a)*clean+torch.sqrt(1-a)*epsilon
            h=net.latent(noisy,a,y[ix])
            loss=(((net.noise(h)-epsilon).square()@group_map).mean())
            # Same parameters, clean measurements and a neutral condition for
            # the control head; its query input never includes the target class.
            clean_h=net.latent(clean,torch.ones_like(a),torch.full_like(y[ix],2))
            loss=loss+.3*F.binary_cross_entropy_with_logits(net.output(clean_h).ravel(),y[ix].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    direct=[]
    differences=[]
    rng=torch.Generator().manual_seed(1721)
    with torch.no_grad():
        for batch in np.array_split(q,int(np.ceil(len(q)/256))):
            clean=x[batch]
            ones=torch.ones(len(batch),1)
            hypothesis=torch.zeros(len(batch),dtype=torch.long)
            direct.append(net.output(net.latent(clean,ones,hypothesis+2)).ravel().numpy())
            by_time=[]
            for a in (.12,.35,.65,.88):
                err=torch.zeros(len(batch),int(field.max())+1)
                for _ in range(4):
                    e=torch.randn(clean.shape,generator=rng)
                    for epsilon in (e,-e):
                        noisy=np.sqrt(a)*clean+np.sqrt(1-a)*epsilon
                        mse=[]
                        for label in (0,1):
                            h=net.latent(noisy,ones*a,hypothesis+label)
                            mse.append((net.noise(h)-epsilon).square()@group_map)
                        err+=mse[0]-mse[1]
                by_time.append((err/8).numpy())
            differences.append(np.column_stack(by_time))
    return np.concatenate(direct),np.concatenate(differences)


def run():
    row,y,base,old,_,_,_=panel()
    archived=[predictions(row,y,s,dict(**NEW,native_carte=("carte","p"))) for s in SEEDS]
    historical=(np.concatenate([old,np.stack([a[1] for a in archived])],2)>=.5)==y[None,:,None]
    hard=((base>=.5)!=y).all(0)
    no_stable=hard & ~historical.all(0).any(1)
    never=hard & ~historical.any(axis=(0,2))
    covered=cohorts(row,y)["new50"]
    for path,key in (("broad_rotation_forest","rotation"),("broad_mondrian_partition","mondrian")):
        h=[]
        for seed,(_,_,fold) in zip(SEEDS,archived):
            with np.load(ART/f"nov23_{path}/{seed}.npz") as z:
                np.testing.assert_array_equal(z["row"],row)
                np.testing.assert_array_equal(z["fold"],fold)
                np.testing.assert_array_equal(z["y"],y)
                h.append((z[key]>=.5)==y)
        covered=covered|(no_stable&np.all(h,0))
    assert covered.sum()==58
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Name","Transported")).iloc[row]
    frame=compact_features(raw)
    groups=raw.PassengerId.str.split("_").str[0].to_numpy()
    out=ART/"nov23_broad_passenger_diffusion"
    out.mkdir(exist_ok=True)
    hits,metrics={},{}
    for i,(seed,(ref,_,fold)) in enumerate(zip(SEEDS,archived)):
        np.testing.assert_array_equal(base[i],ref)
        scores={k:np.full(len(y),np.nan) for k in ("loss_vote","direct","loss_fields","base_direct","base_direct_loss")}
        diffs=None
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(groups[t],groups[q]).size
            x,fields=encode(frame,t)
            pred,z=fit(x,fields,y,t,q,7342+int(f))
            if diffs is None:
                diffs=np.full((len(y),z.shape[1]),np.nan)
            diffs[q]=z
            scores["loss_vote"][q]=1/(1+np.exp(-z.mean(1)))
            scores["direct"][q]=1/(1+np.exp(-pred))
            b=logit(ref[q].clip(.001,.999))
            for a,v in GroupKFold(3,shuffle=True,random_state=718).split(q,groups=groups[q]):
                assert not np.intersect1d(groups[q[a]],groups[q[v]]).size
                for key,d in (("loss_fields",z),("base_direct",np.column_stack((b,pred))),
                              ("base_direct_loss",np.column_stack((b,pred,z)))):
                    normal=StandardScaler().fit(d[a])
                    model=LogisticRegression(C=.03,max_iter=250).fit(normal.transform(d[a]),y[q[a]])
                    scores[key][q[v]]=model.predict_proba(normal.transform(d[v]))[:,1]
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,diffs=diffs,**scores)
        metrics[seed]={}
        for name,p in scores.items():
            assert np.isfinite(p).all()
            ok=(p>=.5)==y
            hits.setdefault(name,[]).append(ok)
            ref_ok=(ref>=.5)==y
            metrics[seed][name]=dict(accuracy=ok.mean(),fixed=int((ok&~ref_ok).sum()),
                                     harmed=int((~ok&ref_ok).sum()))
        print(seed,metrics[seed],flush=True)
    coverage={}
    for name,h in hits.items():
        st=np.all(h,0)
        coverage[name]=dict(repeat=int((st&hard).sum()),new_to46=int((st&no_stable).sum()),
                            overlap58=int((st&covered).sum()),fresh=int((st&no_stable&~covered).sum()),
                            never_any=int((np.any(h,0)&never).sum()),never_stable=int((st&never).sum()))
    folder=ROOT/"reports/nov23_broad_passenger_diffusion"
    folder.mkdir(exist_ok=True)
    (folder/"result.json").write_text(json.dumps(dict(metrics=metrics,coverage=coverage),indent=2)+"\n")


if __name__=="__main__":
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
