"""Train a categorical restricted Boltzmann machine by exact masked-field conditionals."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import logit
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import ART,ROOT,SEEDS,load,historical,strata,evaluate,write,model
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import BILLS
from broad_refit_dissent_n23 import features
from exaone_validation_data import compact_features,CATEGORICAL

OUT=ART/"nov23_manifest_energy"
REPORT=ROOT/"reports/nov23_error_tail_ranking"


def manifest():
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{n}.csv",usecols=lambda c:c not in ("Name","Transported"))
                   for n in ("train","test")],ignore_index=True)
    group=raw.PassengerId.str.split("_").str[0].astype(int)
    frame=compact_features(raw)
    frame["BookingSize"]=group.map(group.value_counts())
    return frame,group.to_numpy()


def encode(frame,t):
    codes=[];ncat=[]
    for key in frame:
        col=frame[key]
        if key in CATEGORICAL:
            val=pd.factorize(col,sort=True)[0]
        else:
            # Keep a distinct zero class for spending and distinguish a missing entry.
            x=col.to_numpy(float)
            ok=np.isfinite(x)
            positive=ok&(x>0) if key in BILLS else ok
            sample=x[t[positive[t]]]
            cuts=np.unique(np.quantile(sample,np.arange(1,12)/12))
            val=np.where(ok,1,0)
            val[positive]=2+np.searchsorted(cuts,x[positive])
        codes.append(val)
        ncat.append(int(val.max()+1))
    return np.column_stack(codes),ncat


class ManifestEnergy(nn.Module):
    def __init__(self,sizes,counts):
        super().__init__()
        self.sizes=sizes
        self.starts=np.r_[0,np.cumsum(sizes)[:-1]].tolist()
        self.weights=nn.Parameter(.035*torch.randn(sum(sizes),24))
        bias=np.concatenate([np.log((c+1)/(c.sum()+len(c))) for c in counts])
        self.visible=nn.Parameter(torch.tensor(bias,dtype=torch.float32))
        self.hidden=nn.Parameter(torch.zeros(24))
        self.register_buffer("offset",torch.tensor(self.starts))

    def contributions(self,codes):
        w=self.weights[codes+self.offset]
        return w,self.hidden+w.sum(1)

    def conditional(self,parts,total,j):
        s=self.starts[j]
        w=self.weights[s:s+self.sizes[j]]
        bias=self.visible[s:s+self.sizes[j]]
        # Marginalize each binary hidden unit analytically for every possible
        # value of the masked field. The observed value is removed first.
        return F.softplus((total-parts[:,j])[:,None,:]+w).sum(-1)+bias


def learn(codes,sizes,t):
    counts=[np.bincount(codes[t,j],minlength=n) for j,n in enumerate(sizes)]
    torch.manual_seed(543)
    model=ManifestEnergy(sizes,counts)
    opt=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=.01)
    rng=np.random.default_rng(931)
    data=torch.tensor(codes,dtype=torch.long)
    for _ in range(30):
        for ids in np.array_split(rng.permutation(t),int(np.ceil(len(t)/384))):
            batch=data[ids]
            parts,total=model.contributions(batch)
            fields=rng.choice(len(sizes),size=4,replace=False)
            loss=torch.stack([F.cross_entropy(model.conditional(parts,total,j),batch[:,j]) for j in fields]).mean()
            opt.zero_grad();loss.backward();opt.step()
    return model,counts


@torch.no_grad()
def transform(model,counts,codes):
    data=torch.tensor(codes,dtype=torch.long)
    records=[];field_loss=[]
    for batch in data.split(512):
        parts,total=model.contributions(batch)
        columns=[torch.sigmoid(total)]
        per_field=[]
        for j,n in enumerate(model.sizes):
            pred=F.softmax(model.conditional(parts,total,j),dim=1)
            obs=batch[:,j:j+1]
            prior=torch.tensor((counts[j]+1)/(np.sum(counts[j])+n),dtype=pred.dtype)
            logprob=pred.gather(1,obs).log()
            expected=(pred*torch.arange(n,dtype=pred.dtype)[None,:]).sum(1,keepdim=True)
            size=float(max(n-1,1))
            columns.extend([logprob-prior[obs].log(),pred.max(1,keepdim=True).values,
                            (obs-expected)/size,pred[:,0:1]])
            per_field.append(-logprob)
        field_loss.append(torch.cat(per_field,1).numpy())
        records.append(torch.cat(columns,1).numpy())
    return np.vstack(records),np.vstack(field_loss)


def run():
    row,y,local,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig);np.testing.assert_array_equal(y,labels)
    frame,party=manifest()
    np.testing.assert_array_equal(g,party[row])
    assert frame.iloc[row].reset_index(drop=True).equals(local)
    t,h=next(GroupShuffleSplit(1,test_size=.2,random_state=134).split(frame,groups=party))
    assert not np.intersect1d(party[t],party[h]).size
    codes,sizes=encode(frame,t)
    shuffled=codes.copy()
    rng=np.random.default_rng(653)
    for j in range(codes.shape[1]):shuffled[t,j]=rng.permutation(codes[t,j])
    out,losses={},{}
    for k,z in (("real",codes),("sham",shuffled)):
        fitted,counts=learn(z,sizes,t)
        representation,l=transform(fitted,counts,codes)
        out[k]=representation[row]
        losses[k]=dict(heldout_nll_by_field=l[h].mean(0).tolist(),heldout_mean=float(l[h].mean()))
    plain=np.stack([-np.log((np.bincount(codes[t,j],minlength=n)[codes[h,j]]+1)/(len(t)+n))
                    for j,n in enumerate(sizes)])
    losses["marginal"]=dict(heldout_nll_by_field=plain.mean(1).tolist(),heldout_mean=float(plain.mean()))
    print("reconstruction",[(k,v["heldout_mean"]) for k,v in losses.items()],flush=True)
    x=features(local)[0]
    xs=dict(raw=x,latent=out["real"][:,:24],surprise=out["real"][:,24:],
            real=np.c_[x,out["real"]],sham=np.c_[x,out["sham"]])
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/"representations.npz",row=row,**out)
    scope=dict(**masks,repeat_solo=masks["repeat"]&(local.BookingSize==1).to_numpy(),
               repeat_paid=masks["repeat"]&(strata(local)==2)&y)
    plans,history={},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_companion_information/{name}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            base,fold,ref=z["base"],z["fold"],z["ref"]
        scores={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            for k,design in xs.items():
                scores.setdefault(k,np.zeros(len(y)))[q]=model(220).fit(design[t],y[t]).predict_proba(design[q])[:,1]
        for k in ("raw","real","sham"):
            cols=[base,scores["raw"]]+([] if k=="raw" else [scores[k]])
            design=np.column_stack([logit(v.clip(.001,.999)) for v in cols])
            p=np.zeros(len(y))
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=543).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    p[q]=calibrated(design[t],design[q],y,t)
            scores["update_"+k]=p
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",scope)
        if name!="v131":
            for k,(hit,_) in hits.items():history.setdefault(k,[]).append(hit)
        plans[name]=result
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,ref=ref,base=base,**scores)
        print(name,result["update_real"]["accuracy"],flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in scope.items()} for k,v in history.items()}
    write(REPORT/"manifest_energy.json",dict(fields=list(frame.columns),reconstruction=losses,plans=plans,stable=stable))


if __name__=="__main__":
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
