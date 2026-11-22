"""Fit a task-specific classifier on frozen contextual passenger representations from unseen bookings."""
import sys
from grouped_tabswift_n23 import OUT, SOURCE, DIGEST, ART, ROOT, prepare, compact_features, read_panel, verify, write
import hashlib

import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from conditional_calibration_n23 import offsets


def embeddings(net,x,ids):
    captures=[]
    def save(module,args):
        values=args[0]
        captures.append(values[:,-len(ids):].detach().cpu().numpy())
    handle=net.model_.icl_predictor.decoder.register_forward_pre_hook(save)
    try:
        p=net.predict_proba(x[ids])[:,1]
    finally:
        handle.remove()
    latent=np.concatenate(captures,axis=0)
    assert latent.shape[0]==2 and latent.shape[1]==len(ids)
    return p,latent.transpose(1,0,2).reshape(len(ids),-1)


def run():
    from tabswift.classifier import TabSwiftClassifier
    seed=20261112
    torch.set_num_threads(2)
    weight=OUT/"swift.ckpt"
    assert hashlib.sha256(weight.read_bytes()).hexdigest()==DIGEST
    rows=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    y,hits,others=read_panel()
    hard=~hits.any(0)&~others.any(axis=(0,1))
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ["Name","Transported"])
    frame=compact_features(raw).iloc[rows].reset_index(drop=True)
    g=raw.PassengerId.str.split("_").str[0].to_numpy()[rows]
    with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
        fold,base=z["fold"],z["base"]
        verify(z,rows,y,fold)
    out=ART/"nov23_tabswift_readout"
    out.mkdir(exist_ok=True)
    pools={name:np.zeros(len(y)) for name in ["frozen","score_only","representation","permuted"]}
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.intersect1d(g[t],g[q]).size
        for part,(a,b) in enumerate(GroupKFold(2,shuffle=True,random_state=722).split(t,groups=g[t])):
            support,cal=t[a],t[b]
            assert not np.intersect1d(g[support],g[cal]).size
            x=prepare(frame,support)
            net=TabSwiftClassifier(n_estimators=2,device="cpu",batch_size=1,model_path=weight,allow_auto_download=False,
                                   use_amp=False,random_state=43+int(k))
            net.fit(x[support],y[support])
            pc,ec=embeddings(net,x,cal)
            pq,eq=embeddings(net,x,q)
            # Project and whiten using only the supervised readout's training rows.
            proj=PCA(n_components=24,whiten=True,svd_solver="randomized",random_state=622).fit(ec)
            feature=np.vstack([proj.transform(ec),proj.transform(eq)]).clip(-5,5)
            prior=np.r_[pc,pq]
            linear=logit(prior.clip(.005,.995))
            train=np.arange(len(cal))
            test=np.arange(len(cal),len(prior))
            labels=np.r_[y[cal],np.zeros(len(q))]  # query labels need never enter the readout
            null=feature.copy()
            bins=np.floor(10*prior).astype(int).clip(0,9)
            rng=np.random.default_rng(957+part)
            for partition in [train,test]:
                for bin_id in np.unique(bins[partition]):
                    ix=partition[bins[partition]==bin_id]
                    null[ix]=feature[rng.permutation(ix)]
            score=np.column_stack([np.ones(len(prior)),linear])
            for name,matrix in [("score_only",score),("representation",np.column_stack([score,feature])),
                                ("permuted",np.column_stack([score,null]))]:
                penalty=np.full(matrix.shape[1],32.)
                penalty[:2]=2.
                pools[name][q]+=.5*offsets(matrix,linear,labels,train,test,penalty)
            pools["frozen"][q]+=.5*pq
        print("completed readout fold",int(k),flush=True)
    old=(base>=.5)==y
    metrics={}
    for name,p in pools.items():
        assert np.isfinite(p).all() and np.all((p>0)&(p<1))
        for tag,score in [(name,p),(name+"_blend",.75*base+.25*p)]:
            hit=(score>=.5)==y
            metrics[tag]=dict(accuracy=float(hit.mean()),fixed=int((hit&~old).sum()),harmed=int((~hit&old).sum()),
                              persistent1070=int(hit[~hits.any(0)].sum()),unresolved466=int(hit[hard].sum()))
    np.savez_compressed(out/f"{seed}.npz",row=rows,y=y,fold=fold,base=base,**pools)
    write(ROOT/"reports/nov23_tabswift_readout/result.json",dict(results=metrics,
          definition="Outer training bookings split into two disjoint halves: one gives neural network support, the other learns the readout. Swap roles, average predictions. Strong baseline blends are from the original outer fold."))
    print(metrics,flush=True)


if __name__=="__main__":
    sys.path.insert(0,str(OUT/SOURCE))
    with threadpool_limits(2):
        run()
