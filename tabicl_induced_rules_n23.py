"""Induce supervised cross-field rules on a group-disjoint split for TabICL contexts."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from tabicl_routed_context_n23 import ART,ROOT,WEIGHT,DIGEST,CAT,BILLS,hard_masks,infer


def indicators(x,y,discovery):
    encoded=pd.get_dummies(x,columns=CAT,dummy_na=False,prefix_sep="=")
    encoded=encoded.fillna(-1).astype(float)
    keys=list(encoded)
    names=np.asarray([col.split("=")[0] for col in keys])
    model=ExtraTreesClassifier(n_estimators=36,max_depth=4,min_samples_leaf=40,
                               random_state=289,n_jobs=2)
    model.fit(encoded.iloc[discovery],y[discovery])
    values=encoded.to_numpy(np.float32)
    candidates=[]
    prior=y[discovery].mean()
    for tree in model.estimators_:
        nodes=tree.apply(values)
        used=tree.decision_path(values[discovery]).tocsc()
        shape=tree.tree_
        for leaf in np.flatnonzero(shape.children_left<0):
            indices=used[:,leaf].indices
            freq=len(indices)/len(discovery)
            if not .06<=freq<=.4:
                continue
            sample=discovery[indices[0]]
            path=tree.decision_path(values[[sample]]).indices
            active=shape.feature[path]
            if len(set(names[active[active>=0]]))<2:
                continue
            lift=float(y[discovery[indices]].mean()-prior)
            score=abs(lift)*np.sqrt(freq/(1-freq))
            candidates.append((score,nodes==leaf,lift))
    selected=[]
    for score,col,lift in sorted(candidates,key=lambda v:-v[0]):
        a=col[discovery]
        if any((a&b[discovery]).sum()/(a|b[discovery]).sum()>.72 for b in selected):
            continue
        selected.append(col)
        if len(selected)==8:
            break
    if len(selected)!=8:
        raise ValueError("Insufficient cross-field rules")
    return np.column_stack(selected).astype(float)


def run():
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    seed=20261112
    with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
        row,fold,y,base=[z[k] for k in ["row","fold","y","base"]]
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Transported","Name")).iloc[row].reset_index(drop=True)
    group=raw.PassengerId.str.split("_").str[0].to_numpy()
    x=raw.drop(columns=["PassengerId","Cabin"]).copy()
    x[["Deck","CabinNumber","Side"]]=raw.Cabin.str.split("/",expand=True)
    x.CabinNumber=pd.to_numeric(x.CabinNumber)
    x[CAT]=x[CAT].fillna("__MISSING__").astype(str)
    hard447,hard356=hard_masks(row,y)
    output=ART/"nov23_tabicl_rules"
    output.mkdir(exist_ok=True)
    ps={name:np.zeros(len(row)) for name in ["plain","rules","permuted_rules"]}
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not set(group[t])&set(group[q])
        for idx,(a,b) in enumerate(GroupKFold(2,shuffle=True,random_state=483).split(t,groups=group[t])):
            d,s=t[a],t[b]
            assert not set(group[d])&set(group[s])
            cols=indicators(x,y,d)
            rng=np.random.default_rng(680+int(k)*2+idx)
            noise=cols.copy()
            # Permute rows jointly to preserve correlations between rule columns.
            noise[s]=cols[rng.permutation(s)]
            noise[q]=cols[rng.permutation(q)]
            def append(block):
                return x.join(pd.DataFrame(block,columns=[f"cross_{j}" for j in range(8)]))
            for name,v in [("plain",x),("rules",append(cols)),("permuted_rules",append(noise))]:
                ps[name][q]+=.5*infer(v,y,s,q,71+int(k))
            print(int(k),idx,flush=True)
        np.savez_compressed(output/f"fold{k}.npz",row=row[q],y=y[q],**{n:v[q] for n,v in ps.items()})
    summary={}
    for name,p in ps.items():
        assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
        for tag,score in [(name,p),(name+"_plus_base",.5*(p+base))]:
            ok=(score>=.5)==y
            basemiss=(base>=.5)!=y
            summary[tag]=dict(accuracy=float(ok.mean()),fixed=int((basemiss&ok).sum()),harmed=int((~basemiss&~ok).sum()),
              hard447_correct=int(ok[hard447].sum()),hard356_correct=int(ok[hard356].sum()),
              fold_accuracy=[float(ok[fold==k].mean()) for k in np.unique(fold)])
    for ref in ["plain","permuted_rules"]:
        a=(ps[ref]>=.5)==y
        b=(ps["rules"]>=.5)==y
        summary["rules_vs_"+ref]=dict(fixed=int((b&~a).sum()),harmed=int((~b&a).sum()))
    np.savez_compressed(output/f"{seed}.npz",row=row,fold=fold,y=y,base=base,**ps)
    r=ROOT/"reports/nov23_tabicl_rules/result.json"
    r.parent.mkdir(exist_ok=True)
    r.write_text(json.dumps(dict(seed=seed,checkpoint=DIGEST,results=summary),indent=2)+"\n")
    print(summary,flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
