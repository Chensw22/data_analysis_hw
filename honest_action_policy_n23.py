"""Choose competing actions by realized net repairs; confirm a proposed switch on separate bookings."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from boarding_sequence_n23 import ART,ROOT,SEEDS,hard_rows
from grouped_five_way_coverage_n23 import EXPERIMENTS,verify
from relational_error_corrector import write


def grow(x,utility,rows,depth):
    # Action zero keeps the strong reference. Every switch is charged for each
    # previously correct passenger it would harm, equally to the credit for a repair.
    total=utility[rows].sum(0)
    action=int(np.argmax(total))
    best=float(total[action])
    split=None
    if depth and len(rows)>=80:
        for j in range(x.shape[1]):
            order=rows[np.argsort(x[rows,j],kind="stable")]
            val=x[order,j]
            cuts=np.flatnonzero(val[1:]!=val[:-1])+1
            cuts=cuts[(cuts>=40)&(cuts<=len(rows)-40)]
            if not len(cuts):
                continue
            left=np.cumsum(utility[order],axis=0)[cuts-1]
            gain=left.max(1)+(total-left).max(1)
            i=int(np.argmax(gain))
            if gain[i]>best:
                pos=cuts[i]
                best=float(gain[i])
                split=(j,.5*(val[pos]+val[pos-1]),order[:pos],order[pos:])
    if split is None:
        return action
    j,cut,left,right=split
    return (j,cut,grow(x,utility,left,depth-1),grow(x,utility,right,depth-1))


def apply(tree,x,rows,actions,leaves,node=1):
    if isinstance(tree,int):
        actions[rows]=tree
        leaves[rows]=node
        return
    j,cut,left,right=tree
    low=x[rows,j]<=cut
    apply(left,x,rows[low],actions,leaves,node*2)
    apply(right,x,rows[~low],actions,leaves,node*2+1)


def suggest(x,utility,choices,fit,check,query):
    tree=grow(x,utility,fit,2)
    chosen=np.zeros(len(x),int)
    leaf=np.zeros(len(x),int)
    apply(tree,x,np.r_[check,query],chosen,leaf)
    action=chosen[query]
    allow=np.zeros(len(query),bool)
    for node in np.unique(leaf[query]):
        seen=check[leaf[check]==node]
        hit=leaf[query]==node
        candidate=action[hit][0]
        # No retuning the action after inspecting this confirmation partition.
        allow[hit]=bool(len(seen) and utility[seen,candidate].sum()>0)
    return choices[query,action],choices[query,np.where(allow,action,0)]


def run():
    people=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    dev,y=people.row.to_numpy(),people.target.to_numpy(bool)
    hard=hard_rows(dev,y)
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId","Age","CryoSleep"]).iloc[dev]
    group=raw.PassengerId.str.split("_").str[0].to_numpy()
    out=ART/"nov23_honest_switch"
    out.mkdir(exist_ok=True)
    summary={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            base,fold=ref["base"],ref["fold"]
            verify(ref,dev,y,fold)
            scores=[base,ref["learned_twice"]]
        for folder,key in EXPERIMENTS.items():
            with np.load(ART/f"nov23_{folder}/{seed}.npz") as z:
                verify(z,dev,y,fold)
                scores.append(z[key])
        scores=np.column_stack(scores)
        choices=scores>=.5
        anchor=choices[:,0]
        orient=np.where(anchor,-1,1)
        features=np.column_stack([anchor,np.abs(base-.5),
            (scores[:,1:]-base[:,None])*orient[:,None],np.std(scores,axis=1),
            raw.Age.fillna(-1),raw.CryoSleep.eq(True)])
        utility=(choices==y[:,None]).astype(int)-(anchor==y).astype(int)[:,None]
        variants={name:anchor.copy() for name in ["unconfirmed","confirmed","both_confirmed"]}
        seen=np.zeros(len(y),int)
        for k in np.unique(fold):
            ids=np.flatnonzero(fold==k)
            pieces=[ids[q] for _,q in GroupKFold(3,shuffle=True,random_state=7001).split(ids,groups=group[ids])]
            for i,q in enumerate(pieces):
                a,b=pieces[(i+1)%3],pieces[(i+2)%3]
                assert not (set(group[a])&set(group[b]) or set(group[a])&set(group[q]) or set(group[b])&set(group[q]))
                raw_q,one=suggest(features,utility,choices,a,b,q)
                _,two=suggest(features,utility,choices,b,a,q)
                variants["unconfirmed"][q]=raw_q
                variants["confirmed"][q]=one
                variants["both_confirmed"][q]=np.where(one==two,one,anchor[q])
                seen[q]+=1
        assert (seen==1).all()
        result={}
        for name,p in variants.items():
            fixed=(p==y)&(anchor!=y)
            harmed=(p!=y)&(anchor==y)
            result[name]=dict(accuracy=float((p==y).mean()),fixed=int(fixed.sum()),harmed=int(harmed.sum()),
                               hard447_correct=int((hard&(p==y)).sum()),fold_net=[int((fixed[fold==k]).sum()-(harmed[fold==k]).sum()) for k in np.unique(fold)])
        np.savez_compressed(out/f"{seed}.npz",row=dev,y=y,fold=fold,base=base,**variants)
        summary[str(seed)]=result
        print(seed,result,flush=True)
    write(ROOT/"reports/nov23_honest_switch/result.json",summary)


if __name__=="__main__":
    run()
