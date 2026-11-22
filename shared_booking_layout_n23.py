"""Use cross-deck co-bookings as empirical links between cabin corridors."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import spsolve
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS
from class_analogue_experts_n23 import design
from earth_bill_order_n23 import BILLS
from exaone_validation_data import compact_features
from relational_error_corrector import write


def layout(raw):
    cabin=raw.Cabin.str.split("/",expand=True)
    slots=pd.DataFrame(dict(deck=cabin[0],side=cabin[2],block=pd.to_numeric(cabin[1])//8))
    valid=slots.notna().all(axis=1)
    distinct=slots[valid].drop_duplicates().sort_values(["deck","side","block"]).reset_index(drop=True)
    index=pd.MultiIndex.from_frame(distinct)
    node=index.get_indexer(pd.MultiIndex.from_frame(slots))
    group=raw.PassengerId.str.split("_").str[0]
    src,dst,weight=[],[],[]
    for _,corridor in distinct.groupby(["deck","side"]):
        ids=corridor.index.to_numpy()
        src.extend(ids[:-1]);dst.extend(ids[1:])
        weight.extend(1/np.diff(corridor.block))
    chain=len(src)
    bridges=[]
    for g,rows in group.groupby(group).indices.items():
        stops=np.unique(node[rows])
        stops=stops[stops>=0]
        for i,j in combinations(stops,2):
            bridges.append((g,i,j))
    edges=pd.DataFrame(bridges,columns=["booking","left","right"])
    a=edges.left.to_numpy()
    b=edges.right.to_numpy()
    def corridor(ids):
        x=distinct.iloc[ids]
        return (x.deck+x.side).to_numpy()
    def make(u,v,w):
        adjacency=coo_matrix((w,(u,v)),shape=(len(distinct),len(distinct))).tocsr()
        return laplacian(adjacency+adjacency.T).tocsr()
    maps={"corridor":make(src,dst,weight)}
    rng=np.random.default_rng(125)
    null=b.copy()
    for _,ids in pd.DataFrame(dict(left=corridor(a),right=corridor(b))).groupby(["left","right"]).indices.items():
        null[ids]=rng.permutation(b[ids])
    for name,end in [("shared_booking",b),("shuffled",null)]:
        maps[name]=make(np.r_[src,a],np.r_[dst,end],np.r_[weight,np.ones(len(a))])
    return node,maps,dict(blocks=len(distinct),known_people=int(valid.sum()),corridor_links=chain,
             links=len(edges),bookings=edges.booking.nunique(),cross_deck=int((distinct.deck.to_numpy()[a]!=distinct.deck.to_numpy()[b]).sum()))


def offset(lap,node,t,q,prior,y,paid):
    n=lap.shape[0]
    result=np.zeros(len(q))
    for kind in (None,False,True):
        mask=(node[t]>=0) & ((paid[t]==kind) if kind is not None else True)
        query=(node[q]>=0) & ((paid[q]==kind) if kind is not None else True)
        k=node[t[mask]]
        prob=prior[mask]
        observed=np.bincount(k,weights=y[t[mask]]-prob,minlength=n)
        information=np.bincount(k,weights=prob*(1-prob),minlength=n)
        surface=spsolve((4*lap+diags(.35+information)).tocsc(),observed)
        result[query]+=.5*surface[node[q[query]]]
    return result


def run():
    data={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            data[seed]={k:z[k] for k in ("row","y","base","fold")}
    row,y=data[SEEDS[0]]["row"],data[SEEDS[0]]["y"]
    full=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in {"Name","Transported"})
                    for s in ("train","test")],ignore_index=True)
    node,maps,summary=layout(full)
    node=node[row]
    raw=full.iloc[row]
    group=raw.PassengerId.str.split("_").str[0].to_numpy()
    earth=raw.HomePlanet.eq("Earth").to_numpy()
    paid=raw[BILLS].gt(0).any(axis=1).to_numpy()
    frame=compact_features(raw).reset_index(drop=True)
    repeat=np.logical_and.reduce([(v["base"]>=.5)!=y for v in data.values()])
    out=ART/"nov23_shared_booking_layout"
    out.mkdir(exist_ok=True)
    results={}
    for seed,v in data.items():
        base,fold=v["base"],v["fold"]
        np.testing.assert_array_equal(row,v["row"])
        np.testing.assert_array_equal(y,v["y"])
        with np.load(ART/f"nov23_identical_bills/{seed}.npz") as prior:
            np.testing.assert_array_equal(prior["row"],row)
            identical=prior["eligible"]
        pred={k:base.copy() for k in [*maps,"teacher","teacher_with_links"]}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(group[t],group[q]).size
            train_prob=np.empty(len(t))
            query_scores=[]
            for a,b in GroupKFold(3,shuffle=True,random_state=234).split(t,groups=group[t]):
                _,x=design(frame,t[a])
                reg=HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=15,min_samples_leaf=20,
                                     l2_regularization=2,early_stopping=False,random_state=573)
                reg.fit(x[t[a]],y[t[a]])
                train_prob[b]=reg.predict_proba(x[t[b]])[:,1]
                query_scores.append(reg.decision_function(x[q]))
            teacher=np.mean(query_scores,axis=0)
            pred["teacher"][q]=expit(teacher)
            for name,lap in maps.items():
                effect=offset(lap,node,t,q,train_prob,y,paid)
                pred[name][q]=expit(logit(base[q].clip(.0001,.9999))+effect)
                if name=="shared_booking":
                    pred["teacher_with_links"][q]=expit(teacher+effect)
        scores={}
        for name,p in pred.items():
            ok=(p>=.5)==y
            before=(base>=.5)==y
            scores[name]=dict(accuracy=float(ok.mean()),fixed=int((ok&~before).sum()),harmed=int((~ok&before).sum()),
                repeat=int((ok&repeat).sum()),earth804=int((ok&earth&repeat).sum()),
                earth430=int((ok&earth&repeat&identical).sum()))
        results[seed]=scores
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,base=base,fold=fold,**pred)
        print(seed,scores["shared_booking"],flush=True)
    write(ROOT/"reports/nov23_shared_booking_layout/result.json",dict(layout=summary,results=results))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
