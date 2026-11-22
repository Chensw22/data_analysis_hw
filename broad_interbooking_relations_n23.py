"""Classify equality across unrelated bookings, then decode against labeled training passengers."""
import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from broad_partition_evidence_n23 import ART,ROOT,SEEDS,evaluate,historical,load,write
from broad_boosting_path_n23 import calibrated
from class_analogue_experts_n23 import design
from broad_bill_exception_n23 import strata

OUT=ART/"nov23_interbooking_relations"
REPORT=ROOT/"reports/nov23_interbooking_relations"


def learner():
    return HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=23,min_samples_leaf=22,
                    l2_regularization=3,learning_rate=.08,early_stopping=False,random_state=1443)


def anchors(x,g,source,focal):
    rng=np.random.default_rng(1625)
    nn=NearestNeighbors(n_neighbors=44,algorithm="kd_tree").fit(x[source])
    near=source[nn.kneighbors(x[focal],return_distance=False)]
    result=[]
    for i,candidates in zip(focal,near):
        near=candidates[g[candidates]!=g[i]][:32]
        assert len(near)==32
        random=source[rng.integers(len(source),size=30)]
        random=random[g[random]!=g[i]][:8]
        assert len(random)==8
        result.append(np.r_[near[[0,2,5,9,15,20,25,31]],random])
    return np.asarray(result)


def relational(x,i,j):
    a=x[i];b=x[j]
    # Symmetry under exchanging passengers is imposed by construction.
    return np.column_stack([(a+b)/2,abs(a-b)]).astype(np.float32)


def run_fold(frame,y,g,t,q):
    assert not np.intersect1d(g[t],g[q]).size
    x,phi=design(frame,t)
    x=np.c_[x,np.log(frame.BookingSize)/2]
    phi=np.c_[phi,np.log(frame.BookingSize)/2]
    tr=anchors(x,g,t,t)
    te=anchors(x,g,t,q)
    left=np.repeat(t,tr.shape[1]); right=tr.ravel()
    assert (g[left]!=g[right]).all() and (g[q,None]!=g[te]).all()
    model=learner().fit(relational(phi,left,right),y[left]==y[right])
    a=np.repeat(q,te.shape[1])
    prob=model.predict_proba(relational(phi,a,te.ravel()))[:,1].reshape(te.shape)
    signed=np.where(y[te],prob,1-prob)
    direct=learner().fit(phi[t],y[t]).predict_proba(phi[q])[:,1]
    return dict(direct=direct,neighbor=(y[te[:,:8]].sum(1)+.5)/9,
                relation=signed.mean(1),relation_near=signed[:,:8].mean(1),
                relation_far=signed[:,8:].mean(1))


def run():
    row,y,frame,g,_=load()
    rows,labels,masks=historical()
    np.testing.assert_array_equal(row,rows)
    np.testing.assert_array_equal(y,labels)
    masks=dict(**masks,solo_repeat=masks["repeat"]&(frame.BookingSize.to_numpy()==1),
               repeat_paid=masks["repeat"]&(strata(frame)==2)&y)
    OUT.mkdir(exist_ok=True)
    records,history={},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{name}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            fold,base,ref=z["fold"],z["base"],z["ref"]
        scores={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            for k,v in run_fold(frame,y,g,t,q).items():
                scores.setdefault(k,np.zeros(len(y)))[q]=v
        for k,terms in (("control",["direct","neighbor"]),("relation",["direct","neighbor","relation_near","relation_far"])):
            x=np.column_stack([logit(p.clip(.001,.999)) for p in [base,*[scores[s] for s in terms]]])
            p=np.zeros(len(y))
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=698).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    p[q]=calibrated(x[t],x[q],y,t)
            scores["update_"+k]=p
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",masks)
        for k,(hit,_) in hits.items():
            if name!="v131": history.setdefault(k,[]).append(hit)
        records[name]=result
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**scores)
        print(name,[(k,r["fixed"],r["harmed"]) for k,r in result.items()],flush=True)
    stable={k:{s:int((m&np.logical_and.reduce(h)).sum()) for s,m in masks.items()} for k,h in history.items()}
    rel=np.logical_and.reduce(history["relation"])
    direct=np.logical_and.reduce(history["direct"])
    knn=np.logical_and.reduce(history["neighbor"])
    unique={s:int((m&rel&~direct&~knn).sum()) for s,m in masks.items()}
    write(REPORT/"results.json",dict(plans=records,stable=stable,unique_vs_controls=unique,
                                    sizes={s:int(m.sum()) for s,m in masks.items()}))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
