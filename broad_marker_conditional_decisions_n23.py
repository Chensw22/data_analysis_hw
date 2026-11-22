"""Test whether known marker values identify when to trust the corresponding exception models."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from broad_observed_agreement_n23 import ART,REPORT,SEEDS,historical,load,strata,witnesses,evaluate,write

OUT=ART/"nov23_marker_conditional_decisions"


def shuffled(bits,base,spread,ids,rng):
    bins=np.digitize(base[ids],np.quantile(base[ids],np.arange(1,9)/9))
    bins=bins*3+np.digitize(spread[ids],np.quantile(spread[ids],[1/3,2/3]))
    order=np.arange(len(ids))
    for k in np.unique(bins):
        sel=np.flatnonzero(bins==k)
        order[sel]=rng.permutation(sel)
    return bits[ids[order]]


def fit(train,test,target):
    scale=StandardScaler().fit(train)
    model=LogisticRegression(C=.04,max_iter=500).fit(scale.transform(train),target)
    return model.predict_proba(scale.transform(test))[:,1]


def designs(anchor,direct,ex,bit):
    # Keep the shrinkage identical in the real/shuffled branches. Conditional
    # terms can represent distinct reliability when a witness is true vs false.
    main=np.c_[anchor,direct,ex]
    s=2*bit.astype(float)-1
    return dict(global_scores=main,
                values_only=np.c_[anchor,direct,s],
                conditional=np.c_[main,s,(ex-anchor[:,None])*s],
                joint=np.c_[main,s,((ex-anchor[:,None])[:,:,None]*s[:,None,:]).reshape(len(s),-1)])


def main():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    cases=list(witnesses(frame))
    bits=np.column_stack([v for _,v,_ in cases])
    scope=dict(**masks,repeat_solo=masks["repeat"]&(frame.BookingSize==1).to_numpy(),
               repeat_paid=masks["repeat"]&(strata(frame)==2)&y)
    plans,history={},{}
    OUT.mkdir(exist_ok=True)
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_observed_agreement/{name}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            base,ref,fold=z["base"],z["ref"],z["fold"]
            direct=np.column_stack([logit(z[k+"_direct"]) for k,_,_ in cases])
            ex=np.column_stack([logit(z[k+"_agreement"]) for k,_,_ in cases])
        anchor=logit(base.clip(.001,.999))
        real=designs(anchor,direct,ex,bits)
        spread=(ex>=0).mean(1)
        scores={}
        for f in np.unique(fold):
            ids=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=515).split(ids,groups=g[ids]):
                t,q=ids[a],ids[b]
                assert not np.intersect1d(g[t],g[q]).size
                rng=np.random.default_rng(521)
                tr,te=[designs(anchor[ix],direct[ix],ex[ix],shuffled(bits,base,spread,ix,rng))
                       for ix in (t,q)]
                for k in real:
                    scores.setdefault("update_"+k,np.zeros(len(y)))[q]=fit(real[k][t],real[k][q],y[t])
                    if k in ("conditional","joint"):
                        scores.setdefault("update_placebo_"+k,np.zeros(len(y)))[q]=fit(tr[k],te[k],y[t])
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",scope)
        if name!="v131":
            for k,(hit,_) in hits.items():history.setdefault(k,[]).append(hit)
        plans[name]=result
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**scores)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in scope.items()} for k,v in history.items()}
    write(REPORT/"conditional_decisions.json",dict(plans=plans,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
