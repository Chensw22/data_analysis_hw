"""Hold reference bookings out and recover an individual probability from pair equality."""
import numpy as np
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from broad_interbooking_relations_n23 import (ART,REPORT,SEEDS,anchors,calibrated,design,evaluate,historical,
                    learner,load,relational,strata,write)


def independent(frame,y,g,t,q):
    i,j=next(GroupKFold(3,shuffle=True,random_state=982).split(t,groups=g[t]))
    fit,bank=t[i],t[j]
    assert not np.intersect1d(g[fit],g[bank]).size
    assert not np.intersect1d(g[t],g[q]).size
    x,phi=design(frame,fit)
    x=np.c_[x,np.log(frame.BookingSize)/2]
    phi=np.c_[phi,np.log(frame.BookingSize)/2]
    train_anchors=anchors(x,g,fit,fit)
    left=np.repeat(fit,train_anchors.shape[1]);right=train_anchors.ravel()
    clf=learner().fit(relational(phi,left,right),y[left]==y[right])
    direct=learner().fit(phi[fit],y[fit])
    p=direct.predict_proba(phi[q])[:,1]
    bank_p=np.full(len(y),np.nan)
    bank_p[bank]=direct.predict_proba(phi[bank])[:,1]
    # Queries and reference labels are both held out of both fits.
    near=anchors(x,g,bank,q)[:,:8]
    random=np.random.default_rng(309).choice(bank,(len(q),40))
    refs=np.column_stack([near,random])
    assert (g[refs]!=g[q,None]).all()
    prob=clf.predict_proba(relational(phi,np.repeat(q,refs.shape[1]),refs.ravel()))[:,1].reshape(refs.shape)
    unit=2*bank_p[refs]-1
    sign=2*prob-1
    truth=2*y[refs].astype(float)-1
    # Independent outcomes with probabilities p,r have signed equality expectation
    # (2p-1)(2r-1). Invert by least squares with a one-unit ridge towards indifference.
    expected=.5+.5*(unit*sign).sum(1)/(1+(unit**2).sum(1))
    known=.5+.5*(truth*sign).mean(1)
    factor=.5+.5*(2*p[:,None]-1)*unit
    correct=y[q,None]==y[refs]
    rng=np.random.default_rng(781)
    placebo=unit.copy()
    # Keep the two anchor pools separate when breaking pair-specific information.
    for block in (slice(0,8),slice(8,None)):
        placebo[:,block]=rng.permuted(unit[:,block],axis=1)
    fake=.5+.5*(placebo*sign).sum(1)/(1+(placebo**2).sum(1))
    pair_loss={k:float(log_loss(correct.ravel(),v.ravel())) for k,v in (("learned",prob),("product",factor))}
    return {k:np.clip(v,.001,.999) for k,v in dict(product_direct=p,decoded=expected,observed=known,
                                                   shuffled=fake).items()},pair_loss


def run():
    row,y,frame,g,_=load()
    orig,label,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,label)
    masks=dict(**masks,repeat_paid=masks["repeat"]&(strata(frame)==2)&y,
               solo_repeat=masks["repeat"]&(frame.BookingSize.to_numpy()==1))
    records,hist={},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{name}.npz") as z:
            for field,v in (("row",row),("y",y)): np.testing.assert_array_equal(z[field],v)
            fold,base,ref=z["fold"],z["base"],z["ref"]
        scores={}; losses=[]
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            vals,loss=independent(frame,y,g,t,q)
            losses.append(loss)
            for k,v in vals.items():
                scores.setdefault(k,np.zeros(len(y)))[q]=v
        for k,terms in (("direct",[]),("relational",["decoded","observed"]),("placebo",["shuffled","observed"])):
            ds=np.column_stack([logit(v.clip(.001,.999)) for v in (base,scores["product_direct"],*[scores[s] for s in terms])])
            pred=np.zeros(len(y))
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=698).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    pred[q]=calibrated(ds[t],ds[q],y,t)
            scores["update_"+k]=pred
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",masks)
        records[name]=dict(models=result,pair_losses=losses)
        if name!="v131":
            for k,(hit,_) in hits.items(): hist.setdefault(k,[]).append(hit)
        print(name,result["update_relational"]["accuracy"],flush=True)
    stable={k:{s:int((m&np.logical_and.reduce(h)).sum()) for s,m in masks.items()} for k,h in hist.items()}
    write(REPORT/"factor.json",dict(plans=records,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):run()
