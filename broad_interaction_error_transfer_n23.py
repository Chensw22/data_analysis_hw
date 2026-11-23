"""Measure new coverage beyond archived predictors and deploy agreement across two interaction models."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel,predictions
from broad_bill_exception_n23 import load,strata
from broad_billing_response_surface_n23 import REPORT
from broad_new_coverage_union_n23 import collect
from broad_partition_evidence_n23 import ART,EXTRA,SOURCES,SEEDS,write
from broad_release_miss_coverage_n23 import summarize
from released_correction_audit_n23 import development


def fetch(folder,keys,row,y,fold,base,seed):
    with np.load(ART/f"nov23_{folder}/{seed}.npz") as z:
        for k,v in (("row",row),("y",y),("fold",fold),("base",base)):
            np.testing.assert_array_equal(z[k],v)
        return [z[k] for k in keys]


def joint(base,tree,surface,halves,full,y,groups,fold):
    def logodds(v): return logit(v.clip(.002,.998))
    offset=logodds(base)
    main=np.c_[offset,logodds(tree[0]),logodds(surface[0]),logodds(full)]
    dx=logodds(tree[1])-main[:,1]
    dy=logodds(surface[1])-main[:,2]
    mutual=np.where(dx*dy>0,np.sign(dx)*np.minimum(abs(dx),abs(dy)),0.)
    half_change=logodds(halves.mean(0))-logodds(full)
    support=((halves>=.5)==(mutual[None,:]>0)).mean(0)
    pair=np.c_[main,dx,dy,half_change]
    designs={"main":main,"separate":pair,"confirmed":np.c_[pair,mutual,mutual*support]}
    out={k:np.full(len(y),np.nan) for k in designs}
    for f in np.unique(fold):
        ids=np.flatnonzero(fold==f)
        for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=groups[ids]):
            t,q=ids[a],ids[b]
            assert not np.intersect1d(groups[t],groups[q]).size
            for k,x in designs.items():
                model=LogisticRegression(C=.5,max_iter=500).fit(x[t],y[t])
                out[k][q]=model.predict_proba(x[q])[:,1]
    return out


def run():
    row,y,base,old,_,_,_=panel()
    ids,labels,frame,g,_=load()
    np.testing.assert_array_equal(ids,row);np.testing.assert_array_equal(labels,y)
    labels,release,m=development()
    np.testing.assert_array_equal(labels,y)
    miss=np.logical_and.reduce([p!=y for p in release.values()])
    state=strata(frame)
    paid=miss&y&(state==2)
    zero=miss&~y&(state==0)
    masks=dict(miss=miss,core=paid|zero,paid=paid,zero=zero,repeat=m["repeat"])
    archived,history,new=[],{},{}
    for i,s in enumerate(SEEDS):
        b,extra,fold=predictions(row,y,s,{**EXTRA,**dict(list(SOURCES.items())[:4])})
        np.testing.assert_array_equal(b,base[i])
        recent=collect(row,y,b,s,fold)
        bank=np.column_stack([old[i],extra,*recent.values()])
        assert bank.shape==(len(y),61)
        archived.append((bank>=.5)==y[:,None])
        tree=fetch("context_interaction_audit",["additive","age_sleep_deck"],row,y,fold,b,s)
        surface=fetch("billing_response_surface",["additive","interaction"],row,y,fold,b,s)
        choices=fetch("adaptive_training_coalitions",["native_local"],row,y,fold,b,s)
        with np.load(ART/f"nov23_broad_support_trajectories/{s}.npz") as z:
            for k,v in (("row",row),("y",y),("fold",fold)): np.testing.assert_array_equal(z[k],v)
            half=z["bank"][z["subsets"]==2];full=z["bank"][-1]
        assert len(half)==6
        for k,v in dict(additive_tree=tree[0],interaction_tree=tree[1],additive_surface=surface[0],
                        interaction_surface=surface[1],adaptive_subsets=choices[0],mean_half=half.mean(0)).items():
            new.setdefault(k,[]).append((v>=.5)==y)
        for k,v in joint(b,tree,surface,half,full,y,g,fold).items():
            assert np.isfinite(v).all()
            history.setdefault(k,[]).append((v>=.5)==y)
    old_hit=np.stack(archived)
    already=old_hit.all(0).any(1)
    never=~old_hit.any((0,2))
    stable={k:np.logical_and.reduce(v) for k,v in new.items()}
    stable["any"]=np.logical_or.reduce(list(stable.values()))
    coverage={k:dict(previous=int((mask&already).sum()),new={
        key:dict(stable=int((mask&h).sum()),additional=int((mask&~already&h).sum()),
                 previously_never=int((mask&never&h).sum())) for key,h in stable.items()}) for k,mask in masks.items()}
    assert coverage["core"]["previous"]==259
    stats={k:summarize(np.stack(v),(base>=.5)==y,masks) for k,v in history.items()}
    write(REPORT/"transfer.json",dict(coverage=coverage,combined=stats))
    print(coverage["core"],stats)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
