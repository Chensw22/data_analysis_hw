"""Remove or amplify the registration-bill interaction learned from a held-out teacher's scores."""
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load, strata
from broad_billing_response_surface_n23 import REPORT, surface_inputs
from broad_error_vote_tables_n23 import bank
from broad_partition_evidence_n23 import ART, SEEDS, evaluate, write
from released_correction_audit_n23 import development


def kernel(x):
    d=cdist(x,x,metric="sqeuclidean")
    # Positive-distance median avoids the atom of exactly zero bills.
    scale=max(np.median(d[d>0]),1e-8)
    return (.5*np.exp(-d/scale)+.5*np.exp(-4*d/scale))


def centered(k,t,q):
    kt=k[np.ix_(t,t)]
    kq=k[np.ix_(q,t)]
    mean=kt.mean(0)
    return kt-mean-mean[:,None]+mean.mean(),kq-mean-kq.mean(1)[:,None]+mean.mean()


def projection(reg,bill,z,t,q,state):
    mean=z[t].mean()
    aa,aq=centered(reg,t,q)
    bb,bq=centered(bill,t,q)
    added,query=aa+bb,aq+bq
    full,cf=aa*bb,aq*bq
    order=t.copy()
    rng=np.random.default_rng(953)
    for s in np.unique(state[t]):
        keep=state[t]==s
        order[keep]=rng.permutation(t[keep])
    sa,sq=centered(reg,order,q)
    pred={}
    for name,v,w in (("additive",added,query),("interaction",added+full,query+cf),
                     ("shuffled",added+sa*bb,query+sq*bq)):
        gram=v.copy()
        gram.flat[::len(gram)+1]+=.6
        coef=solve(gram,z[t]-mean,assume_a="pos")
        pred[name]=mean+w@coef
    return pred


def run():
    row,y,frame,g,_=load()
    labels,releases,masks=development()
    np.testing.assert_array_equal(labels,y)
    miss=np.logical_and.reduce([p!=y for p in releases.values()])
    state=strata(frame)
    paid=miss&y&(state==2)
    zero=miss&~y&(state==0)
    masks={**masks,"miss":miss,"core":paid|zero,"paid":paid,"zero":zero}
    assert [int(masks[k].sum()) for k in ("miss","paid","zero")]==[941,415,281]
    x,b=surface_inputs(frame)
    archives,history,metrics={},{},{}
    for name in [*map(str,SEEDS),"v131"]:
        fold,base,ref,_,_=bank(name,row,y)
        z=logit(base.clip(.001,.999))
        out={k:np.full(len(y),np.nan) for k in ("additive","interaction","shuffled","update_removed",
                            "update_doubled","update_calibrated","update_interaction","update_shuffled")}
        for f in np.unique(fold):
            ids=np.flatnonzero(fold==f)
            rmat,bmat=kernel(x[ids]),kernel(b[ids])
            for a,qi in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
                # The teacher was evaluated on this entire outer fold. The
                # following two sets of bookings serve only imitation/calibration.
                sums={k:np.zeros(len(qi)) for k in out}
                for it,ic in GroupKFold(2,shuffle=True,random_state=489).split(a,groups=g[ids[a]]):
                    t,c=a[it],a[ic]
                    assert not (np.intersect1d(g[ids[t]],g[ids[c]]).size or np.intersect1d(g[ids[a]],g[ids[qi]]).size)
                    batch=np.r_[c,qi]
                    p=projection(rmat,bmat,z[ids],t,batch,state[ids])
                    delta=p["interaction"]-p["additive"]
                    true_z=z[ids[batch]]
                    for k in p:
                        sums[k]+=.5*p[k][len(c):]
                    sums["update_removed"]+=.5*expit((true_z-delta)[len(c):])
                    sums["update_doubled"]+=.5*expit((true_z+delta)[len(c):])
                    for key,design in (("calibrated",true_z[:,None]),("interaction",np.c_[true_z,delta]),
                                ("shuffled",np.c_[true_z,p["shuffled"]-p["additive"]])):
                        cal=LogisticRegression(C=.5,max_iter=400).fit(design[:len(c)],y[ids[c]])
                        sums["update_"+key]+=.5*cal.predict_proba(design[len(c):])[:,1]
                for k,v in sums.items():
                    out[k][ids[qi]]=expit(v) if not k.startswith("update") else v
        metrics[name]={k:float(np.mean((z-logit(out[k].clip(1e-10,1-1e-10)))**2))
                       for k in ("additive","interaction","shuffled")}
        archives[name],hits=evaluate(y,base,ref,out,fold,name=="v131",masks)
        if name!="v131":
            for k,(hit,_) in hits.items():
                history.setdefault(k,[]).append(hit)
        folder=ART/"nov23_teacher_interaction_ablation"
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**out)
        print(name,metrics[name],[(k,v["fixed"],v["harmed"]) for k,v in archives[name].items()],flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in masks.items()} for k,v in history.items()}
    write(REPORT/"teacher_ablation.json",dict(plans=archives,teacher_logit_mse=metrics,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
