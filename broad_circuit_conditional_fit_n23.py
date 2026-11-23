"""Train a generative mixture directly for the transport decision, preserving its component structure."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit,logsumexp,softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel
from broad_bill_exception_n23 import load
from broad_class_transport_n23 import encode
from broad_error_scope_balance_n23 import archived_votes
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_population_circuit_n23 import field_block,fit_circuit
from broad_release_miss_coverage_n23 import summarize
from broad_twin_surfaces_n23 import REPORT,scopes


def normalized(params,starts,field):
    root=params[0]-logsumexp(params[0])
    # Normalize each field separately within each latent class component.
    vals=params[1:]
    maximum=np.maximum.reduceat(vals,starts,axis=0)
    sums=np.add.reduceat(np.exp(vals-maximum[field]),starts,axis=0)
    return root,vals-(maximum+np.log(sums))[field]


def fit(initial,features,y,starts,field,mode):
    nfield=len(starts)
    rooted=mode=="root_only"
    size=initial[:1].shape if rooted else initial.shape
    target=np.eye(2)[y.astype(int)]
    joint_weight=(1./nfield if mode=="hybrid" else 0.)
    def cost(vector):
        shift=vector.reshape(size)
        theta=initial.copy()
        if rooted:
            theta[:1]+=shift
        else:
            theta+=shift
        root,tables=normalized(theta,starts,field)
        joint=(features@tables+root).reshape(len(y),2,-1)
        class_lp=logsumexp(joint,axis=2)
        norm=logsumexp(class_lp,axis=1)
        true_lp=(target*class_lp).sum(1)
        loss=(norm-true_lp).sum()-joint_weight*true_lp.sum()
        derivative=(softmax(class_lp,axis=1)-(1+joint_weight)*target)[:,:,None]*softmax(joint,axis=2)
        derivative=derivative.reshape(len(y),-1)
        total=derivative.sum(0)
        root_grad=total-np.exp(root)*total.sum()
        if rooted:
            grad=root_grad[None,:]
        else:
            field_grad=features.T@derivative-np.exp(tables)*total
            grad=np.vstack([root_grad,field_grad])
        penalty=2.
        return loss+.5*penalty*(shift*shift).sum(),(grad+penalty*shift).ravel()
    result=minimize(cost,np.zeros(np.prod(size)),method="L-BFGS-B",jac=True,
                    options={"maxiter":400,"ftol":2e-9})
    assert result.success,result.message
    theta=initial.copy()
    theta[:size[0]]+=result.x.reshape(size)
    return theta


def infer(theta,x,starts,field):
    root,table=normalized(theta,starts,field)
    ll=logsumexp((x@table+root).reshape(x.shape[0],2,-1),axis=2)
    return ll[:,1]-ll[:,0]


def main():
    row,y,frame,g,_=load()
    encoded=encode(frame)
    names=["HomePlanet","Destination","VIP","Deck","Side","CabinNumber","BookingSize","Age","CryoSleep",
           "FoodCourt","ShoppingMall","Spa","VRDeck","RoomService"]
    cols=frame.columns.get_indexer(names)
    assert min(cols)>=0 and len(set(cols))==encoded.shape[1]
    block=field_block(encoded,cols,encoded.max(0)+1)
    data,field,starts=block
    masks=scopes(y,frame)
    a,b,bases,archive,*_=panel()
    np.testing.assert_array_equal(a,row); np.testing.assert_array_equal(b,y)
    old=(archived_votes(row,y,bases,archive)==y[None,:,None]).all(0).any(1)
    history,refs={},[]
    store=ART/"nov23_circuit_conditional_fit"
    store.mkdir(exist_ok=True)
    for i,seed in enumerate(SEEDS):
        with np.load(ART/f"nov23_population_circuit/{seed}.npz") as z:
            for k,v in (("row",row),("y",y),("base",bases[i])): np.testing.assert_array_equal(z[k],v)
            fold,base,original=z["fold"],z["base"],z["one_subtype"]
        direct={}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(g[t],g[q]).size
            params=[]
            for label in (False,True):
                tr=t[y[t]==label]
                root,tab=fit_circuit([block],tr,q,4,1,262,return_parameters=True)
                params.append(np.vstack([root+np.log(len(tr)/len(t)),tab[0]]))
            initial=np.column_stack(params)
            for k in ("generative","root_only","hybrid","conditional"):
                theta=initial if k=="generative" else fit(initial,data[t],y[t],starts,field,k)
                direct.setdefault(k,np.zeros(len(y)))[q]=infer(theta,data[q],starts,field)
        np.testing.assert_allclose(expit(direct["generative"]),original,atol=1.e-11)
        out={mode:expit(v) for mode,v in direct.items()}
        for mode,score in direct.items():
            x=np.c_[logit(base.clip(.001,.999)),direct["generative"]]
            if mode!="generative":
                x=np.c_[x,score-direct["generative"]]
            cal=np.zeros(len(y))
            for f in np.unique(fold):
                idx=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(idx,groups=g[idx]):
                    t,q=idx[a],idx[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    cal[q]=LogisticRegression(C=.5,max_iter=400).fit(x[t],y[t]).predict_proba(x[q])[:,1]
            out["combined_"+mode]=cal
        ref=(base>=.5)==y
        refs.append(ref)
        for k,p in out.items():
            h=(p>=.5)==y
            history.setdefault(k,[]).append(h)
        np.savez_compressed(store/f"{seed}.npz",row=row,y=y,base=base,fold=fold,**out)
        print(seed,[(k,int((v[-1]&~ref).sum()),int((~v[-1]&ref).sum())) for k,v in history.items()],flush=True)
    novel=masks["repeat"]&~old&np.logical_and.reduce(history["generative"])
    assert novel.sum()==91
    stats={k:summarize(np.stack(h),np.stack(refs),{**masks,"novel91":novel,"beyond67":masks["repeat"]&~old})
           for k,h in history.items()}
    write(REPORT/"conditional_fit.json",stats)


if __name__=="__main__":
    with threadpool_limits(2):
        main()
