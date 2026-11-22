"""Learn shared low-rank changes in how registration profiles modify bill effects."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load, strata
from broad_geographic_transfer_n23 import BILLS
from broad_error_vote_tables_n23 import bank
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, evaluate, write
from broad_passenger_transport_n23 import manifest_coordinates
from released_correction_audit_n23 import development

REPORT = ROOT / "reports/nov23_billing_response_surface"


def surface_inputs(frame):
    reg = frame.drop(columns=BILLS)
    x = manifest_coordinates(reg) * np.sqrt(reg.shape[1])
    age = frame.Age.to_numpy()
    x = np.c_[x, *[(age>=c).astype(float) for c in (12,18,25,40,60)]]
    bills = frame[BILLS].to_numpy()
    amount = np.log1p(np.nan_to_num(bills))
    shares = np.nan_to_num(bills)/(1+np.nansum(bills,axis=1,keepdims=True))
    b = np.c_[np.isnan(bills),bills>0,amount/7,shares,
              np.maximum(amount-5.5,0)/2,np.maximum(amount-7,0)/2,
              np.log1p(np.nansum(bills,axis=1))/7,strata(frame)==0]
    return x,b


def fold_inputs(x,b,t):
    def expand(values):
        scaled = (values-values[t].mean(0))/values[t].std(0).clip(.25)
        leaf = RandomTreesEmbedding(n_estimators=24,max_depth=3,min_samples_leaf=14,
                 random_state=459,sparse_output=False,n_jobs=1).fit(values[t]).transform(values)
        return scaled,np.column_stack([scaled,leaf/np.sqrt(6)])
    reg,reg_terms = expand(x)
    bill,bill_terms = expand(b)
    return np.c_[np.ones(len(x)),reg_terms,bill_terms],reg,bill


def solve(a,r,b,y):
    lam_main,lam_factors=4.,16.
    def main_loss(w):
        z=a@w
        loss=np.logaddexp(0,z).sum()-y@z+.5*lam_main*(w[1:]@w[1:])
        return loss,a.T@(expit(z)-y)+np.r_[0,lam_main*w[1:]]
    fit=minimize(main_loss,np.zeros(a.shape[1]),method="L-BFGS-B",jac=True,
                 options={"maxiter":450,"ftol":1e-10})
    assert fit.success,fit.message
    w0=fit.x
    residual=y-expit(a@w0)
    u,d,vt=np.linalg.svd((r.T*residual)@b,full_matrices=False)
    rank=5
    start=np.r_[w0,(.08*u[:,:rank]).ravel(),(.08*vt[:rank].T).ravel()]
    def unpack(params):
        w,left,right=np.split(params,[len(w0),len(w0)+r.shape[1]*rank])
        return w,left.reshape(-1,rank),right.reshape(-1,rank)
    def objective(params):
        w,left,right=unpack(params)
        hr,hb=r@left,b@right
        z=a@w+(hr*hb).sum(1)
        err=expit(z)-y
        loss=np.logaddexp(0,z).sum()-y@z+.5*lam_main*(w[1:]@w[1:])
        loss+=.5*lam_factors*(np.sum(left**2)+np.sum(right**2))
        grad=np.r_[a.T@err+np.r_[0,lam_main*w[1:]],
                   (r.T@(err[:,None]*hb)+lam_factors*left).ravel(),
                   (b.T@(err[:,None]*hr)+lam_factors*right).ravel()]
        return loss,grad
    fit=minimize(objective,start,method="L-BFGS-B",jac=True,
                 options={"maxiter":750,"ftol":1e-9})
    assert fit.success,fit.message
    w,left,right=unpack(fit.x)
    return w0,w,left@right.T


def run():
    row,y,frame,g,_=load()
    labels,releases,masks=development()
    np.testing.assert_array_equal(labels,y)
    missed=np.logical_and.reduce([p!=y for p in releases.values()])
    state=strata(frame)
    paid=missed&(state==2)&y
    zero=missed&(state==0)&~y
    core=paid|zero
    masks=dict(**masks,missed=missed,paid=paid,zero=zero,core=core,core_solo=core&(frame.BookingSize.to_numpy()==1))
    assert [masks[k].sum() for k in ("missed","paid","zero","core_solo")]==[941,415,281,435]
    x,b=surface_inputs(frame)
    records,history,diagnostics={},{},{}
    for name in [*map(str,SEEDS),"v131"]:
        fold,base,ref,_,_=bank(name,row,y)
        scores={};details=[]
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            a,regs,bills=fold_inputs(x,b,t)
            order=t.copy()
            rng=np.random.default_rng(186)
            for s in np.unique(state[t]):
                ix=state[t]==s
                order[ix]=rng.permutation(t[ix])
            for k,tr in (("interaction",t),("shuffled",order)):
                w0,w,cross=solve(a[t],regs[tr],bills[t],y[t].astype(float))
                z=a[q]@w+((regs[q]@cross)*bills[q]).sum(1)
                scores.setdefault(k,np.full(len(y),np.nan))[q]=expit(z).clip(1e-10,1-1e-10)
                if k=="interaction":
                    p=expit(a[q]@w0)
                    scores.setdefault("additive",np.full(len(y),np.nan))[q]=p
                    details.append(dict(fold=int(f),singular_values=np.linalg.svd(cross,compute_uv=False)[:5].tolist(),
                           median_abs_logit_change=float(np.median(abs(z-a[q]@w0)))))
        combined={}
        for k in scores:
            design=np.c_[logit(base.clip(.001,.999)),logit(scores["additive"].clip(.001,.999))]
            if k!="additive":
                shift=logit(scores[k].clip(.001,.999))-design[:,1]
                design=np.c_[design,shift]
            pred=base.clip(.001,.999).copy()
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,bval in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[bval]
                    assert not np.intersect1d(g[t],g[q]).size
                    pred[q]=LogisticRegression(C=.5,max_iter=500).fit(design[t],y[t]).predict_proba(design[q])[:,1]
            combined["update_"+k]=pred
        scores.update(combined)
        diagnostics[name]=details
        records[name],hits=evaluate(y,base,ref,scores,fold,name=="v131",masks)
        if name!="v131":
            for k,(hit,_) in hits.items():
                history.setdefault(k,[]).append(hit)
        dest=ART/"nov23_billing_response_surface"
        dest.mkdir(exist_ok=True)
        np.savez_compressed(dest/f"{name}.npz",row=row,y=y,fold=fold,ref=ref,base=base,**scores)
        print(name,[(k,s["fixed"],s["harmed"]) for k,s in records[name].items()],flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in masks.items()} for k,v in history.items()}
    write(REPORT/"results.json",dict(plans=records,stable=stable,diagnostics=diagnostics))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
