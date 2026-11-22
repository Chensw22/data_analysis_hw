"""Train on counts of outcomes in overlapping groups of passengers, hiding individual labels from the objective."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from broad_direct_accuracy_n23 import basis
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,historical,evaluate,interval,teachers,write
from class_analogue_experts_n23 import design

OUT=ART/"nov23_broad_count_supervision"
REPORT=ROOT/"reports/nov23_broad_count_supervision"
REPEATS=3


def partition(order,groups,size=8):
    order=list(map(int,order))
    bags=[]
    while order:
        bag=[]
        used=set()
        i=0
        while len(bag)<size and i<len(order):
            j=order[i]
            if groups[j] in used:
                i+=1
                continue
            used.add(groups[j])
            bag.append(order.pop(i))
        bags.append(bag)
    flat=np.concatenate(bags)
    np.testing.assert_array_equal(np.sort(flat),np.arange(len(groups)))
    return bags


def counts(x,groups,y,random):
    rng=np.random.default_rng(1223)
    result={}
    for _ in range(REPEATS):
        projection=x@rng.normal(size=x.shape[1])+rng.normal(scale=1e-5,size=len(x))
        order=rng.permutation(len(x)) if random else np.argsort(projection,kind="stable")
        for bag in partition(order,groups):
            result.setdefault(len(bag),[]).append(bag)
    blocks=[]
    for n,items in result.items():
        ids=np.asarray(items)
        blocks.append((ids,y[ids].sum(1).astype(int)))
    return blocks


def convolve(prev,p):
    new=prev*(1-p[:,None])
    new[:,1:]+=prev[:,:-1]*p[:,None]
    return new


def observed_counts(p,total):
    # Prefix and suffix probability polynomials give each passenger's
    # conditional probability of being positive given the bag total.
    n,k=p.shape
    init=np.zeros((n,k+1))
    init[:,0]=1.
    prefix=[init]
    for j in range(k):
        prefix.append(convolve(prefix[-1],p[:,j]))
    ix=np.arange(n)
    probability=prefix[-1][ix,total].clip(1e-250,None)
    suffix=init
    posterior=np.zeros_like(p)
    for j in reversed(range(k)):
        degrees=np.arange(j+1)
        remainder=total[:,None]-1-degrees
        valid=(remainder>=0)&(remainder<=k-j-1)
        complement=(prefix[j][:,:j+1]*suffix[ix[:,None],remainder.clip(0,k)]*valid).sum(1)
        posterior[:,j]=p[:,j]*complement/probability
        suffix=convolve(suffix,p[:,j])
    return -np.log(probability).sum(),p-posterior


def fit(x,z,y,blocks,offset,offset_q):
    ridge=np.full(x.shape[1],10.)
    ridge[0]=1.
    def objective(beta):
        eta=offset+x@beta
        p=expit(eta)
        if blocks is None:
            loss=np.sum(np.logaddexp(0,eta)-y*eta)
            grad=p-y
        else:
            loss=0.
            grad=np.zeros(len(y))
            for ids,total in blocks:
                value,change=observed_counts(p[ids],total)
                loss+=value/REPEATS
                grad+=np.bincount(ids.ravel(),weights=change.ravel(),minlength=len(y))/REPEATS
        return loss+.5*ridge@(beta**2),x.T@grad+ridge*beta
    m=minimize(objective,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",options=dict(maxiter=450,ftol=2e-9))
    if not m.success:
        raise RuntimeError(m.message)
    return expit(offset_q+z@m.x).clip(.001,.999)


def run():
    row,y,frame,g,_=load()
    orig,target,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,target)
    OUT.mkdir(exist_ok=True)
    records,history={},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{name}.npz") as z:
            for k,v in (("row",row),("y",y)):
                np.testing.assert_array_equal(z[k],v)
            fold,base,ref=z["fold"],z["base"],z["ref"]
        bank={}
        sizes=[]
        for k in np.unique(fold):
            q=np.flatnonzero(fold==k)
            t=np.flatnonzero((fold!=k)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            x,z=basis(frame,t,q)
            coordinates,_=design(frame,t)
            coord=coordinates[t]
            coord=(coord-coord.mean(0))/coord.std(0).clip(.3)
            pt,_=teachers(frame,y,g,t,q,115+int(k))
            variants={"individual":None,"similar":counts(coord,g[t],y[t],False),"random":counts(coord,g[t],y[t],True)}
            sizes.append({s:{str(ids.shape[1]):len(ids) for ids,_ in b} for s,b in variants.items() if b is not None})
            for method,blocks in variants.items():
                for prefix,a,b in (("native",np.zeros(len(t)),np.zeros(len(q))),
                                   ("update",logit(pt.clip(.002,.998)),logit(base[q].clip(.002,.998)))):
                    p=fit(x,z,y[t].astype(float),blocks,a,b)
                    bank.setdefault(prefix+"_"+method,np.full(len(y),np.nan))[q]=p
            print(name,"fold",k,flush=True)
        summary,hits=evaluate(y,base,ref,bank,fold,name=="v131",masks)
        records[name]=dict(comparison=summary,bag_sizes=sizes)
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**bank)
        if name!="v131":
            for s,v in hits.items():
                history.setdefault(s,[]).append(v)
        write(REPORT/"results.json",dict(plans=records))
    stable={}
    for s,trials in history.items():
        hit=np.logical_and.reduce([h for h,_ in trials])
        stable[s]={n:int((hit&m).sum()) for n,m in masks.items()}
        stable[s]["paired_gain_ci95_pp"]=interval(np.mean([d for _,d in trials],0),g)
    write(REPORT/"results.json",dict(plans=records,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
