"""Match both risk scores and test contrasts in the source of model evidence."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_attribution_transfer_n23 import REPORT
from broad_geographic_transfer_n23 import load,BILLS
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_release_miss_coverage_n23 import summarize
from broad_twin_surfaces_n23 import scopes


def match(scores,groups,cases,controls):
    if not len(cases) or not len(controls):
        return np.array([],int),np.array([],int)
    tree=cKDTree(scores[controls])
    lists=tree.query_ball_point(scores[cases],r=.02,p=np.inf)
    i=np.repeat(np.arange(len(cases)),[len(v) for v in lists])
    j=np.asarray([idx for v in lists for idx in v],int)
    ok=groups[cases[i]]!=groups[controls[j]]
    i,j=i[ok],j[ok]
    dist=np.linalg.norm(scores[cases[i]]-scores[controls[j]],axis=1)
    n=len(cases)
    # A private dummy for every row. Its cost exceeds all real edges' total
    # possible cost, so matching cardinality takes priority over distance.
    cost=np.r_[dist+1.e-7,np.full(n,n+1)]
    graph=coo_matrix((cost,(np.r_[i,np.arange(n)],np.r_[j,len(controls)+np.arange(n)])),
                     shape=(n,len(controls)+n)).tocsr()
    li,ri=min_weight_full_bipartite_matching(graph)
    keep=ri<len(controls)
    left,right=cases[li[keep]],controls[ri[keep]]
    assert len(np.unique(right))==len(right)
    assert np.all(np.abs(scores[left]-scores[right])<.02000001)
    return left,right


def pairs(ids,scores,g,cases,correct):
    # Do not pair records across a change in either model's decision direction.
    cell=(scores[:,0]>=.5).astype(int)*2+(scores[:,1]>=.5)
    result=[]
    for c in np.unique(cell[ids]):
        ix=ids[cell[ids]==c]
        result.append(match(scores,g,ix[cases[ix]],ix[correct[ix]]))
    return tuple(np.concatenate([r[j] for r in result]) for j in (0,1))


def held_model(ids,base,tree,phi,y,g):
    score=np.c_[base,tree]
    err=(base>=.5)!=y
    direction=np.where(base>=.5,1.,-1.)
    centered=phi-phi.mean(1,keepdims=True)
    x=np.c_[centered*direction[:,None], np.abs(centered)]
    b=np.c_[logit(np.maximum(base,1-base).clip(.5,.999)),logit(tree.clip(.001,.999))*direction]
    out={}
    for a,c in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
        t,q=ids[a],ids[c]
        assert not np.intersect1d(g[t],g[q]).size
        # Learn a contrast on half the training bookings, and its contribution
        # to absolute error risk on the other half. Swap halves and average.
        for u,v in GroupKFold(2,shuffle=True,random_state=347).split(t,groups=g[t]):
            learn,cal=t[u],t[v]
            assert not np.intersect1d(g[learn],g[cal]).size
            left,right=pairs(learn,score,g,err,~err)
            assert len(left)>0
            dx=x[left]-x[right]
            yy=np.repeat([1,0],len(left))
            sham=np.random.default_rng(245).choice([-1,1],size=len(dx))
            for name,contrast in (("matched",dx),("shuffled",dx*sham[:,None]),("risk",None)):
                z=b
                if contrast is not None:
                    fit=LogisticRegression(C=.4,fit_intercept=False,max_iter=400).fit(np.r_[contrast,-contrast],yy)
                    z=np.c_[z,fit.decision_function(x)]
                clf=LogisticRegression(C=.5,max_iter=400).fit(z[cal],err[cal])
                e=clf.predict_proba(z[q])[:,1]
                p=np.where(direction[q]>0,1-e,e)
                out.setdefault(name,np.zeros(len(y)))[q]+=.5*p
    return out


def main():
    row,y,frame,g,_=load()
    masks=scopes(y,frame)
    groups={"bills":BILLS,"sleep":["CryoSleep"],"cabin":["Deck","Side","CabinNumber"],
            "registration":["HomePlanet","Destination","VIP"],"age_party":["Age","BookingSize"]}
    refs,history,details=[],{},{}
    for seed in SEEDS:
        with np.load(ART/f"nov23_attribution_transfer/{seed}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            base,tree,phi,fold=z["base"],z["tree"],z["phi"],z["fold"]
        score=np.c_[base,tree]
        ref=(base>=.5)==y
        refs.append(ref)
        output={}
        left,right=[],[]
        for f in np.unique(fold):
            ids=np.flatnonzero(fold==f)
            u,v=pairs(ids,score,g,masks["repeat"],ref)
            left.extend(u);right.extend(v)
            for k,p in held_model(ids,base,tree,phi,y,g).items():
                output.setdefault(k,np.zeros(len(y)))[ids]=p[ids]
        left,right=np.asarray(left),np.asarray(right)
        assert len(np.unique(right))==len(right)
        attrib=np.column_stack([phi[:,frame.columns.get_indexer(cols)].sum(1) for cols in groups.values()])
        details[str(seed)]={}
        for sign in (False,True):
            m=(base[left]>=.5)==sign
            gap=(attrib[left[m]]-attrib[right[m]])*(1 if sign else -1)
            details[str(seed)][str(sign)]=dict(n=int(m.sum()),case_minus_control=gap.mean(0).tolist(),
                   mean_absolute_score_gap=np.abs(score[left[m]]-score[right[m]]).mean(0).tolist())
        for k,p in output.items():
            assert ((p>0)&(p<1)).all()
            history.setdefault(k,[]).append((p>=.5)==y)
        print(seed,details[str(seed)],flush=True)
    write(REPORT/"matched.json",dict(domains=list(groups),paired=details,models={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in history.items()}))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
