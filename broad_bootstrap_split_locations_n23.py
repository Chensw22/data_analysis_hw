"""Estimate the uncertainty of CART split locations by reweighting entire bookings."""
import numpy as np
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import ART,ROOT,SEEDS,historical,load,strata,evaluate,write
from broad_boosting_path_n23 import calibrated
from broad_refit_dissent_n23 import features

OUT=ART/"nov23_bootstrap_split_locations"
REPORT=ROOT/"reports/nov23_bootstrap_split_locations"
DRAWS=10


def boundaries(tree,x,y,ids,g,rng):
    # Each bootstrap resamples relative group weights coherently across all nodes.
    _,gi=np.unique(g[ids],return_inverse=True)
    weights=rng.exponential(size=(gi.max()+1,DRAWS))[gi]
    nodes=tree.tree_
    orig=nodes.threshold
    boot=np.tile(orig[:,None],(1,DRAWS))
    placebo=boot.copy()
    sizes=[]
    def visit(node,rows):
        col=nodes.feature[node]
        if col<0:
            return
        ordered=rows[np.argsort(x[ids[rows],col],kind="stable")]
        vals=x[ids[ordered],col]
        cuts=np.flatnonzero((vals[1:]>vals[:-1])&(np.arange(1,len(vals))>=24)&
                            (len(vals)-np.arange(1,len(vals))>=24))
        if not cuts.size:
            raise ValueError("Original split should be admissible.")
        w=weights[ordered]
        left=np.cumsum(w,axis=0)[cuts]
        pos=np.cumsum(w*y[ids[ordered],None],axis=0)[cuts]
        tot=w.sum(0);npos=w.T@y[ids[ordered]]
        gain=pos**2/left+(npos-pos)**2/(tot-left)
        boundaries=(vals[cuts]+vals[cuts+1])/2.
        # When reflecting, use count ranks rather than distances in currency or cabin number.
        original=int(np.searchsorted(vals,orig[node],side="right"))-1
        win=np.argmax(gain,axis=0)
        boot[node]=boundaries[win]
        reflected=np.where(rng.random(DRAWS)<.5,cuts[win],2*original-cuts[win])
        nearest=np.abs(cuts[:,None]-reflected).argmin(axis=0)
        placebo[node]=boundaries[nearest]
        if len(cuts)>1:
            sizes.append(float(np.mean(np.abs(cuts[win]-original))/len(rows)))
        visit(nodes.children_left[node],ordered[:original+1])
        visit(nodes.children_right[node],ordered[original+1:])
    visit(0,np.arange(len(ids)))
    return boot,placebo,sizes


def route(tree,cut,x):
    nodes=tree.tree_
    loc=np.zeros((len(x),cut.shape[1]),int)
    draw=np.arange(cut.shape[1])
    ix=np.arange(len(x))[:,None]
    for _ in range(nodes.max_depth):
        col=nodes.feature[loc]
        left=x[ix,np.maximum(col,0)]<=cut[loc,draw]
        loc=np.where(col<0,loc,np.where(left,nodes.children_left[loc],nodes.children_right[loc]))
    assert (nodes.feature[loc]<0).all()
    return loc


def estimate(tree,cut,x,y,t,q):
    loc=route(tree,cut,x[np.r_[t,q]])
    results=[]
    # Weak empirical smoothing also defines the prediction for empty estimation leaves.
    for a,b in zip(loc[:len(t)].T,loc[len(t):].T):
        total=np.bincount(a,minlength=tree.tree_.node_count)
        pos=np.bincount(a,weights=y[t],minlength=tree.tree_.node_count)
        results.append((pos[b]+2*y[t].mean())/(total[b]+2))
    return np.column_stack(results)


def ensemble(x,y,g,t,q):
    outs={k:[] for k in ("hard","bootstrap","jitter")}
    uncertainty={k:[] for k in ("bootstrap","jitter")}
    movement=[]
    rng=np.random.default_rng(612)
    for seed in (319,713):
        for u,v in GroupKFold(3,shuffle=True,random_state=seed).split(t,groups=g[t]):
            grow,estimate_rows=t[u],t[v]
            assert not np.intersect1d(g[grow],g[estimate_rows]).size
            for _ in range(5):
                tree=DecisionTreeClassifier(max_depth=7,min_samples_leaf=24,max_features=.70,
                            random_state=int(rng.integers(2**31))).fit(x[grow],y[grow])
                boot,jitter,shift=boundaries(tree,x,y,grow,g,rng)
                movement.extend(shift)
                for k,cut in (("hard",tree.tree_.threshold[:,None]),("bootstrap",boot),("jitter",jitter)):
                    draws=estimate(tree,cut,x,y,estimate_rows,q)
                    outs[k].append(draws.mean(1))
                    if k in uncertainty: uncertainty[k].append(draws.var(1))
    return ({k:np.mean(v,axis=0) for k,v in outs.items()},
            {k:np.sqrt(np.mean(v,axis=0)) for k,v in uncertainty.items()},float(np.mean(movement)))


def risk_check(uncertainty,base,ref,y):
    # Check the population tail without fitting a threshold to the true labels.
    chosen=uncertainty>=np.quantile(uncertainty,.75)
    errors=ref!=y
    rng=np.random.default_rng(811)
    draws=np.zeros(4000,int)
    for direction in (False,True):
        ids=np.flatnonzero(ref==direction)
        for cell in np.array_split(ids[np.argsort(base[ids],kind="stable")],20):
            size=int(chosen[cell].sum());nwrong=int(errors[cell].sum())
            if size:
                draws+=rng.hypergeometric(nwrong,len(cell)-nwrong,size,len(draws))
    return dict(passengers=int(chosen.sum()),errors=int((chosen&errors).sum()),
                expected=float(draws.mean()),reference95=np.quantile(draws,[.025,.975]).tolist())


def run():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(orig,row);np.testing.assert_array_equal(labels,y)
    scope=dict(**masks,repeat_paid=masks["repeat"]&(strata(frame)==2)&y,
               repeat_solo=masks["repeat"]&(frame.BookingSize==1).to_numpy())
    x=features(frame)[0]
    OUT.mkdir(exist_ok=True)
    plans,hist={},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_companion_information/{name}.npz") as z:
            np.testing.assert_array_equal(z["row"],row);np.testing.assert_array_equal(z["y"],y)
            base,ref,fold=z["base"],z["ref"],z["fold"]
        scores={};spread={};movement=[]
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            pred,unc,shift=ensemble(x,y,g,t,q)
            movement.append(shift)
            for k,v in pred.items():scores.setdefault(k,np.zeros(len(y)))[q]=v
            for k,v in unc.items():spread.setdefault(k,np.zeros(len(y)))[q]=v
        for k in ("hard","bootstrap","jitter"):
            cols=[logit(v.clip(.001,.999)) for v in (base,scores["hard"])]
            if k in spread:cols.extend([logit(scores[k]),spread[k],spread[k]*(2*ref.astype(int)-1)])
            design=np.column_stack(cols)
            p=np.zeros(len(y))
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=551).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    p[q]=calibrated(design[t],design[q],y,t)
            scores["update_"+k]=p
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",masks=scope)
        plans[name]=dict(models=result,mean_rank_displacement=movement,
                         error_risk=risk_check(spread["bootstrap"],base,ref,y))
        if name!="v131":
            for k,(hit,_) in hits.items():hist.setdefault(k,[]).append(hit)
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,split_sd=spread["bootstrap"],**scores)
        print(name,[(k,round(100*v["accuracy"],3)) for k,v in result.items()],flush=True)
    stable={k:np.logical_and.reduce(v) for k,v in hist.items()}
    write(REPORT/"result.json",dict(plans=plans,stable={k:{s:int((m&v).sum()) for s,m in scope.items()}
          for k,v in stable.items()},exclusive={s:int((m&stable["bootstrap"]&~stable["hard"]&~stable["jitter"]).sum())
          for s,m in scope.items()}))


if __name__=="__main__":
    with threadpool_limits(2):run()
