"""Separate changes in ranking from the analytic odds shift caused by class weights."""
import numpy as np
from catboost import CatBoostClassifier, Pool
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_circuit_operating_points_n23 import align
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write
from broad_release_miss_coverage_n23 import summarize
from broad_twin_surfaces_n23 import scopes
from exaone_validation_data import CATEGORICAL
from released_correction_audit_n23 import development

REPORT=ROOT/"reports/nov23_population_costs"
OUT=ART/"nov23_class_cost_path"


def cost_logits(frame,y,g,fold):
    ratios=(1/3,1.,3.)
    logits=np.empty((len(y),len(ratios)))
    for f in np.unique(fold):
        q=np.flatnonzero(fold==f)
        t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
        assert not np.intersect1d(g[t],g[q]).size
        query=Pool(frame.iloc[q],cat_features=CATEGORICAL)
        for j,r in enumerate(ratios):
            w=np.where(y[t],r,1.)
            w/=w.mean()
            model=CatBoostClassifier(iterations=420,depth=6,learning_rate=.065,thread_count=2,
                      verbose=False,random_seed=731,allow_writing_files=False)
            model.fit(Pool(frame.iloc[t],y[t],cat_features=CATEGORICAL,weight=w))
            logits[q,j]=model.predict(query,prediction_type="RawFormulaVal")
    return logits


def decisions(base,ref,fold,logits,y,g):
    z=logits-np.log([1/3,1,3])
    avg=z.mean(1)
    out={"ordinary":z[:,1]>=0,"negative_weight":logits[:,0]>=0,"positive_weight":logits[:,2]>=0,
         "average_direct":avg>=0}
    placement={}
    for j,k in [(0,"negative"),(2,"positive")]:
        native=logits[:,j]>=0
        threshold=align(z[:,1,None],fold,native,381)[:,0]
        equal=align(z[:,j,None],fold,z[:,1]>=0,381)[:,0]
        out[k+"_threshold_control"]=threshold
        out[k+"_at_ordinary_count"]=equal
        placement[k]=dict(rank_correlation=float(spearmanr(z[:,1],z[:,j]).statistic),
            n_positive=int(native.sum()),ordinary_positive=int((z[:,1]>=0).sum()),
            threshold_disagreements=int((threshold!=native).sum()),
            direct_disagreements_at_equal_quota=int((equal!=(z[:,1]>=0)).sum()))
    eligible=ref==(base>=.5)
    main=np.c_[logit(base.clip(.001,.999)),z[:,1]]
    delta=z[:,[0,2]]-z[:,1,None]
    for f in np.unique(fold):
        ids=np.flatnonzero((fold==f)&eligible)
        for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
            t,q=ids[a],ids[b]
            assert not np.intersect1d(g[t],g[q]).size
            placebo=delta.copy()
            bins=np.digitize(main[:,0],np.quantile(main[t,0],np.arange(1,8)/8))*6
            bins+=np.digitize(main[:,1],np.quantile(main[t,1],np.arange(1,6)/6))
            for ix in (t,q):
                rng=np.random.default_rng(386)
                for c in np.unique(bins[ix]):
                    cell=ix[bins[ix]==c]
                    placebo[cell]=delta[rng.permutation(cell)]
            for name,x in (("base_tree",main),("separate_costs",np.c_[main,delta]),
                           ("shuffled",np.c_[main,placebo])):
                fit=LogisticRegression(C=.5,max_iter=500).fit(x[t],y[t])
                out.setdefault(name,ref.copy())[q]=fit.predict(x[q]).astype(bool)
    return out,placement


def main():
    row,y,frame,g,_=load()
    masks=scopes(y,frame)
    labels,release,_=development()
    np.testing.assert_array_equal(labels,y)
    masks.update(repeat_true=masks["repeat"]&y,repeat_false=masks["repeat"]&~y)
    OUT.mkdir(exist_ok=True)
    history,refs,plans,placements={},[],{},{}
    for seed in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{seed}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            base,fold,ref=z["base"],z["fold"],z["ref"]
        np.testing.assert_array_equal(ref,release[131] if seed=="v131" else base>=.5)
        logits=cost_logits(frame,y,g,fold)
        out,placements[seed]=decisions(base,ref,fold,logits,y,g)
        reference=ref==y
        if seed!="v131":refs.append(reference)
        stats={}
        for k,v in out.items():
            h=v==y
            stats[k]=dict(accuracy=float(h.mean()),fixed=int((h&~reference).sum()),
                          harmed=int((~h&reference).sum()),correct_repeat=int((h&masks["repeat"]).sum()))
            if seed!="v131":
                history.setdefault(k,[]).append(h)
        np.savez_compressed(OUT/f"{seed}.npz",row=row,y=y,base=base,fold=fold,ref=ref,logits=logits,**out)
        plans[seed]=stats
        print(seed,[(k,stats[k]["fixed"],stats[k]["harmed"]) for k in ("ordinary","average_direct","separate_costs")],flush=True)
    result={k:summarize(np.stack(v),np.stack(refs),masks) for k,v in history.items()}
    write(REPORT/"costs.json",dict(plans=plans,grouped=result,placement=placements))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
