"""Measure whether the direction and timing of boosting updates help identify incorrect decisions."""
import numpy as np
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import CATEGORICAL,load
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,historical,evaluate,interval,write

STAGES=(8,24,64,160,384,800)
REPORT=ROOT/"reports/nov23_broad_boosting_path"
OUT=ART/"nov23_broad_boosting_path"


def train_path(frame,y,t,q):
    paths=[]
    for seed in (1049,2347):
        clf=CatBoostClassifier(iterations=STAGES[-1],depth=6,learning_rate=.06,thread_count=2,
                               random_seed=seed,allow_writing_files=False,verbose=False)
        clf.fit(frame.iloc[t],y[t],cat_features=CATEGORICAL)
        paths.append(np.column_stack([clf.predict(frame.iloc[q],prediction_type="RawFormulaVal",ntree_end=s)
                                      for s in STAGES]))
    return np.stack(paths,axis=2)


def designs(base,paths):
    mean=paths.mean(2)
    final=mean[:,-1]
    anchor=logit(base.clip(.002,.998))
    # Keep the final score and final between-model dispersion in both readouts.
    endpoint=np.column_stack([anchor,final,paths[:,-1].std(1)])
    delta=np.diff(mean,axis=1)
    change=np.diff(mean>=0,axis=1).sum(1)
    path=np.column_stack([delta,abs(delta).sum(1),np.min(mean,axis=1),np.max(mean,axis=1),
                          change,paths[:,:-1].std(2)])
    return endpoint,path


def shuffled_path(end,path,ids,rng):
    # Shuffle complete histories within similar final and anchor logits.
    # Split fitting and evaluation bookings before constructing the placebo.
    code=np.digitize(end[ids,0],np.quantile(end[ids,0],[.25,.5,.75]))*4
    code+=np.digitize(end[ids,1],np.quantile(end[ids,1],[.25,.5,.75]))
    src=ids.copy()
    for value in np.unique(code):
        cells=np.flatnonzero(code==value)
        src[cells]=rng.permutation(ids[cells])
    return path[src]


def calibrated(a,b,y,t):
    scale=StandardScaler().fit(a)
    model=LogisticRegression(C=.15,max_iter=500).fit(scale.transform(a),y[t])
    return model.predict_proba(scale.transform(b))[:,1]


def run():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    OUT.mkdir(exist_ok=True)
    report,history,paths_all={},{},[]
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{name}.npz") as z:
            for k,v in (("row",row),("y",y)):
                np.testing.assert_array_equal(z[k],v)
            base,fold,ref=z["base"],z["fold"],z["ref"]
        paths=np.full((len(y),len(STAGES),2),np.nan)
        for k in np.unique(fold):
            q=np.flatnonzero(fold==k)
            t=np.flatnonzero((fold!=k)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            paths[q]=train_path(frame,y,t,q)
        assert np.isfinite(paths).all()
        scores={f"trees_{stage}":expit(paths[:,j].mean(1)) for j,stage in enumerate(STAGES)}
        end,features=designs(base,paths)
        for k in np.unique(fold):
            ids=np.flatnonzero(fold==k)
            for a,b in GroupKFold(3,shuffle=True,random_state=436).split(ids,groups=g[ids]):
                t,q=ids[a],ids[b]
                assert not np.intersect1d(g[t],g[q]).size
                rng=np.random.default_rng(692)
                fake_t,fake_q=[shuffled_path(end,features,part,rng) for part in (t,q)]
                for m,u,v in (("base",end[t,:1],end[q,:1]),("end",end[t],end[q]),
                              ("path",np.column_stack([end[t],features[t]]),np.column_stack([end[q],features[q]])),
                              ("shuffled",np.column_stack([end[t],fake_t]),np.column_stack([end[q],fake_q]))):
                    scores.setdefault("update_"+m,np.full(len(y),np.nan))[q]=calibrated(u,v,y,t)
        results,hits=evaluate(y,base,ref,scores,fold,name=="v131",masks)
        stage_hit=np.stack([hits[f"trees_{s}"][0] for s in STAGES])
        per_scope={}
        for s,mask in {**masks,"all":np.ones(len(y),bool)}.items():
            per_scope[s]=dict(early_right_late_wrong=int((mask&stage_hit[0]&~stage_hit[-1]).sum()),
                           early_wrong_late_right=int((mask&~stage_hit[0]&stage_hit[-1]).sum()),
                           at_least_one_right=int((mask&stage_hit.any(0)).sum()))
        if name!="v131":
            paths_all.append(stage_hit)
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,base=base,fold=fold,ref=ref,paths=paths,**scores)
        report[name]=dict(models=results,movement=per_scope)
        write(REPORT/"results.json",dict(plans=report,stages=STAGES))
        print(name,[(m,r["fixed"],r["harmed"]) for m,r in results.items() if m.startswith("update")],flush=True)
    stable={}
    for s,trials in history.items():
        good=np.logical_and.reduce([h for h,_ in trials])
        stable[s]={n:int((m&good).sum()) for n,m in masks.items()}
        stable[s]["gain_ci95_pp"]=interval(np.mean([d for _,d in trials],0),g)
    # An oracle can select one common stage per passenger across the three fits.
    oracle=np.stack(paths_all).all(0).any(0)
    write(REPORT/"results.json",dict(plans=report,stages=STAGES,stable=stable,
         stable_at_one_stage={s:int((oracle&m).sum()) for s,m in masks.items()}))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
