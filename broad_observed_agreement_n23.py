"""Predict whether outcomes agree with a separate observed passenger attribute."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_companion_information_n23 import ART,REPORT,SEEDS,historical,load,strata,evaluate,write,calibrated
from broad_geographic_transfer_n23 import BILLS
from exaone_validation_data import CATEGORICAL

OUT=ART/"nov23_observed_agreement"


def witnesses(frame):
    bills=frame[BILLS].fillna(0)
    total=bills.sum(axis=1)
    retail=bills.FoodCourt+bills.ShoppingMall
    unordered=np.sort(bills.to_numpy(),axis=1)[:,::-1]
    for key,bit,drop in (("sleep",frame.CryoSleep=="True",["CryoSleep"]),
                         ("earth",frame.HomePlanet=="Earth",["HomePlanet"]),
                         ("side",frame.Side=="S",["Side"]),
                         ("pay",total>0,BILLS),
                         ("retail",retail>total-retail,BILLS)):
        raw=frame.drop(columns=drop).copy()
        for name in set(raw.columns)&set(BILLS):
            raw[name]=np.log1p(raw[name])
        x=pd.get_dummies(raw,columns=[k for k in CATEGORICAL if k in raw]).fillna(-2.).to_numpy(float)
        if key=="retail":
            x=np.column_stack([x,np.log1p(unordered),frame[BILLS].isna().sum(axis=1)])
        flag=bit.to_numpy(bool)
        assert .03<flag.mean()<.97,key
        yield key,flag,x


def learner():
    return HistGradientBoostingClassifier(max_iter=140,learning_rate=.065,max_leaf_nodes=15,
                    l2_regularization=3,min_samples_leaf=22,early_stopping=False,random_state=492)


def reconstruct(py,pz,agree,z):
    joint=(py+pz+agree-1)/2
    # Estimate the three binary moments independently, then project the joint
    # cell back into the interval consistent with both fitted marginals.
    lower=np.maximum(0,py+pz-1)
    upper=np.minimum(py,pz)
    cell=np.clip(joint,lower,upper)
    prob=np.where(z,cell/pz,(py-cell)/(1-pz))
    return prob,(joint<lower)|(joint>upper)


def one_view(x,z,y,t,q):
    def predict(target):
        return learner().fit(x[t],target[t]).predict_proba(x[q])[:,1].clip(.002,.998)
    py,pz,pa=[predict(label) for label in (y,z,y==z)]
    pure=np.where(z[q],pa,1-pa)
    conditional,clipped=reconstruct(py,pz,pa,z[q])
    design=np.column_stack([x,z])
    direct=learner().fit(design[t],y[t]).predict_proba(design[q])[:,1]
    joint_label=2*z.astype(int)+y.astype(int)
    assert len(np.unique(joint_label[t]))==4
    multi=learner().fit(x[t],joint_label[t]).predict_proba(x[q]).reshape(-1,2,2)
    selected=multi[np.arange(len(q)),z[q].astype(int)]
    exact=selected[:,1]/selected.sum(1)
    return dict(masked=py,agreement=pure,moments=conditional,joint=exact,direct=direct),int(clipped.sum())


def main():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    scopes=dict(**masks,repeat_paid=masks["repeat"]&(strata(frame)==2)&y,
                repeat_zero=masks["repeat"]&(strata(frame)==0)&~y,repeat_solo=masks["repeat"]&(frame.BookingSize==1).to_numpy())
    cases=list(witnesses(frame))
    OUT.mkdir(exist_ok=True)
    plans,history,unions={}, {}, {}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_companion_information/{name}.npz") as z:
            for k,v in (("row",row),("y",y)):
                np.testing.assert_array_equal(z[k],v)
            fold,base,ref=z["fold"],z["base"],z["ref"]
        scores={}
        clamps={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            for key,bit,x in cases:
                pred,n=one_view(x,bit,y,t,q)
                clamps[key]=clamps.get(key,0)+n
                for mode,p in pred.items():
                    scores.setdefault(key+"_"+mode,np.zeros(len(y)))[q]=p.clip(.001,.999)
        mode_names=("masked","direct","agreement","moments","joint")
        for mode in (*mode_names,"all"):
            chosen=[logit(scores[k+"_"+m]) for k,_,_ in cases for m in
                         (("direct","agreement","moments","joint") if mode=="all" else (mode,))]
            ds=np.column_stack([logit(base.clip(.001,.999)),*chosen])
            pred=np.full(len(y),np.nan)
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=706).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    pred[q]=calibrated(ds[t],ds[q],y,t)
            scores["update_"+mode]=pred
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",scopes)
        if name!="v131":
            for k,(hit,_) in hits.items():history.setdefault(k,[]).append(hit)
        plans[name]=dict(models=result,clipped_cells=clamps)
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**scores)
        print(name,[(k,d["fixed"],d["harmed"]) for k,d in result.items() if k.startswith("update")],flush=True)
    stable={k:np.logical_and.reduce(v) for k,v in history.items()}
    for mode in mode_names:
        union=np.logical_or.reduce([stable[k+"_"+mode] for k,_,_ in cases])
        direct=np.logical_or.reduce([stable[k+"_direct"] for k,_,_ in cases])
        unions[mode]={s:dict(total=int((union&m).sum()),exclusive_to_direct=int((union&~direct&m).sum()))
                      for s,m in scopes.items()}
    write(REPORT/"observed.json",dict(plans=plans,stable={k:{s:int((h&m).sum()) for s,m in scopes.items()}
                     for k,h in stable.items()},unions=unions))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
