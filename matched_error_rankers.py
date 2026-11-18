"""Rank potential errors against correct passengers who received comparable base predictions."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit,StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import error_trained_experts as experts
from scipy.special import expit
from peer_bill_correction import load_v96

ROOT=Path(__file__).resolve().parent
ref=experts.ref
NAMES=["bills","passenger","matched_party"]


def pairs(idx,base,y,frame,groups,seed):
    wrong=((base>=.5)!=y)
    direction=base>=.5
    confidence=np.maximum(base,1-base)
    buckets=np.minimum((confidence*10).astype(int),9)
    planet=frame.HomePlanet.iloc[idx].to_numpy()
    rng=np.random.default_rng(seed)
    all_rows=[];labels=[];pair_ids=[]
    for i in np.flatnonzero(wrong):
        controls=(~wrong)&(groups[idx]!=groups[idx[i]])&(direction==direction[i])
        exact=controls&(buckets==buckets[i])&(planet==planet[i])
        if exact.sum()<3:
            exact=controls&(buckets==buckets[i])
        possible=np.flatnonzero(exact)
        if not len(possible):continue
        neg=rng.choice(possible,min(3,len(possible)),replace=False)
        all_rows.extend([i,*neg]);labels.extend([1,*np.zeros(len(neg),dtype=int)])
        pair_ids.extend([i]*(len(neg)+1))
    return np.asarray(all_rows),np.asarray(labels),np.asarray(pair_ids)


def frame_with_direction(frame,idx,base):
    return frame.iloc[idx].assign(predicted_positive=(base>=.5).astype(float))


def meta(base,scores=None):
    confidence=np.maximum(base,1-base)
    direction=(base>=.5).astype(float)
    prior=ref.logit(1-confidence)
    components=[prior,direction,prior*direction]
    if scores is not None:
        components.extend([scores,scores*direction[:,None]])
    return np.column_stack(components)


def evaluate_risk(score,base,y,hard,fold):
    pred=base>=.5
    result={}
    for threshold in [.35,.5,.65]:
        flip=score>=threshold
        result[f"threshold_{threshold}"]=experts.summarize(y,pred^flip,pred,hard,fold)
    for fraction in [.01,.05,.10]:
        flip=np.zeros(len(y),bool)
        for part in range(3):
            ids=np.flatnonzero(fold==part)
            order=ids[np.argsort(-score[ids],kind="stable")]
            flip[order[:max(1,int(np.ceil(fraction*len(ids))))]]=True
        result[f"capacity_{fraction}"]=experts.summarize(y,pred^flip,pred,hard,fold)
    return result


def run():
    data,splits,fold,bases,evidence=ref.overlap.load_authenticated()
    dev,y,raw=data["dev"],data["y"],bases["rawMitra"]
    frames,cats=experts.views(data);groups=np.asarray(data["groups"])
    v96=load_v96(dev,fold)
    with np.load(ROOT/"artifacts/peer_bill_offsets.npz") as z:
        np.testing.assert_array_equal(z["dev"],dev);proxy=expit(ref.logit(v96)+z["peer_bills"])
    baselines={"raw":raw,"v96":v96,"v108_proxy":proxy}
    exact=ROOT/"artifacts/v108_exact_oof/oof.npz"
    if exact.exists():
        with np.load(exact) as z:
            np.testing.assert_array_equal(dev,z["dev_ids"]);baselines["v108"]=z["v108"].copy()
    oof={name:{k:np.empty(len(y)) for k in ["confidence","joint",*NAMES]} for name in baselines}
    for outer,(a,b) in enumerate(splits):
        innerp=ref.load_inner(outer,a,b,dev,y,evidence)
        cal_scores=[];cal_base=[];cal_target=[]
        eval_scores={k:[] for k in baselines}
        packets=StratifiedKFold(3,shuffle=True,random_state=20261117+outer)
        for inner,(_,packet) in enumerate(packets.split(a,y[a])):
            seed=20261118+100*outer+inner
            loc_d,loc_c=next(GroupShuffleSplit(n_splits=1,test_size=.33,random_state=seed).split(packet,groups=groups[a[packet]]))
            d,c=packet[loc_d],packet[loc_c]
            rows,target,match=pairs(a[d],innerp[d],y[a[d]],frames["matched_party"],groups,seed)
            predictions={k:[] for k in baselines};cal=[]
            for j,name in enumerate(NAMES):
                x=frames[name];fit=CatBoostRanker(iterations=160,depth=3,learning_rate=.06,l2_leaf_reg=12,
                    loss_function="PairLogit",random_seed=seed+j,thread_count=2,verbose=False,allow_writing_files=False,
                    cat_features=[col for col in cats if col in x])
                # group_id consists of one error and up to three correct controls. Query outcomes are absent.
                fit.fit(frame_with_direction(x,a[d][rows],innerp[d][rows]),target,group_id=match)
                cal.append(fit.predict(frame_with_direction(x,a[c],innerp[c])))
                for k,base in baselines.items():
                    predictions[k].append(fit.predict(frame_with_direction(x,b,base[b])))
            cal_scores.append(np.column_stack(cal));cal_base.append(innerp[c]);cal_target.append((innerp[c]>=.5)!=y[a[c]])
            for k in baselines:eval_scores[k].append(np.column_stack(predictions[k]))
        cp,cs,ct=np.concatenate(cal_base),np.concatenate(cal_scores),np.concatenate(cal_target)
        for tag,cols in [("confidence",[]),("joint",[0,1,2]),*[(n,[j]) for j,n in enumerate(NAMES)]]:
            train=meta(cp,cs[:,cols] if cols else None)
            arbiter=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000)).fit(train,ct)
            for k,p in baselines.items():
                oof[k][tag][b]=np.mean([arbiter.predict_proba(meta(p[b],s[:,cols] if cols else None))[:,1] for s in eval_scores[k]],axis=0)
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    result={k:{tag:evaluate_risk(risk,p,y,hard,fold) for tag,risk in oof[k].items()} for k,p in baselines.items()}
    ref.write(ROOT/"reports/matched_error_rankers/result.json",result)
    np.savez_compressed(ROOT/"artifacts/matched_error_rankers.npz",dev=dev,fold=fold,
                        **{k+"_"+n:v for k,dd in oof.items() for n,v in dd.items()})
    for k in ["raw","v108_proxy"]:
        for n in ["confidence","joint"]:
            print(k,n,{tag:(z["repairs"],z["breaks"],z["hard546_new"]) for tag,z in result[k][n].items()})


if __name__=="__main__":
    run()
