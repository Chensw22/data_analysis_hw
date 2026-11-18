"""Train specialists on base-model errors, then fit a separate arbiter on untouched parties."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit,StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import relational_error_corrector as ref
import teacher_distillation as teachers
from peer_bill_correction import BILLS,context,load_v96

ROOT=Path(__file__).resolve().parent
EXPERTS=["bills","passenger","two_teacher_errors","matched_party"]
REPORT=ROOT/"reports/error_trained_experts"


def views(inputs):
    dev=inputs["dev"]
    x=teachers.residual.features().iloc[dev].reset_index(drop=True)
    cat=[c for c in teachers.residual.CATS if c in x]
    money=[*BILLS,*[c for c in x if "Spend" in c or c.endswith("_missing")]]
    money=list(dict.fromkeys(["Age","CryoSleep",*money]))
    personal=["HomePlanet","CryoSleep","Destination","Age","VIP","Deck","CabinNumber","Side"]
    ids,_,_,party=context()
    np.testing.assert_array_equal(ids.iloc[dev],inputs["passengers"])
    enriched=pd.concat([x,pd.DataFrame(party[dev],columns=[f"peer_{i}" for i in range(party.shape[1])])],axis=1)
    return {"bills":x[money],"passenger":x[personal],"two_teacher_errors":x,"matched_party":enriched},cat


def select_correct_controls(idx,raw,y,frame,seed):
    wrong=(raw>=.5)!=y
    selected=wrong.copy()
    strata=pd.DataFrame(dict(sign=raw>=.5,planet=frame.HomePlanet.iloc[idx].to_numpy(),
                             sleep=frame.CryoSleep.iloc[idx].to_numpy()))
    rng=np.random.default_rng(seed)
    for rows in strata.groupby(list(strata.columns)).indices.values():
        controls=rows[~wrong[rows]]
        count=min(len(controls),int(wrong[rows].sum()))
        if count:
            selected[rng.choice(controls,count,replace=False)]=True
    return selected


def fit_specialists(frames,cat,idx,raw,kumo,y,queries,seed):
    wrong=(raw>=.5)!=y
    masks={"bills":wrong,"passenger":wrong,"two_teacher_errors":wrong&((kumo>=.5)!=y),
           "matched_party":select_correct_controls(idx,raw,y,frames["matched_party"],seed)}
    pred=np.empty((len(queries),len(EXPERTS)))
    for j,name in enumerate(EXPERTS):
        frame=frames[name];mask=masks[name]
        if len(np.unique(y[mask]))<2:
            raise ValueError("Insufficient error classes")
        model=CatBoostClassifier(iterations=180,depth=3,learning_rate=.05,l2_leaf_reg=10,
                random_seed=seed+j,thread_count=2,verbose=False,allow_writing_files=False,
                cat_features=[c for c in cat if c in frame])
        model.fit(frame.iloc[idx[mask]],y[mask])
        pred[:,j]=model.predict_proba(frame.iloc[queries])[:,1]
    return pred,{k:int(v.sum()) for k,v in masks.items()}


def routing_features(base,experts,people):
    pos=base>=.5
    flip_sign=np.where(pos,-1.,1.)
    challenge=logit(np.clip(experts,.001,.999))*flip_sign[:,None]
    strength=np.abs(ref.logit(base))
    sleep=people.CryoSleep.eq("True").to_numpy(float)
    earth=people.HomePlanet.eq("Earth").to_numpy(float)
    return np.column_stack([strength,pos,challenge,challenge*strength[:,None],sleep,earth,sleep*earth,
                            (challenge>0).sum(axis=1),np.log1p(people[BILLS].sum(axis=1).to_numpy())])


def summarize(y,p,base,hard,fold):
    correct=p==y
    wrong=base!=y
    repaired=wrong&correct;broken=~wrong&~correct
    return dict(accuracy=float(correct.mean()),base_wrong=int(wrong.sum()),base_correct=int((~wrong).sum()),
                repairs=int(repaired.sum()),breaks=int(broken.sum()),net=int(repaired.sum()-broken.sum()),
                error_recall=float(repaired.sum()/wrong.sum()),correct_lost=float(broken.sum()/(~wrong).sum()),
                change_precision=float(repaired.sum()/max(1,repaired.sum()+broken.sum())),
                hard546_correct=int((correct&hard).sum()),hard546_new=int((repaired&hard).sum()),
                fold_net=[int((repaired&(fold==j)).sum()-(broken&(fold==j)).sum()) for j in range(3)])


def score_reference(base,y,hard,fold,expert_prob,gate_risks):
    pred=base>=.5
    result={"experts":{},"joint":{}}
    repairs={}
    for j,specialist in enumerate(EXPERTS):
        q=expert_prob[:,j]>=.5
        result["experts"][specialist]=summarize(y,q,pred,hard,fold)
        repairs[specialist]=(q==y)&(pred!=y)
    all_repaired=np.column_stack(list(repairs.values()))
    for j,k in enumerate(EXPERTS):
        result["experts"][k]["unique_repairs"]=int((all_repaired[:,j]&(all_repaired.sum(axis=1)==1)).sum())
        result["experts"][k]["unique_hard"]=int((hard&all_repaired[:,j]&(all_repaired.sum(axis=1)==1)).sum())
    result["oracle_only"]=dict(any_expert_recovers=int(all_repaired.any(axis=1).sum()),
        any_expert_recovers_hard=int((all_repaired.any(axis=1)&hard).sum()),
        pairwise_intersection={i:{j:int((repairs[i]&repairs[j]).sum()) for j in EXPERTS} for i in EXPERTS})
    divergent=np.any((expert_prob>=.5)!=pred[:,None],axis=1)
    for t in [.35,.5,.65]:
        change=divergent&(gate_risks>=t)
        q=pred^change
        result["joint"][str(t)]=summarize(y,q,pred,hard,fold)
    # Direct majority of specialist outputs is a separate, deployable control.
    votes=np.sum((expert_prob>=.5)!=pred[:,None],axis=1)
    result["joint"]["three_of_four"]=summarize(y,pred^(votes>=3),pred,hard,fold)
    return result


def run():
    inp,splits,fold,bases,evidence=ref.overlap.load_authenticated()
    dev,y,raw=inp["dev"],inp["y"],bases["rawMitra"]
    groups=np.asarray(inp["groups"])
    frames,cat=views(inp)
    v96=load_v96(dev,fold)
    with np.load(ROOT/"artifacts/peer_bill_offsets.npz") as z:
        np.testing.assert_array_equal(dev,z["dev"])
        from scipy.special import expit
        proxy=expit(ref.logit(v96)+z["peer_bills"])
    baselines={"raw":raw,"v96":v96,"v108_proxy":proxy}
    exact=ROOT/"artifacts/v108_exact_oof/oof.npz"
    if exact.exists():
        record=json.loads((ROOT/"reports/v108_exact_oof/result.json").read_text())
        assert ref.sha(exact)==record["oof_sha256"]
        with np.load(exact) as z:
            np.testing.assert_array_equal(z["dev_ids"],dev)
            np.testing.assert_array_equal(z["fold_id"],fold)
            baselines["v108"]=z["v108"].copy()
    expert_oof=np.empty((len(dev),len(EXPERTS)))
    members=np.empty((len(dev),3,len(EXPERTS)))
    risks={k:np.empty(len(dev)) for k in baselines}
    splits_record=[]
    for outer,(a,b) in enumerate(splits):
        inside_raw=ref.load_inner(outer,a,b,dev,y,evidence)
        inside_kumo=teachers.kumo_inner(outer,a,b,dev,y,evidence)
        training=[];targets=[];outer_preds=[];records=[]
        parts=StratifiedKFold(3,shuffle=True,random_state=20261117+outer)
        for inner,(_,packet) in enumerate(parts.split(a,y[a])):
            seed=20261118+100*outer+inner
            local_d,local_c=next(GroupShuffleSplit(n_splits=1,test_size=.33,random_state=seed).split(packet,groups=groups[a[packet]]))
            d,c=packet[local_d],packet[local_c]
            assert not np.intersect1d(groups[a[d]],groups[a[c]]).size
            # Both discovery and calibration are in one teacher query packet. No label from either
            # subset was in that teacher's support, and calibration outcomes are never fit by a specialist.
            pc,counts=fit_specialists(frames,cat,a[d],inside_raw[d],inside_kumo[d],y[a[d]],
                                      np.r_[a[c],b],seed)
            cal=pc[:len(c)];outer_preds.append(pc[len(c):])
            args=routing_features(inside_raw[c],cal,frames["two_teacher_errors"].iloc[a[c]])
            challenged=np.any((cal>=.5)!=(inside_raw[c,None]>=.5),axis=1)
            training.append(args[challenged])
            targets.append(((inside_raw[c]>=.5)!=y[a[c]])[challenged])
            records.append(dict(packet=inner,discovery=len(d),calibration=len(c),fitted=counts))
        gate=make_pipeline(StandardScaler(),LogisticRegression(C=.15,max_iter=1000))
        gate.fit(np.concatenate(training),np.concatenate(targets))
        members[b]=np.stack(outer_preds,axis=1)
        expert_oof[b]=members[b].mean(axis=1)
        for name,base in baselines.items():
            member_risk=[gate.predict_proba(routing_features(base[b],p,frames["two_teacher_errors"].iloc[b]))[:,1]
                         for p in outer_preds]
            risks[name][b]=np.mean(member_risk,axis=0)
        splits_record.append(records)
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(dev,catalog.row_id)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    results={name:score_reference(p,y,hard,fold,expert_oof,risks[name]) for name,p in baselines.items()}
    results["training"]=splits_record
    ref.write(REPORT/"result.json",results)
    np.savez_compressed(ROOT/"artifacts/error_trained_experts.npz",dev=dev,fold=fold,
                        experts=expert_oof,member_experts=members,**{k+"_risk":v for k,v in risks.items()})
    for name in ["raw","v108_proxy"]:
        print(name, {k:(v["repairs"],v["breaks"],v["hard546_correct"]) for k,v in results[name]["experts"].items()},
              results[name]["oracle_only"]["any_expert_recovers"],results[name]["joint"]["0.5"],flush=True)


if __name__=="__main__":
    run()
