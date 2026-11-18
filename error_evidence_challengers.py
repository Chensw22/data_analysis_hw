"""Prospective case evidence plus independent challengers; per-packet honest arbitration."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','FALSE')
from pathlib import Path
import json
import time
import signal
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold,GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import iterate
import relational_error_corrector as r
import error_overlap_diagnostic as overlap
import mixed_type_forest_imputation as forest
import latent_class_em_probe as latent
import error_case_retrieval as retrieval
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/error_evidence_challengers'
REPORT=ROOT/'reports/error_evidence_challengers'
FAMILIES=['forest_cat','latent_em']


def gate_features(base,challengers,evidence=None):
    flip_sign=np.where(base>=.5,-1.,1.)
    margins=r.logit(challengers)*flip_sign[:,None]
    features=np.column_stack([np.abs(r.logit(base)),(base>=.5).astype(float),margins])
    return features if evidence is None else np.column_stack([features,r.logit(evidence)])


def fit_gate(features,base,challengers,y,seed):
    active=np.any((challengers>=.5)!=(base[:,None]>=.5),axis=1)
    errors=(base>=.5)!=y
    if active.sum()==0 or len(np.unique(errors[active]))<2:
        model=dict(constant=float((errors[active].sum()+1)/(active.sum()+2)))
        metadata=dict(training_rows=int(active.sum()),errors=int(errors[active].sum()),constant_fallback=model['constant'])
    else:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.15,max_iter=1000,random_state=seed))
        model.fit(features[active],errors[active].astype(int))
        lr=model.named_steps['logisticregression']
        if np.max(lr.n_iter_)>=1000:raise RuntimeError('Fixed arbiter did not converge')
        metadata=dict(training_rows=int(active.sum()),errors=int(errors[active].sum()),coefficients=lr.coef_.tolist(),intercept=lr.intercept_.tolist(),iterations=lr.n_iter_.tolist())
    return model,metadata


def risk(model,features):
    return np.full(len(features),model['constant']) if isinstance(model,dict) else model.predict_proba(features)[:,1]


def replay(base,members,evidence_by_direction,fold_ids,models):
    """No targets. Recompute exactly for another authenticated base, e.g. v108 OOF.

    members[n,3,2], evidence[n,3,direction(0/1),view(0/1)].
    Caller must authenticate reference IDs/order before passing another base.
    """
    n=len(base)
    if members.shape!=(n,3,2) or evidence_by_direction.shape!=(n,3,2,2):raise ValueError('Aligned complete packets required')
    risks={k:np.zeros((n,3)) for k in ['margin','case']};active=np.any((members>=.5)!=(base[:,None,None]>=.5),axis=2)
    for outer in range(3):
        ids=np.flatnonzero(fold_ids==outer);direction=(base[ids]>=.5).astype(int)
        for j in range(3):
            ev=evidence_by_direction[ids,j,direction,:];ch=members[ids,j]
            for key in risks:
                z=gate_features(base[ids],ch,ev if key=='case' else None)
                risks[key][ids,j]=risk(models[(outer,j)][key],z)*active[ids,j]
    return {k:dict(member_risk=v,mean_risk=v.mean(1),flip=v.mean(1)>=.5) for k,v in risks.items()}


def fit_challengers(train_x,y,query_x,seed):
    """Only teacher-support labels and attributes reach either challenger fit."""
    chain=forest.ForestChain();filled=chain.fit_transform(forest.semantic(train_x),seed)
    repaired=forest.repaired_features(train_x,filled)
    cats=train_x.select_dtypes(include=['object','str','bool']).columns.tolist()
    cat=CatBoostClassifier(**forest.CATPARAMS,cat_features=cats,random_seed=seed);cat.fit(repaired,y)
    predictor=forest.Predictor(cat,chain);p1=predictor.predict_proba(query_x)[:,1]
    encoder=latent.Encoder().fit(train_x[latent.COLS]);z=encoder.transform(train_x[latent.COLS])
    initial=latent.initialize(z,encoder.cards,seed)
    model=latent.LatentClasses(encoder.cards).fit(z,y,initial,True);p2=model.predict(encoder.transform(query_x[latent.COLS]))
    return np.column_stack([p1,p2]),dict(forest=predictor,latent_encoder=encoder,latent=model)


def check():
    p=np.array([.8,.2]);challengers=np.array([[.3,.9],[.7,.1]])
    z=gate_features(p,challengers,np.full((2,2),.4))
    assert z.shape==(2,6) and (z[:,2]>0).all() and (z[:,3]<0).all()
    models={(i,j):{'margin':{'constant':.75},'case':{'constant':.75}} for i in range(3) for j in range(3)}
    members=np.tile(challengers[:,None,:],(1,3,1));out=replay(p,members,np.full((2,3,2,2),.4),np.array([0,2]),models)
    assert out['case']['flip'].all()
    aligned=np.tile(p[:,None,None],(1,3,2));out=replay(p,aligned,np.full((2,3,2,2),.4),np.array([0,2]),models)
    assert not out['case']['flip'].any() and (out['case']['mean_risk']==0).all()
    return dict(oriented_margin=True,no_challenger_disagreement_means_abstain=True,replay_has_no_y_argument=True)


def run():
    if OUT.exists() or REPORT.exists():raise FileExistsError('No overwrite or tuning')
    checks=check();data,splits,folds,bases,evidence=overlap.load_authenticated();dev,y=data['dev'],data['y'];raw=bases['rawMitra'];groups=np.asarray(data['groups'],dtype=str)
    frozen=r.nested.read(ROOT/'artifacts/planet_mixture_20261117/freeze.json');r.nested.verify_hashes(frozen['sources']);evidence.update(frozen['sources'])
    inner=[r.load_inner(i,a,b,dev,y,evidence) for i,(a,b) in enumerate(splits)]
    attrs=pd.read_csv(ROOT/'data/raw/train.csv',usecols=lambda c:c not in ['Name','Transported']);x,_=iterate.build_features(attrs,False)
    assert x.shape==(8693,30);np.testing.assert_array_equal(attrs.PassengerId.iloc[dev].to_numpy(),data['passengers'])
    names=[Path(__file__).name,'error_trained_experts.py','mixed_type_forest_imputation.py','latent_class_em_probe.py','error_case_retrieval.py','iterate.py','experiment.py','error_overlap_diagnostic.py','relational_error_corrector.py','artifacts/error_rescue_20261117/catalog_rows.csv']
    evidence.update({n:r.sha(ROOT/n) for n in names});OUT.mkdir(parents=True);REPORT.mkdir(parents=True)
    plan=dict(status='fixed_before_new_fits',teacher='Authenticated original raw inner3 packets; each support T disjoint from entirequery D+C. Originalouterquery never fit.',
        partition='Exactly error_trained_experts: innerSKF seed20261117+outer; packet seed20261118+100outer+inner; GroupShuffleSplit(test_size=.33) withinpacketquery gives D/C. No sharedGroup D↔C.',
        challengers=dict(forest_cat=dict(imputer='Original mixed_type_forest_imputation two-round50trees/depth10/leaf2; observed attributes only, trained onlyT',target=forest.CATPARAMS),latent_em='Original supervisedK24/80EM and original bins/smoothing, onlyT labels. No resampling/selection.'),
        cases='D labels plus corresponding teacher rawOOF define errors/correctcontrols; two original retrieval views,K32,samepredicted direction,exclude sameID/Group,library-only scaling. AllD retained, no hard546 mask.',
        gates='Separate for EACH packet: C challenged rows only, StandardScaler and LogisticRegression(C=.15,max_iter1000); control4features absbase logit/baseclass/two opposite-orientedchallengerlogits; experimental adds2 retrieval logits. No cross-packet calibration pool.',
        selection='Packet risk zero when neither challenger opposes base. Fixed arithmetic mean of3packet risks, flip iff>=.5; no weights/threshold search.',
        fallbacks='If challenged C has fewer than2 errorclasses, smoothed constant(errorcount+1)/(n+2), recorded. No expert discarded for low outerperformance.',
        replay='Save outerquery challenger probabilities[n,3,2], retrievalscores[n,3,2directions,2views], all18gate models, exactdev/Passenger/foldIDs. replay(newauthenticatedbase,...) recomputes margins/direction/active and .5decision without fitting or targets.',
        limitations=['Adaptive original6954dev; no independent holdout.','Outerraw uses4636support versus innerteacher3090; calibration margin distribution shift remains.','Challenges within same packet are independent of D/C targets; D→C and T→D/C boundaries fixed.','Teacher/challenger T may shareGroup with packetquery under originalrandomCV; only case neighbors and D/C have explicitGroup exclusion.','Oracle any-challenger correctness is diagnostic, not deployable selection.'],
        budget_seconds=1200,CPU_threads=4,no_GPU=True,threshold=.5)
    r.write(REPORT/'protocol.json',plan);evidence[str((REPORT/'protocol.json').relative_to(ROOT))]=r.sha(REPORT/'protocol.json');r.write(REPORT/'freeze.json',evidence);r.write(REPORT/'checks.json',checks)
    members=np.full((len(y),3,2),np.nan);evby=np.full((len(y),3,2,2),np.nan);models={};records=[]
    started=time.monotonic();old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU1200s challenger arbitration')));signal.setitimer(signal.ITIMER_REAL,1200)
    try:
        with threadpool_limits(limits=4):
            for outer,(a,b) in enumerate(splits):
                parts=StratifiedKFold(3,shuffle=True,random_state=20261117+outer)
                for j,(t,packet) in enumerate(parts.split(a,y[a])):
                    tick=time.monotonic();seed=20261118+100*outer+j
                    ld,lc=next(GroupShuffleSplit(n_splits=1,test_size=.33,random_state=seed).split(packet,groups=groups[a[packet]]));d,c=packet[ld],packet[lc]
                    assert not np.intersect1d(groups[a[d]],groups[a[c]]).size
                    assert not np.intersect1d(dev[a[t]],np.r_[dev[a[d]],dev[a[c]],dev[b]]).size
                    query_ids=np.r_[dev[a[c]],dev[b]]
                    cp,challenger_models=fit_challengers(x.loc[dev[a[t]]],y[a[t]].copy(),x.loc[query_ids],seed)
                    cal=cp[:len(c)];members[b,j]=cp[len(c):]
                    path=OUT/f'outer{outer}_packet{j}_challengers.joblib';joblib.dump(challenger_models,path)
                    restored=joblib.load(path)
                    np.testing.assert_array_equal(cp[:,0],restored['forest'].predict_proba(x.loc[query_ids])[:,1])
                    np.testing.assert_array_equal(cp[:,1],restored['latent'].predict(restored['latent_encoder'].transform(x.loc[query_ids,latent.COLS])))
                    challenger_sha=r.sha(path);del challenger_models,restored
                    cal_ev=np.empty((len(c),2));neighbor_evidence={}
                    for v,view in enumerate(['consumption','passenger_cabin']):
                        score,neigh,available,_=retrieval.retrieve(x.loc[dev[a[d]]],y[a[d]],inner[outer][d],dev[a[d]],groups[a[d]],x.loc[dev[a[c]]],inner[outer][c],dev[a[c]],groups[a[c]],view)
                        cal_ev[:,v]=score;neighbor_evidence[view+'_cal_neighbors']=neigh
                        for direction in [0,1]:
                            score,neigh,available,_=retrieval.retrieve(x.loc[dev[a[d]]],y[a[d]],inner[outer][d],dev[a[d]],groups[a[d]],x.loc[dev[b]],np.full(len(b),.25 if direction==0 else .75),dev[b],groups[b],view)
                            if (available<32).any():raise RuntimeError('Fewer than32 legal outercase neighbors')
                            evby[b,j,direction,v]=score;neighbor_evidence[view+f'_outer{direction}_neighbors']=neigh
                    fits={};models[(outer,j)]={}
                    for key in ['margin','case']:
                        z=gate_features(inner[outer][c],cal,cal_ev if key=='case' else None)
                        model,meta=fit_gate(z,inner[outer][c],cal,y[a[c]],seed)
                        models[(outer,j)][key]=model;fits[key]=meta
                    gatepath=OUT/f'outer{outer}_packet{j}_gates.joblib';joblib.dump(models[(outer,j)],gatepath);reloaded=joblib.load(gatepath)
                    for key in ['margin','case']:
                        z=gate_features(inner[outer][c],cal,cal_ev if key=='case' else None)
                        np.testing.assert_array_equal(risk(models[(outer,j)][key],z),risk(reloaded[key],z))
                    packetpath=OUT/f'outer{outer}_packet{j}.npz';np.savez_compressed(packetpath,teacher_support_ids=dev[a[t]],D_ids=dev[a[d]],C_ids=dev[a[c]],outer_query_ids=dev[b],C_base=inner[outer][c],D_base=inner[outer][d],C_challengers=cal,C_evidence=cal_ev,**neighbor_evidence)
                    record=dict(outer=outer,packet=j,seed=seed,support_rows=len(t),D_rows=len(d),C_rows=len(c),query_rows=len(b),teacher_support_sha256=r.nested.ids_hash(dev[a[t]]),D_sha256=r.nested.ids_hash(dev[a[d]]),C_sha256=r.nested.ids_hash(dev[a[c]]),outer_query_sha256=r.nested.ids_hash(dev[b]),challenger_model_sha256=challenger_sha,gate_model_sha256=r.sha(gatepath),packet_sha256=r.sha(packetpath),fits=fits,all_model_reload_max_diff=0.,seconds=time.monotonic()-tick)
                    records.append(record);r.write(REPORT/f'outer{outer}_packet{j}.json',record);print(json.dumps(dict(outer=outer,packet=j,seconds=record['seconds'],challenged_C=fits['case']['training_rows'])),flush=True)
        if not np.isfinite(members).all() or not np.isfinite(evby).all():raise RuntimeError('Incomplete packets')
        outputs=replay(raw,members,evby,folds,models)
        loaded={(i,j):joblib.load(OUT/f'outer{i}_packet{j}_gates.joblib') for i in range(3) for j in range(3)}
        duplicate=replay(raw,members,evby,folds,loaded)
        for key in outputs:
            for field in outputs[key]:np.testing.assert_array_equal(outputs[key][field],duplicate[key][field])
        rows=pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',dtype={'PassengerId':str})
        for key,w in [('row_id',dev),('y',y),('fold',folds),('PassengerId',data['passengers'])]:np.testing.assert_array_equal(rows[key],w)
        hard=rows.all_recipes_still_wrong.to_numpy(bool);assert hard.sum()==546
        base_label=raw>=.5;expert_prob=members.mean(1)
        labels={name:expert_prob[:,j]>=.5 for j,name in enumerate(FAMILIES)}
        labels.update({key:base_label^v['flip'] for key,v in outputs.items()});labels['raw']=base_label
        def summary(mask):
            out={}
            for name,pred in labels.items():
                correct=pred==y;was=base_label==y;repair=(correct&~was&mask).sum();broken=(~correct&was&mask).sum()
                out[name]=dict(rows=int(mask.sum()),baseline_errors=int((~was&mask).sum()),baseline_correct=int((was&mask).sum()),correct=int((correct&mask).sum()),repairs=int(repair),breaks=int(broken),net=int(repair-broken),accuracy=float(correct[mask].mean()) if mask.any() else None)
            return out
        masks=dict(all=np.ones(len(y),bool),hard546=hard,raw_false_positive=base_label&(y==0),raw_false_negative=~base_label&(y==1),raw_correct=base_label==y)
        wrong=base_label!=y;mean_rescue=((expert_prob>=.5)==y[:,None])&wrong[:,None];member_rescue=((members>=.5)==y[:,None,None])&wrong[:,None,None]
        increment=r.pair(labels['case'].astype(float),labels['margin'].astype(float),y)
        increment_hard=r.pair(labels['case'].astype(float),labels['margin'].astype(float),y,hard)
        cache=OUT/'oof.npz';np.savez_compressed(cache,dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(data['passengers'],dtype=str),raw=raw,members=members,evidence_by_direction=evby,expert_prob=expert_prob,**{key+'_'+field:value for key,data_ in outputs.items() for field,value in data_.items()})
        for name,digest in evidence.items():assert r.sha(ROOT/name)==digest,name
        result=dict(status='complete',subsets={key:summary(mask) for key,mask in masks.items()},case_increment_over_margin=increment,case_increment_hard546=increment_hard,
            oracle_not_deployable=dict(two_mean_challengers=int(mean_rescue.any(1).sum()),two_mean_hard546=int((mean_rescue.any(1)&hard).sum()),any_of_six_members=int(member_rescue.any((1,2)).sum()),any_member_hard546=int((member_rescue.any((1,2))&hard).sum())),
            fold_results=[dict(fold=i,all=summary(folds==i),hard546=summary(hard&(folds==i))) for i in range(3)],packets=records,seconds=time.monotonic()-started,
            oof_sha256=r.sha(cache),source_hashes=evidence,replay_API='replay(base_prob,members,evidence_by_direction,fold_ids,models); caller authenticates newbaseIDs, no targets or refit required',no_GPU=True,no_upload=True)
        r.write(REPORT/'result.json',result);print(json.dumps(dict(seconds=result['seconds'],all=result['subsets']['all'],hard=result['subsets']['hard546'],increment=increment)),flush=True)
    except BaseException as exc:r.write(REPORT/'failure.json',dict(type=type(exc).__name__,error=str(exc)));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
if __name__=='__main__':run()
