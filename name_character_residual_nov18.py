"""One supervised sparse Name-character residual, without SVD or selected hard cases."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer

import relational_error_corrector as relational
import error_overlap_diagnostic as overlap
import exception_error_detector_nov18 as previous

ROOT=Path(__file__).resolve().parent
PLAN=ROOT/'reports/error_targeted_20261118/name_protocol.json'
REPORT=PLAN.parent
OUT=ROOT/'artifacts/error_targeted_20261118/name_character_residual'
VECTOR=dict(analyzer='char_wb',ngram_range=(2,4),min_df=5,max_features=2048,sublinear_tf=True,norm='l2',lowercase=True)
OPTIONS=dict(maxiter=500,ftol=1e-12,gtol=1e-6)
sha,write=relational.sha,relational.write


def objective(theta,x,y,offset):
    w=theta[1:];z=offset+theta[0]+np.asarray(x@w).ravel()
    error=expit(z)-y
    loss=float(np.sum(np.logaddexp(0,z)-y*z)+.5*np.dot(w,w))
    gradient=np.r_[error.sum(),np.asarray(x.T@error).ravel()+w]
    return loss,gradient


def fit_offset(x,y,offset):
    y=np.asarray(y);offset=np.asarray(offset)
    if x.shape[0]!=len(y) or offset.shape!=y.shape or not np.isfinite(offset).all():raise ValueError('Misaligned finite offsets.')
    if not np.isin(y,[0,1]).all():raise ValueError('Binary training labels required.')
    result=minimize(objective,np.zeros(x.shape[1]+1),args=(x,y,offset),method='L-BFGS-B',jac=True,options=OPTIONS)
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError('Fixed sparse optimizer failed: '+str(result.message))
    return result.x,dict(success=True,iterations=int(result.nit),objective=float(result.fun),gradient_max=float(np.max(np.abs(result.jac))))


def infer(vectorizer,theta,names,raw):
    x=vectorizer.transform(names)
    return expit(relational.logit(raw)+theta[0]+np.asarray(x@theta[1:]).ravel()),x


def check():
    x=sparse.csr_matrix([[0.,1.],[1.,.3],[.2,0.]])
    y=np.array([0.,1.,0.]);offset=np.array([-.7,.4,1.2]);theta=np.array([.3,-.2,.6])
    _,analytic=objective(theta,x,y,offset);numeric=[]
    for i in range(len(theta)):
        step=np.zeros_like(theta);step[i]=1e-6
        numeric.append((objective(theta+step,x,y,offset)[0]-objective(theta-step,x,y,offset)[0])/2e-6)
    np.testing.assert_allclose(analytic,numeric,rtol=1e-6,atol=1e-8)
    vectorizer=TfidfVectorizer(**VECTOR).fit(['Alpha Person']*5+['Beta Name']*5)
    weights=np.zeros(len(vectorizer.get_feature_names_out())+1);weights[0]=.2
    raw=np.array([.3,.7]);p,empty=infer(vectorizer,weights,['',''],raw)
    assert empty.nnz==0
    np.testing.assert_array_equal(p,expit(relational.logit(raw)+.2))
    # Constant-only is exactly the same objective with zero columns and no penalty.
    value,grad=objective(np.array([.2]),sparse.csr_matrix((3,0)),y,offset)
    assert abs(value-np.sum(np.logaddexp(0,offset+.2)-y*(offset+.2)))<1e-12
    assert abs(grad[0]-np.sum(expit(offset+.2)-y))<1e-12
    print('Synthetic sparse finite-difference/empty-name/constant-objective checks passed; no model optimizer run.')


def protocol():
    plan=json.loads(PLAN.read_text())
    actual={**VECTOR,'ngram_range':list(VECTOR['ngram_range'])}
    if plan['vectorizer']!=actual or plan['optimizer_options']!=OPTIONS:raise RuntimeError('Frozen recipe changed.')
    for name,digest in {**plan['source_hashes'],**plan['historical_evidence_sha256']}.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Frozen input changed: '+name)
    return plan


def summary(p,reference,y,mask=None):
    mask=np.ones(len(y),bool) if mask is None else mask
    return dict(**relational.pair(p,reference,y,mask),candidate_correct=int(((p[mask]>=.5)==y[mask]).sum()))


def run():
    if OUT.exists() or (REPORT/'name_result.json').exists() or (REPORT/'name_failure.json').exists():
        raise FileExistsError('One new fixed experiment; no overwrite/resume.')
    plan=protocol();sources=[*plan['source_hashes'],str(PLAN.relative_to(ROOT))]
    for name in sources:
        data=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
        if hashlib.sha256(data).hexdigest()!=sha(ROOT/name):raise RuntimeError('Commit source/protocol first: '+name)
    OUT.mkdir(parents=True);started=time.monotonic()
    def timeout(*_):raise TimeoutError('Fixed CPU120s deadline exceeded.')
    oldhandler=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,120)
    try:
        inputs,splits,fold_ids,bases,evidence=overlap.load_authenticated()
        dev,y=inputs['dev'],inputs['y'];raw=bases['rawMitra'];groups=inputs['groups']
        attrs=pd.read_csv(ROOT/'data/raw/train.csv',usecols=['PassengerId','Name'],dtype={'PassengerId':str})
        np.testing.assert_array_equal(attrs.PassengerId.iloc[dev].to_numpy(),inputs['passengers'])
        names=attrs.Name.iloc[dev].fillna('').astype(str).to_numpy()
        frozen=relational.nested.read(ROOT/'artifacts/planet_mixture_20261117/freeze.json')
        relational.nested.verify_hashes(frozen['sources']);evidence.update(frozen['sources'])
        evidence.update(plan['historical_evidence_sha256']);evidence.update(plan['source_hashes'])
        p=np.full(len(dev),np.nan);constant=p.copy();seen_group=np.zeros(len(dev),bool);empty_features=np.zeros(len(dev),bool)
        folds=[]
        for outer,(a,b) in enumerate(splits):
            inner=relational.load_inner(outer,a,b,dev,y,evidence)
            vectorizer=TfidfVectorizer(**VECTOR)
            train=vectorizer.fit_transform(names[a]);offset=relational.logit(inner)
            theta,fit=fit_offset(train,y[a],offset)
            intercept,controlfit=fit_offset(sparse.csr_matrix((len(a),0)),y[a],offset)
            query_p,query=infer(vectorizer,theta,names[b],raw[b])
            constant[b]=expit(relational.logit(raw[b])+intercept[0]);p[b]=query_p
            seen_group[b]=np.isin(groups[b],groups[a]);empty_features[b]=query.getnnz(axis=1)==0
            modelpath=OUT/f'fold{outer}_model.joblib'
            joblib.dump(dict(vectorizer=vectorizer,theta=theta,intercept_control=intercept,columns='Name only',epsilon=relational.EPS),modelpath)
            saved=joblib.load(modelpath)
            np.testing.assert_array_equal(query_p,infer(saved['vectorizer'],saved['theta'],names[b],raw[b])[0])
            np.testing.assert_array_equal(constant[b],expit(relational.logit(raw[b])+saved['intercept_control'][0]))
            np.savez_compressed(OUT/f'fold{outer}_rows.npz',train_ids=dev[a],query_ids=dev[b],inner_raw=inner,prob=query_p,constant=constant[b])
            row=dict(fold=outer,training_rows=len(a),query_rows=len(b),feature_count=train.shape[1],fit=fit,constant_fit=controlfit,
                model_sha256=sha(modelpath),rows_sha256=sha(OUT/f'fold{outer}_rows.npz'),reload_max_abs_difference=0.,
                query_empty_name_rows=int((names[b]=='').sum()),query_empty_feature_rows=int(empty_features[b].sum()),
                query_empty_name_fraction=float(np.mean(names[b]=='')),query_empty_feature_fraction=float(empty_features[b].mean()),
                metrics={k:relational.score(v,y[b]) for k,v in [('raw',raw[b]),('constant',constant[b]),('name',p[b])]},
                paired={k:summary(p[b],v,y[b]) for k,v in [('raw',raw[b]),('constant',constant[b])]})
            folds.append(row);write(REPORT/f'name_fold{outer}.json',row)
            print(f'Completed Name residual fold {outer+1}/3',flush=True)
        if len(folds)!=3 or not np.isfinite(p).all() or not np.isfinite(constant).all():raise RuntimeError('Incomplete folds.')
        # Historical error masks first loaded after all three optimizations/predictions/reloads.
        protocol();masks=previous.reporting_masks(dev,fold_ids,y,inputs,bases,evidence)
        masks.update(query_group_seen_in_outertrain=seen_group,query_group_unseen_in_outertrain=~seen_group)
        frame=inputs['features'].iloc[dev]
        masks.update(Earth_sleep=(frame.HomePlanet.eq('Earth')&frame.CryoSleep.eq('True')).to_numpy(),
                     Earth_awake=(frame.HomePlanet.eq('Earth')&frame.CryoSleep.eq('False')).to_numpy())
        probabilities={'raw':raw,'constant':constant,'name':p}
        subsets={name:dict(rows=int(mask.sum()),paired={k:summary(p,v,y,mask) for k,v in [('raw',raw),('constant',constant)]},
            confusion={k:previous.confusion(v,y,mask) for k,v in probabilities.items()}) for name,mask in masks.items()}
        for row,(_,b) in zip(folds,splits):
            row['group_overlap_slices']={name:dict(rows=int(mask[b].sum()),paired={k:summary(p[b],v[b],y[b],mask[b]) for k,v in [('raw',raw),('constant',constant)]})
                for name,mask in [('seen',seen_group),('unseen',~seen_group)]}
            write(REPORT/f"name_fold{row['fold']}.json",row)
        np.savez_compressed(OUT/'oof.npz',dev_ids=dev,fold_id=fold_ids,passenger_ids=np.asarray(inputs['passengers'],dtype=str),
            **probabilities,**masks,empty_name=names=='',empty_features=empty_features)
        protocol()
        for name,digest in evidence.items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Evidence changed: '+name)
        result=dict(status='complete',protocol_sha256=sha(PLAN),folds=folds,metrics={k:relational.score(v,y) for k,v in probabilities.items()},
            paired={k:summary(p,v,y) for k,v in [('raw',raw),('constant',constant)]},subsets=subsets,
            oof_path=str((OUT/'oof.npz').relative_to(ROOT)),oof_sha256=sha(OUT/'oof.npz'),seconds=time.monotonic()-started,
            query_empty_name_fraction=float(np.mean(names=='')),query_empty_feature_fraction=float(empty_features.mean()),
            no_new_base_fits=True,no_old_holdout_labels=True,no_upload=True,limitations=plan['limitations'])
        write(REPORT/'name_evidence.json',evidence);write(REPORT/'name_result.json',result)
    except BaseException as error:
        if (OUT/'oof.npz').exists():(OUT/'oof.npz').rename(OUT/'oof.partial.npz')
        (REPORT/'name_result.json').unlink(missing_ok=True)
        write(REPORT/'name_failure.json',dict(status='failed_inconclusive',error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,oldhandler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();check() if args.check else run()
