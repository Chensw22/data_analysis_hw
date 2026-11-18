"""Fixed hierarchical hurdle-dose GAM and equal rawMitra combination; no causal claim."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

if __name__=='__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import SplineTransformer

import error_overlap_diagnostic as overlap
import relational_error_corrector as helpers
import exception_error_detector_nov18 as reporting

ROOT=Path(__file__).resolve().parent
PLAN=ROOT/'reports/mechanism_combo/protocol.json'
REPORT=PLAN.parent
OUT=ROOT/'artifacts/mechanism_combo_empty_query_fix'
SPEND=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
CATS=['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
GLOBAL_L2=3.
STATE_L2=30.
OPTIONS=dict(maxiter=1000,ftol=1e-11,gtol=1e-5)
sha,write=helpers.sha,helpers.write


def states(x):return x.HomePlanet.astype(str).to_numpy()+'|'+x.CryoSleep.astype(str).to_numpy()


class Curve:
    def fit(self,values,reference):
        values=np.asarray(values,float);values=values[np.isfinite(values)]
        self.reference=float(reference);self.spline=None
        if len(values)>1 and np.ptp(values)>0:
            self.spline=SplineTransformer(n_knots=4,degree=2,knots='uniform',include_bias=False,extrapolation='linear').fit(values[:,None])
            self.origin=self.spline.transform([[self.reference]])
        self.width=0 if self.spline is None else self.origin.shape[1]
        return self

    def transform(self,values):
        values=np.asarray(values,float);valid=np.isfinite(values)
        out=np.zeros((len(values),self.width))
        if self.width and valid.any():out[valid]=self.spline.transform(values[valid,None])-self.origin
        return out


class Design:
    def fit(self,x):
        self.vocab={col:sorted(x[col].astype(str).unique()) for col in CATS}
        self.state_names=sorted(set(states(x)))
        self.dose={}
        for col in SPEND:
            values=x[col].to_numpy(float)
            if np.any(values[np.isfinite(values)]<0):raise ValueError('Nonnegative service spending required.')
            # Include structural zero as a boundary; positive-dose knot spacing is learned only on training values.
            self.dose[col]=Curve().fit(np.r_[0.,np.log1p(values[np.isfinite(values)&(values>0)])],0.)
        age=x.Age.to_numpy(float);self.age=Curve().fit(age,np.nanmedian(age))
        self.spatial={}
        keys=x.Deck.astype(str).to_numpy()+'|'+x.Side.astype(str).to_numpy();number=x.CabinNumber.to_numpy(float)
        for key in sorted(set(keys)):
            vals=number[(keys==key)&np.isfinite(number)]
            if len(vals)>1 and np.ptp(vals)>0:self.spatial[key]=Curve().fit(vals,float(np.median(vals)))
        self.transform(x,record=True)
        return self

    def transform(self,x,record=False):
        n=len(x);blocks=[];names=[];penalties=[]
        def add(array,labels,penalty):
            array=sparse.csr_matrix(array)
            if array.shape[1]!=len(labels):raise RuntimeError('Feature naming mismatch.')
            blocks.append(array);names.extend(labels);penalties.extend([penalty]*len(labels))
        add(np.ones((n,1)),['intercept'],0.)
        for col in CATS:
            values=x[col].astype(str).to_numpy()
            add(np.column_stack([values==v for v in self.vocab[col]]),[col+'='+v for v in self.vocab[col]],GLOBAL_L2)
        state=states(x)
        add(np.column_stack([state==v for v in self.state_names]),['state='+v for v in self.state_names],GLOBAL_L2)
        dose=[];dose_names=[]
        for col in SPEND:
            values=x[col].to_numpy(float);positive=np.isfinite(values)&(values>0);missing=~np.isfinite(values)
            curve=self.dose[col].transform(np.log1p(values))
            # Missing is its own hurdle, never mistaken for measured zero.
            dose.extend([positive[:,None],missing[:,None],curve])
            dose_names.extend([col+':positive',col+':missing']+[f'{col}:dose{j}' for j in range(curve.shape[1])])
        dose=np.column_stack(dose);add(dose,dose_names,GLOBAL_L2)
        for value in self.state_names:
            add(dose*(state==value)[:,None],[value+'*'+name for name in dose_names],STATE_L2)
        age=x.Age.to_numpy(float);age_basis=self.age.transform(age)
        add(np.column_stack([~np.isfinite(age),age_basis]),['Age:missing']+[f'Age:spline{j}' for j in range(age_basis.shape[1])],GLOBAL_L2)
        for value in self.state_names:
            add(age_basis*(state==value)[:,None],[value+f'*Age:spline{j}' for j in range(age_basis.shape[1])],STATE_L2)
        number=x.CabinNumber.to_numpy(float);keys=x.Deck.astype(str).to_numpy()+'|'+x.Side.astype(str).to_numpy()
        add((~np.isfinite(number))[:,None],['CabinNumber:missing'],GLOBAL_L2)
        for key,curve in self.spatial.items():
            values=curve.transform(np.where(keys==key,number,np.nan))
            add(values,[key+f':CabinNumber{s}' for s in range(values.shape[1])],GLOBAL_L2)
        result=sparse.hstack(blocks,format='csr',dtype=np.float64)
        if not np.isfinite(result.data).all():raise ValueError('Nonfinite design.')
        if record:self.names=names;self.penalties=np.asarray(penalties)
        elif names!=self.names:raise RuntimeError('Feature schema drift.')
        return result


def objective(theta,x,y,penalties):
    z=np.asarray(x@theta).ravel();r=expit(z)-y
    return float(np.sum(np.logaddexp(0,z)-y*z)+.5*np.sum(penalties*theta**2)),np.asarray(x.T@r).ravel()+penalties*theta


def fit_model(x,y):
    design=Design().fit(x);matrix=design.transform(x)
    initial=np.zeros(matrix.shape[1]);fraction=float(np.mean(y));initial[0]=np.log(fraction/(1-fraction))
    fit=minimize(objective,initial,args=(matrix,y,design.penalties),jac=True,method='L-BFGS-B',options=OPTIONS)
    if not fit.success or not np.isfinite(fit.x).all():raise RuntimeError('Fixed optimizer failed: '+str(fit.message))
    return design,fit.x,dict(success=True,iterations=int(fit.nit),objective=float(fit.fun),gradient_max=float(np.abs(fit.jac).max()),features=matrix.shape[1])


def check_protocol():
    plan=json.loads(PLAN.read_text())
    if plan['global_l2']!=GLOBAL_L2 or plan['state_l2']!=STATE_L2 or plan['optimizer_options']!=OPTIONS:raise RuntimeError('Recipe drift.')
    for name,digest in {**plan['source_hashes'],**plan['historical_hashes']}.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Source/evidence changed: '+name)
    return plan


def check():
    x=sparse.csr_matrix([[1.,.3,0],[1.,0.,.7],[1.,.2,-.1]])
    y=np.array([0.,1.,1.]);w=np.array([.1,.3,-.2]);pen=np.array([0.,3.,30.])
    _,grad=objective(w,x,y,pen);numeric=[]
    for j in range(3):
        d=np.zeros(3);d[j]=1e-6;numeric.append((objective(w+d,x,y,pen)[0]-objective(w-d,x,y,pen)[0])/2e-6)
    np.testing.assert_allclose(grad,numeric,rtol=1e-6,atol=1e-8)
    c=Curve().fit([0,1,2,3],0);assert not c.transform([0,np.nan]).any();assert not c.transform([np.nan,np.nan]).any()
    print('Sparse penalized gradient and zero/missing curve checks passed.')


def run():
    plan=check_protocol()
    if OUT.exists() or (REPORT/'result.json').exists():raise FileExistsError('Fixed new run only.')
    OUT.mkdir(parents=True);started=time.monotonic()
    def timeout(*_):raise TimeoutError('Fixed300second CPU budget exceeded.')
    previous=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,300)
    try:
        inputs,splits,fold_ids,bases,evidence=overlap.load_authenticated();dev,y=inputs['dev'],inputs['y'];raw=bases['rawMitra']
        features=inputs['features'];mechanism=np.full(len(dev),np.nan);combo=mechanism.copy();folds=[]
        for fold,(a,b) in enumerate(splits):
            train=features.iloc[dev[a]];query=features.iloc[dev[b]]
            design,theta,fit=fit_model(train,y[a])
            p=expit(np.asarray(design.transform(query)@theta).ravel());blend=.5*raw[b]+.5*p
            path=OUT/f'fold{fold}.joblib';joblib.dump(dict(design=design,theta=theta),path)
            loaded=joblib.load(path);reload=expit(np.asarray(loaded['design'].transform(query)@loaded['theta']).ravel())
            np.testing.assert_array_equal(p,reload)
            mechanism[b]=p;combo[b]=blend
            np.savez_compressed(OUT/f'fold{fold}_rows.npz',train_ids=dev[a],query_ids=dev[b])
            coefficients=dict(zip(design.names,theta.tolist()))
            write(REPORT/f'fold{fold}_coefficients.json',coefficients)
            row=dict(fold=fold,fit=fit,training_rows=len(a),query_rows=len(b),model_sha256=sha(path),reload_difference=0.,
                metrics={k:helpers.score(v,y[b]) for k,v in [('raw',raw[b]),('mechanism',p),('fixed_half',blend)]},
                paired={k:helpers.pair(v,raw[b],y[b]) for k,v in [('mechanism',p),('fixed_half',blend)]})
            folds.append(row);write(REPORT/f'fold{fold}.json',row);print(json.dumps(row),flush=True)
        if not np.isfinite(combo).all():raise RuntimeError('Incomplete OOF.')
        # All report-only outcome masks are first loaded after all predictions finish.
        check_protocol();masks=reporting.reporting_masks(dev,fold_ids,y,inputs,bases,evidence)
        frame=features.iloc[dev]
        masks.update(Earth_sleep=(frame.HomePlanet.eq('Earth')&frame.CryoSleep.eq('True')).to_numpy(),Earth_awake=(frame.HomePlanet.eq('Earth')&frame.CryoSleep.eq('False')).to_numpy())
        probabilities={'raw':raw,'mechanism':mechanism,'fixed_half':combo}
        subsets={name:dict(rows=int(mask.sum()),correct={k:int(((p[mask]>=.5)==y[mask]).sum()) for k,p in probabilities.items()},
            paired={k:helpers.pair(p,raw,y,mask) for k,p in [('mechanism',mechanism),('fixed_half',combo)]},
            confusion={k:reporting.confusion(p,y,mask) for k,p in probabilities.items()}) for name,mask in masks.items()}
        np.savez_compressed(OUT/'oof.npz',dev_ids=dev,fold_id=fold_ids,passenger_ids=np.asarray(inputs['passengers'],dtype=str),**probabilities,**masks)
        check_protocol()
        for n,d in evidence.items():
            if sha(ROOT/n)!=d:raise RuntimeError('Input evidence changed: '+n)
        result=dict(status='complete_adaptive_threefold_experiment',folds=folds,metrics={k:helpers.score(p,y) for k,p in probabilities.items()},
            paired={k:helpers.pair(p,raw,y) for k,p in [('mechanism',mechanism),('fixed_half',combo)]},subsets=subsets,
            oof_path=str((OUT/'oof.npz').relative_to(ROOT)),oof_sha256=sha(OUT/'oof.npz'),protocol_sha256=sha(PLAN),
            seconds=time.monotonic()-started,no_Name=True,no_new_base_fits=True,no_old_holdout_labels=True,no_upload=True,limitations=plan['limitations'])
        write(REPORT/'evidence.json',evidence);write(REPORT/'result.json',result)
        print(json.dumps({k:result[k] for k in ['metrics','paired','seconds']},indent=2))
    except BaseException as error:
        if (OUT/'oof.npz').exists():(OUT/'oof.npz').rename(OUT/'oof.partial.npz')
        write(REPORT/'failure.json',dict(status='failed_incomplete',error_type=type(error).__name__,error=str(error)));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':
    sys.modules.setdefault('mechanism_combo',sys.modules[__name__]);Curve.__module__='mechanism_combo';Design.__module__='mechanism_combo'
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    check() if args.check else run()
