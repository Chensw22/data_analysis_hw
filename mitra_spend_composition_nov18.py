"""One raw-Mitra fold: compact13 plus complete total and five spending shares."""
import os
import sys
if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None,'FALSE'):
    raise RuntimeError('Fresh FALSE-only process required.')
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'

import argparse
import hashlib
import io
import json
from pathlib import Path
import signal
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

import audit_top5_references as audit
import error_overlap_diagnostic as overlap
import error_rescue_feature_probe as previous
import mitra_candidate as base
import mitra_counts_fixed15_online_first as counts

ROOT=Path(__file__).resolve().parent
NAME='Mitra_complete_spending_composition19_zeroshot'
PLAN='reports/error_followup_20261118/composition_protocol.json'
OUT=ROOT/'artifacts/error_followup_20261118/composition_fold0'
REPORT=ROOT/'reports/error_followup_20261118/composition_result.json'
SPEND=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
EXTRA=['TotalSpendComplete']+[name+'CompleteShare' for name in SPEND]
COLUMNS=list(base.COMPACT)+EXTRA
SEED,BUDGET=20261112,1200
if __name__=='__main__':sys.modules['mitra_spend_composition_nov18']=sys.modules[__name__]


def add_composition(x):
    result=x[base.COMPACT].copy()
    values=x[SPEND].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float)
    if np.isinf(values).any() or np.any(values<0):raise ValueError('Spending must be nonnegative finite or missing.')
    complete=np.isfinite(values).all(axis=1)
    total=np.full(len(x),np.nan)
    total[complete]=values[complete].sum(axis=1)
    if np.isinf(total).any():raise ValueError('Complete spending sum overflow.')
    shares=np.full(values.shape,np.nan)
    positive=complete & (total>0);zero=complete & (total==0)
    shares[positive]=values[positive]/total[positive,None]
    shares[zero]=0.
    result[EXTRA[0]]=total
    for i,name in enumerate(EXTRA[1:]):result[name]=shares[:,i]
    return result


class CompositionPreprocessor(base.MitraPreprocessor):
    def fit(self,x,y,seed=SEED):
        y=base.binary_labels(y,len(x))
        self.reference=base.MitraPreprocessor().fit(x,y,seed)
        self.columns=list(COLUMNS);self.categories=list(base.CATEGORICAL)
        self.vocab={name:dict(v) for name,v in self.reference.vocab.items()}
        array=self.encode(x);self.all_missing=np.isnan(array).all(axis=0)
        array[:,self.all_missing]=0.
        self.upstream=base.Preprocessor(dim_embedding=None,n_classes=2,dim_output=10,
            use_quantile_transformer=False,use_feature_count_scaling=False,
            use_random_transforms=False,shuffle_classes=False,shuffle_features=False,
            random_mirror_regression=False,random_mirror_x=base.SETTINGS['random_mirror_x'],task=base.Task.CLASSIFICATION)
        state=np.random.get_state()
        try:
            np.random.seed(seed);self.upstream.fit(array.copy(),y.copy())
        finally:np.random.set_state(state)
        self.active_columns=np.asarray(self.columns)[~self.upstream.singular_features].tolist()
        self.n_features=self.transform(x).shape[1]
        return self

    def encode(self,x):return super().encode(add_composition(x))

    def transform(self,x):
        result=super().transform(x);expected=self.reference.transform(x)
        indices=[self.active_columns.index(c) for c in self.active_columns if c in base.COMPACT]
        if len(indices)!=expected.shape[1] or not np.array_equal(result[:,indices],expected):
            raise RuntimeError('Original compact13 active encoding/imputation/mirror changed.')
        return result


CompositionPreprocessor.__module__='mitra_spend_composition_nov18'


def check():
    x=pd.DataFrame(index=np.arange(16))
    for name in base.COMPACT:
        x[name]=np.arange(16,dtype=float)+1
    for name in base.CATEGORICAL:x[name]=['a','b']*8
    for i,name in enumerate(SPEND):x[name]=(np.arange(16)%(i+3)).astype(float)*(i+1)
    x.loc[0,SPEND]=0.;x.loc[1,SPEND[0]]=np.nan;x.loc[2,SPEND]=[1.,2.,3.,4.,5.]
    out=add_composition(x)
    pd.testing.assert_frame_equal(out[base.COMPACT],x[base.COMPACT],check_exact=True)
    assert out.loc[0,EXTRA].eq(0).all() and out.loc[1,EXTRA].isna().all()
    assert out.loc[2,EXTRA[0]]==15
    np.testing.assert_allclose(out.loc[2,EXTRA[1:]].to_numpy(float),np.arange(1,6)/15)
    bad=x.copy();bad.loc[1,SPEND[1]]=-1
    try:add_composition(bad)
    except ValueError:pass
    else:raise AssertionError('Negative amount accepted.')
    bad.loc[1,SPEND[1]]=np.inf
    try:add_composition(bad)
    except ValueError:pass
    else:raise AssertionError('Infinite amount accepted.')
    prep=CompositionPreprocessor().fit(x,np.arange(16)%2,SEED)
    assert prep.n_features==19 and prep.columns==COLUMNS
    actual=prep.transform(x)
    np.testing.assert_array_equal(actual[:,:13],prep.reference.transform(x))
    query=x.iloc[[3,1,2]].copy();query.loc[3,base.CATEGORICAL[0]]='unseen'
    np.testing.assert_array_equal(prep.transform(query)[:,:13],prep.reference.transform(query))
    buffer=io.BytesIO();joblib.dump(prep,buffer);buffer.seek(0)
    np.testing.assert_array_equal(prep.transform(query),joblib.load(buffer).transform(query))
    print('Synthetic complete/zero/missing/negative/inf rules, original13 prefix, 19 actual columns, unseen query and CPU serialization passed; no model loaded.')


def fit_predict(features,train_ids,y_train,query_ids):
    train_ids=np.asarray(train_ids,dtype=np.int64);query_ids=np.asarray(query_ids,dtype=np.int64)
    if (len(train_ids)!=4636 or len(query_ids)!=2318 or np.intersect1d(train_ids,query_ids).size
            or len(np.unique(train_ids))!=4636 or len(np.unique(query_ids))!=2318):
        raise ValueError('Exact disjoint original fold0 membership required.')
    y_train=base.binary_labels(y_train,len(train_ids))
    x,q=features.loc[train_ids],features.loc[query_ids]
    base.seed_everything(SEED,'mps')
    prep=CompositionPreprocessor().fit(x,y_train,SEED);support=prep.transform(x)
    if support.shape!=(4636,19) or prep.active_columns!=COLUMNS:raise RuntimeError('All19 model inputs required.')
    model=predictor=restored=None
    try:
        model,config=base.load_initial_model(base.CHECKPOINT_DIR,'mps')
        metadata=dict(name=NAME,recipe='zero-shot',seed=SEED,selected_steps=0,
            checkpoint_sha256=base.WEIGHT_SHA,training_rows=4636,query_rows=2318,
            training_ids_sha256=base.ids_hash(train_ids),query_ids_sha256=base.ids_hash(query_ids),
            labels_sha256=base.ids_hash(y_train),input_columns=COLUMNS,actual_features=19,
            original13_prefix_exact=True,query_chunk=2318,full_prediction_support=True)
        predictor=base.MitraPredictor(prep,config,model.state_dict(),support,y_train,'mps',metadata,query_chunk=2318)
        model=None;base.release_memory()
        p=predictor.predict_proba(q)[:,1]
        joblib.dump(predictor,OUT/'model.joblib')
        predictor.release();predictor=None;base.release_memory()
        restored=joblib.load(OUT/'model.joblib')
        if (restored.device!='cpu' or restored._engine is not None or restored.query_chunk!=2318
                or restored.preprocessor.columns!=COLUMNS or restored.support.shape!=(4636,19)):
            raise RuntimeError('Reloaded full19 context differs.')
        np.testing.assert_array_equal(restored.labels,y_train)
        np.testing.assert_array_equal(restored.support,restored.preprocessor.transform(x))
        restored.device='mps';again=restored.predict_proba(q)[:,1]
        difference=float(np.max(np.abs(p-again)))
        if difference>2e-6 or not np.array_equal(p>=.5,again>=.5):raise RuntimeError('Fullfold reload equivalence failed.')
        metadata.update(reload_rows=2318,reload_max_absolute_difference=difference,model_sha256=audit.file_hash(OUT/'model.joblib'))
        return p,again,metadata
    finally:
        for current in (predictor,restored):
            if current is not None:current.release()
        model=predictor=restored=None;base.release_memory()


def run():
    if OUT.exists() or REPORT.exists():raise RuntimeError('No overwrite/resume.')
    plan=json.loads((ROOT/PLAN).read_text())
    if plan['columns']!=COLUMNS or plan['seed']!=SEED or plan['budget_seconds']!=BUDGET:raise RuntimeError('Fixedrecipe differs.')
    for name,digest in plan['pinned_evidence_sha256'].items():
        if audit.file_hash(ROOT/name)!=digest:raise RuntimeError('Pinned evidence changed: '+name)
    original,lock=counts.verify_reference()
    names=sorted(set(previous.OWN+[Path(__file__).name,PLAN,'error_overlap_diagnostic.py']+
        [n for n in original['source_hashes'] if n.startswith('third_party/mitra/')]))
    commit=previous.committed(names)
    OUT.mkdir(parents=True);started=time.monotonic();base.DEADLINE=started+BUDGET
    def timeout(*_):raise TimeoutError('Fixed composition fold exceeded1200seconds.')
    handler=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,BUDGET)
    try:
        inputs,splits,fold_ids,bases,evidence=overlap.load_authenticated()
        a,b=splits[0];dev,y=inputs['dev'],inputs['y'];raw=bases['rawMitra'][b]
        frozen=dict(source_commit=commit,sources={n:audit.file_hash(ROOT/n) for n in names},
            environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),protocol=plan)
        frozen['sources'].update(evidence);frozen['sources'].update(plan['pinned_evidence_sha256'])
        audit.json_write(OUT/'freeze.json',frozen)
        if not torch.backends.mps.is_available():raise RuntimeError('Original MPS required; no fallback.')
        torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65);counts.require_false()
        p,again,metadata=fit_predict(inputs['features'],dev[a],y[a].copy(),dev[b])
        if p.shape!=(2318,) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise RuntimeError('Complete finite fold required.')
        # Historical outcome masks are first parsed after inference and reload.
        rows=pd.read_csv(ROOT/plan['catalog_rows'],dtype={'PassengerId':str})
        for col,expected in [('row_id',dev),('PassengerId',inputs['passengers']),('y',y),('fold',fold_ids)]:
            np.testing.assert_array_equal(rows[col],expected)
        hard=rows.all_recipes_still_wrong.to_numpy(bool);five=rows.family_wrong_count.to_numpy()==5
        assert hard.sum()==546 and five.sum()==914
        q=inputs['features'].loc[dev[b]]
        masks=dict(all=np.ones(len(b),bool),historical546=hard[b],historical914=five[b],
            Earth_awake=(q.HomePlanet.eq('Earth')&q.CryoSleep.eq('False')).to_numpy(),
            Earth_sleep=(q.HomePlanet.eq('Earth')&q.CryoSleep.eq('True')).to_numpy())
        subsets={}
        for name,m in masks.items():
            subsets[name]=dict(**previous.paired(p,raw,y[b],m),
                candidate_correct=int((((p>=.5)==y[b])&m).sum()),raw_correct=int((((raw>=.5)==y[b])&m).sum()))
            for key,probability in [('candidate',p),('raw',raw)]:
                subsets[name][key+'_confusion']=dict(
                    TN=int((m&(y[b]==0)&(probability<.5)).sum()),FP=int((m&(y[b]==0)&(probability>=.5)).sum()),
                    FN=int((m&(y[b]==1)&(probability<.5)).sum()),TP=int((m&(y[b]==1)&(probability>=.5)).sum()))
        np.savez(OUT/'predictions.npz',prob=p,reload_prob=again,raw=raw,train_ids=dev[a],query_ids=dev[b],
            passenger_ids=np.asarray(inputs['passengers'][b],dtype=str),group_ids=np.asarray(inputs['groups'][b],dtype=str),
            y=y[b],**{name+'_mask':m for name,m in masks.items()})
        previous.verify_sources(frozen)
        result=dict(status='complete_single_fold_adaptive_diagnostic',name=NAME,source_commit=commit,
            accuracy=float(np.mean((p>=.5)==y[b])),log_loss=float(log_loss(y[b],p,labels=[0,1])),
            raw_accuracy=float(np.mean((raw>=.5)==y[b])),raw_log_loss=float(log_loss(y[b],raw,labels=[0,1])),
            subsets=subsets,metadata=metadata,seconds=time.monotonic()-started,
            prediction_sha256=audit.file_hash(OUT/'predictions.npz'),freeze_sha256=audit.file_hash(OUT/'freeze.json'),
            no_new_upload_authorization=True,interpretation=plan['limitations'])
        audit.json_write(REPORT,result);print(json.dumps(result,indent=2))
    except BaseException as error:
        audit.json_write(OUT/'failure.json',dict(status='failed_incomplete',error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        base.release_memory();signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();check() if args.check else run()
