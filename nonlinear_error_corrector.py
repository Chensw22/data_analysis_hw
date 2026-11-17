"""One fixed LightGBM logit residual on strict nested rawMitra OOF; CPU only."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit

import error_overlap_diagnostic as overlap
import relational_error_corrector as relational

ROOT = Path(__file__).resolve().parent
PLAN = ROOT/'reports/error_targeted_20261117/residual_protocol.json'
REPORT = ROOT/'reports/error_targeted_20261117'
OUT = ROOT/'artifacts/error_targeted_20261117/residual'
SPEND = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
COLUMNS = ['HomePlanet','CryoSleep','Destination','Age','VIP',*SPEND,'Deck','CabinNumber','Side',
           'MissingCount','TotalSpend','AllSpendingZero','SpendMissingCount','PaidServices','LuxurySpend',
           *[name+suffix for name in SPEND for suffix in ['_log1p','_missing']], 'Age_missing']
CATS = ['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
PARAMS = dict(objective='binary', learning_rate=.03, num_leaves=7, max_depth=3,
              min_data_in_leaf=100, lambda_l2=10., feature_fraction=1., bagging_fraction=1.,
              bagging_freq=0, num_threads=2, deterministic=True, force_col_wise=True,
              boost_from_average=False, verbosity=-1)
ROUNDS = 60


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def check_protocol():
    plan=json.loads(PLAN.read_text())
    if (plan['parameters']!=PARAMS or plan['num_boost_round']!=ROUNDS
            or plan['columns']!=COLUMNS or plan['categorical_columns']!=CATS
            or plan['seed_policy']!='20261112+outer_fold' or plan['epsilon']!=relational.EPS):
        raise RuntimeError('Fixed residual recipe differs from protocol.')
    for name,digest in plan['sources_sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Source changed after protocol freeze: '+name)
    return plan


def features():
    import iterate
    roster=pd.concat([pd.read_csv(ROOT/'data/raw'/name,usecols=lambda c:c!='Transported')
                      for name in ['train.csv','test.csv']],ignore_index=True)
    x,cats=iterate.build_features(roster,False)
    if len(COLUMNS)!=30 or list(x.columns)!=COLUMNS or cats!=CATS:
        raise RuntimeError('Expected exact original engineered_no_ids30 schema; got '+str(list(x.columns)))
    if any(name in x for name in ['Transported','GroupNumber','GroupMember','PassengerId']):
        raise RuntimeError('Target/ID column in residual features.')
    return x


def categorical_frames(train,query,vocabulary=None):
    train=train.copy();query=query.copy()
    if vocabulary is None:
        vocabulary={name:sorted(train[name].dropna().astype(str).unique().tolist()) for name in CATS}
    for name in CATS:
        train[name]=pd.Categorical(train[name],categories=vocabulary[name],ordered=False)
        query[name]=pd.Categorical(query[name],categories=vocabulary[name],ordered=False)
    # The fixed __MISSING__ token is retained if observed in outertrain; unseen values become NaN.
    return train,query,vocabulary


def constant_reference(dev,fold_ids,raw,splits):
    folder=ROOT/'reports/error_rescue_20261117'
    result=json.loads((folder/'relational_result.json').read_text())
    protocol=folder/'relational_protocol.json'
    if result['status']!='complete' or sha(protocol)!=result['protocol_sha256']:
        raise RuntimeError('Original constant-control evidence incomplete.')
    path=ROOT/result['oof_path']
    if sha(path)!=result['oof_sha256']:raise RuntimeError('Constant-control OOF hash changed.')
    with np.load(path,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['dev_ids'],dev)
        np.testing.assert_array_equal(saved['fold_id'],fold_ids)
        np.testing.assert_array_equal(saved['raw'],raw)
        constant=saved['constant'].copy()
    for outer,(a,b) in enumerate(splits):
        record=result['folds'][outer]
        if (record['train_ids_sha256']!=relational.nested.ids_hash(dev[a])
                or record['query_ids_sha256']!=relational.nested.ids_hash(dev[b])):
            raise RuntimeError('Constant-control split differs.')
        np.testing.assert_array_equal(constant[b],expit(relational.logit(raw[b])+record['constant']['intercept']))
    evidence={str(p.relative_to(ROOT)):sha(p) for p in [path,folder/'relational_result.json',protocol]}
    return constant,evidence


def run():
    plan=check_protocol()
    if OUT.exists() or (REPORT/'residual_result.json').exists():
        raise FileExistsError('One frozen residual experiment; no overwrite/resume.')
    OUT.mkdir(parents=True);started=time.monotonic()
    try:
        import lightgbm as lgb
        inputs,splits,fold_ids,bases,evidence=overlap.load_authenticated()
        dev,y=inputs['dev'],inputs['y'];raw=bases['rawMitra']
        x=features();constant,control_evidence=constant_reference(dev,fold_ids,raw,splits)
        evidence.update(control_evidence)
        frozen=relational.nested.read(ROOT/'artifacts/planet_mixture_20261117/freeze.json')
        relational.nested.verify_hashes(frozen['sources'])
        candidate=np.full(len(dev),np.nan);folds=[]
        for outer,(a,b) in enumerate(splits):
            tick=time.monotonic()
            inner=relational.load_inner(outer,a,b,dev,y,evidence)
            train,query,vocab=categorical_frames(x.iloc[dev[a]],x.iloc[dev[b]])
            train_offset=relational.logit(inner)
            dataset=lgb.Dataset(train,label=y[a],init_score=train_offset,
                                categorical_feature=CATS,free_raw_data=True)
            params={**PARAMS,'seed':20261112+outer}
            # No valid_sets, early stopping, custom evaluation or outer labels passed to training.
            model=lgb.train(params,dataset,num_boost_round=ROUNDS)
            residual=np.asarray(model.predict(query,raw_score=True),dtype=np.float64)
            p=expit(relational.logit(raw[b])+residual)
            if p.shape!=(len(b),) or not np.isfinite(p).all():
                raise RuntimeError('Incomplete/nonfinite residual predictions.')
            path=OUT/f'fold{outer}_booster.txt';model.save_model(str(path))
            schema=OUT/f'fold{outer}_schema.json';write(schema,dict(columns=COLUMNS,categories=vocab))
            restored=lgb.Booster(model_file=str(path))
            stored=json.loads(schema.read_text())
            if stored['columns']!=COLUMNS:raise RuntimeError('Saved schema changed.')
            _,restored_query,_=categorical_frames(x.iloc[dev[a]],x.iloc[dev[b]],stored['categories'])
            restored_residual=np.asarray(restored.predict(restored_query,raw_score=True))
            reload_p=expit(relational.logit(raw[b])+restored_residual)
            np.testing.assert_array_equal(residual,restored_residual)
            np.testing.assert_array_equal(p,reload_p)
            candidate[b]=p
            np.savez(OUT/f'fold{outer}_rows.npz',train_ids=dev[a],query_ids=dev[b],inner_raw=inner,
                     residual_logit=residual,probability=p)
            row=dict(fold=outer,training_rows=len(a),query_rows=len(b),seed=params['seed'],
                     training_ids_sha256=relational.nested.ids_hash(dev[a]),query_ids_sha256=relational.nested.ids_hash(dev[b]),
                     inner_base_support_rows='3090/3091; outer cached raw uses4636',
                     requested_rounds=ROUNDS,actual_iterations=model.current_iteration(),
                     vocabulary=vocab,unknown_query_categories={name:int(query[name].isna().sum()) for name in CATS},
                     metrics={key:relational.score(values,y[b]) for key,values in
                              [('raw',raw[b]),('constant',constant[b]),('nonlinear',p)]},
                     paired={key:relational.pair(p,reference,y[b]) for key,reference in [('raw',raw[b]),('constant',constant[b])]},
                     booster_sha256=sha(path),schema_sha256=sha(schema),
                     reload_max_absolute_difference=float(np.max(np.abs(p-reload_p))),
                     feature_gain=dict(zip(COLUMNS,model.feature_importance(importance_type='gain').tolist())),
                     seconds=time.monotonic()-tick)
            folds.append(row);write(REPORT/f'residual_fold{outer}.json',row)
            print(json.dumps(row),flush=True)
            del model,restored,dataset
        if len(folds)!=3 or not np.isfinite(candidate).all():raise RuntimeError('Three complete folds required.')
        # Construct shared-error subsets only after all correction fits have completed.
        wrong=np.column_stack([(bases[name]>=.5)!=y for name in overlap.FAMILIES])
        masks={'five_family_common_wrong':wrong.all(axis=1),'five_family_common_correct':(~wrong).all(axis=1)}
        if int(masks['five_family_common_wrong'].sum())!=914 or int(masks['five_family_common_correct'].sum())!=5263:
            raise RuntimeError('Historical reporting subset differs.')
        probabilities={'raw':raw,'constant':constant,'nonlinear':candidate}
        subsets={name:dict(rows=int(mask.sum()),correct_by_model={k:int(((p[mask]>=.5)==y[mask]).sum()) for k,p in probabilities.items()},
                          paired={k:relational.pair(candidate,reference,y,mask) for k,reference in [('raw',raw),('constant',constant)]})
                 for name,mask in masks.items()}
        path=OUT/'oof.npz'
        np.savez(path,dev_ids=dev,fold_id=fold_ids,passenger_ids=inputs['passengers'],group_ids=inputs['groups'],
                 **probabilities,**masks)
        check_protocol()
        for name,digest in evidence.items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Input evidence changed: '+name)
        result=dict(status='complete',protocol_sha256=sha(PLAN),folds=folds,
                    metrics={k:relational.score(p,y) for k,p in probabilities.items()},
                    paired={k:relational.pair(candidate,p,y) for k,p in [('raw',raw),('constant',constant)]},
                    diagnostic_subsets=subsets,oof_path=str(path.relative_to(ROOT)),oof_sha256=sha(path),
                    seconds=time.monotonic()-started,lightgbm_version=lgb.__version__,
                    base_models_retrained=False,old_holdout_labels_read=False,automatic_upload=False,
                    disclosure='One fixed adaptive-development diagnostic. Strict nested base OOF; no shared-error subset fitting. Inner/outer base training size differs.')
        write(REPORT/'residual_evidence.json',evidence);write(REPORT/'residual_result.json',result)
        return result
    except BaseException as error:
        path=OUT/'oof.npz'
        if path.exists():path.rename(OUT/'oof.partial.npz')
        write(REPORT/'residual_failure.json',dict(status='failed_inconclusive',error_type=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
