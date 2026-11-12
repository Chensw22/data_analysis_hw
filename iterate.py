"""Development-only candidate screening, repeated-CV gate, then full-data export.

No evaluation of the already opened holdout is performed. Group context uses the
known, unlabeled passenger roster (train + test attributes); target values are
explicitly removed before constructing any feature.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

import experiment as base

ROOT = Path(__file__).resolve().parent
SEEDS = [base.SEED, base.SEED+1, base.SEED+2]
BASELINE = 'v6_no_id_features'
MIN_LOCAL_ACCURACY = .83
CAT = dict(depth=6, iterations=700, learning_rate=.045, l2_leaf_reg=5)
CONFIGS = {
    BASELINE: dict(context=False, model='catboost', **CAT),
    'v9_more_trees': dict(context=False, model='catboost', depth=6, iterations=1400, learning_rate=.03, l2_leaf_reg=8),
    'v10_shallow': dict(context=False, model='catboost', depth=5, iterations=1500, learning_rate=.035, l2_leaf_reg=8),
    'v11_group_context': dict(context=True, model='catboost', depth=6, iterations=1100, learning_rate=.035, l2_leaf_reg=8),
    'v12_group_deeper': dict(context=True, model='catboost', depth=7, iterations=1200, learning_rate=.03, l2_leaf_reg=8),
    'v13_histogram': dict(context=False, model='histogram', max_iter=500, max_leaf_nodes=15, learning_rate=.05, l2_regularization=5, min_samples_leaf=25),
    'v14_histogram_group': dict(context=True, model='histogram', max_iter=500, max_leaf_nodes=15, learning_rate=.05, l2_regularization=5, min_samples_leaf=25),
}
BLEND = ['v11_group_context', 'v12_group_deeper', 'v14_histogram_group']


def assert_frozen(out):
    protocol=json.loads((out/'protocol.json').read_text())
    for filename,expected in protocol['sources'].items():
        actual=hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Experiment source changed: {filename}; use a new run, not an old gate.')
    if protocol['configs'] != CONFIGS or protocol['blend'] != BLEND:
        raise RuntimeError('Candidate configuration changed; use a new run.')


def build_features(roster, context):
    roster = roster.drop(columns='Transported', errors='ignore').reset_index(drop=True)
    x, _ = base.features(roster, 'engineered_no_ids')
    if context:
        group = roster.PassengerId.str.split('_').str[0]
        last = roster.Name.str.split().str[-1]
        x['GroupSize'] = group.map(group.value_counts()).astype(float)
        x['SurnameCount'] = last.map(last.value_counts()).astype(float)
        cabin_count = roster.Cabin.map(roster.Cabin.value_counts())
        x['CabinOccupants'] = cabin_count.astype(float)
        sleeping = roster.CryoSleep.astype(float)
        known = sleeping.notna().astype(int)
        numerator = sleeping.fillna(0).groupby(group).transform('sum') - sleeping.fillna(0)
        denominator = known.groupby(group).transform('sum') - known
        x['PeerCryoSleepRate'] = numerator / denominator.replace(0, np.nan)
        x['PeerKnownCryoCount'] = denominator.astype(float)
        age_sum = roster.Age.fillna(0).groupby(group).transform('sum') - roster.Age.fillna(0)
        age_n = roster.Age.notna().astype(int).groupby(group).transform('sum') - roster.Age.notna().astype(int)
        x['PeerMeanAge'] = age_sum / age_n.replace(0, np.nan)
        x['GroupChildren'] = roster.Age.lt(13).astype(int).groupby(group).transform('sum').astype(float)
        # Fill only when the group's observed nonmissing values are unanimous.
        for col in ['HomePlanet', 'Destination']:
            grouped = roster[col].groupby(group)
            unanimous = grouped.nunique().eq(1)
            value = group.map(grouped.first().where(unanimous))
            x[col] = roster[col].fillna(value).fillna('__MISSING__').astype(str)
        decks = roster.Cabin.str.split('/').str[0]
        sides = roster.Cabin.str.split('/').str[-1]
        for col, series in [('Deck', decks), ('Side', sides)]:
            grouped = series.groupby(group)
            value = group.map(grouped.first().where(grouped.nunique().eq(1)))
            x[col] = series.fillna(value).fillna('__MISSING__').astype(str)
    cats = x.select_dtypes(include=['object', 'str', 'bool']).columns.tolist()
    return x, cats


def dataset():
    train, test, _ = base.load_data()
    # Drop target before concatenation: label values cannot reach group aggregation.
    roster = pd.concat([train.drop(columns='Transported'), test], ignore_index=True)
    matrices = {context: build_features(roster, context) for context in [False, True]}
    # Reconstruct the original development partition from committed passenger IDs,
    # so fresh clones don't need ignored NPZ files from previous runs.
    from sklearn.model_selection import train_test_split
    dev, _ = train_test_split(np.arange(len(train)), test_size=.2,
                              stratify=train.Transported, random_state=base.SEED)
    dev = np.sort(dev)
    return train, test, matrices, dev


def fit(name, matrices, train_idx, y, predict_idx, seed):
    config = CONFIGS[name]
    x, cats = matrices[config['context']]
    params = {k:v for k,v in config.items() if k not in ['context','model']}
    if config['model'] == 'catboost':
        model = CatBoostClassifier(**params, cat_features=cats, loss_function='Logloss',
                                   random_seed=seed, thread_count=4, verbose=False, allow_writing_files=False)
    else:
        x = x.copy()
        for col in cats:
            x[col] = x[col].astype('category')
        model = HistGradientBoostingClassifier(**params, categorical_features='from_dtype',
                                                early_stopping=False, random_state=seed)
    with threadpool_limits(limits=4):
        model.fit(x.iloc[train_idx], y)
        pred = model.predict_proba(x.iloc[predict_idx])[:,1]
    return pred, model


def cross_validate(name, matrices, train, dev, seed, out):
    cache = out/f'{name}_seed{seed}.npz'
    if cache.exists():
        r = np.load(cache)
        return r['prob'], r['fold_accuracy'], float(r['seconds'])
    y = train.Transported.iloc[dev].astype(int).to_numpy()
    prob = np.zeros(len(dev))
    scores = []
    started = time.monotonic()
    for fold,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(dev,y)):
        pred,_ = fit(name,matrices,dev[a],y[a],dev[b],seed+fold)
        prob[b] = pred
        scores.append(accuracy_score(y[b],pred>=.5))
        print(name,seed,'fold',fold+1,round(scores[-1],5),flush=True)
    seconds=time.monotonic()-started
    np.savez(cache,prob=prob,fold_accuracy=scores,seconds=seconds)
    return prob,np.array(scores),seconds


def evaluate(name, matrices, train, dev, seed, out):
    if name != 'v15_context_blend':
        return cross_validate(name,matrices,train,dev,seed,out)
    components=[cross_validate(n,matrices,train,dev,seed,out) for n in BLEND]
    prob=np.mean([c[0] for c in components],axis=0)
    y=train.Transported.iloc[dev].astype(int).to_numpy()
    scores=[accuracy_score(y[b],prob[b]>=.5) for _,b in StratifiedKFold(3,shuffle=True,random_state=seed).split(dev,y)]
    return prob,np.array(scores),sum(c[2] for c in components)


def screen(out):
    if out.exists():
        raise RuntimeError('Run already exists. Use a new ID; do not overwrite experiments.')
    out.mkdir(parents=True)
    train,test,matrices,dev=dataset()
    shutil.copy2(__file__,out/'iterate_source.py')
    shutil.copy2(ROOT/'experiment.py',out/'base_source.py')
    shutil.copy2(ROOT/'requirements.lock.txt',out/'requirements.lock.txt')
    protocol=dict(seed=SEEDS[0],confirmation_seeds=SEEDS[1:],folds=3,threshold=.5,
        development_rows=len(dev),configs=CONFIGS,blend=BLEND,
        target_policy='Development labels only; no holdout scoring. Full train labels used only for final fit.',
        context_policy='Transductive unlabeled full train+test roster: group size, peer attributes, unanimous imputation. No target propagation.',
        gate='Screen accuracy >0.83 and improvement >=0.001; both new split seeds improve; confirmation mean accuracy >0.83 and mean gain >=0.001. Candidate frozen before confirmation. User approval required before online submission.',
        sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['iterate.py','experiment.py']})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    y=train.Transported.iloc[dev].astype(int).to_numpy()
    rows=[]
    for name in [*CONFIGS,*(['v15_context_blend'] if BLEND else [])]:
        prob,folds,seconds=evaluate(name,matrices,train,dev,SEEDS[0],out)
        row=dict(version=name,accuracy=float(accuracy_score(y,prob>=.5)),log_loss=float(log_loss(y,prob)),
                 fold_std=float(np.std(folds)),seconds=seconds)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out/'screen_results.csv',index=False)
        print(json.dumps(row),flush=True)
    results=pd.DataFrame(rows).sort_values(['accuracy','log_loss'],ascending=[False,True])
    results.to_csv(out/'screen_results.csv',index=False)
    winner=results.iloc[0]
    baseline=results[results.version==BASELINE].iloc[0]
    selection=dict(version=winner.version,components=BLEND if winner.version=='v15_context_blend' else [winner.version],
                   screen_accuracy=float(winner.accuracy),screen_gain=float(winner.accuracy-baseline.accuracy),stage='frozen_for_confirmation')
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print('FROZEN',json.dumps(selection),flush=True)


def confirm(out):
    assert_frozen(out)
    if (out/'gate.json').exists():
        raise RuntimeError('Confirmation already recorded.')
    train,_,matrices,dev=dataset()
    y=train.Transported.iloc[dev].astype(int).to_numpy()
    selection=json.loads((out/'selection.json').read_text())
    name=selection['version']
    rows=[]
    screen_accuracy=selection.get('screen_accuracy')
    if screen_accuracy is None:
        screen=pd.read_csv(out/'screen_results.csv')
        screen_accuracy=float(screen.loc[screen.version.eq(name),'accuracy'].iloc[0])
    if screen_accuracy > MIN_LOCAL_ACCURACY and selection['screen_gain'] >= .001:
        for seed in SEEDS[1:]:
            baseline,_,_=evaluate(BASELINE,matrices,train,dev,seed,out)
            candidate,_,_=evaluate(name,matrices,train,dev,seed,out)
            a=float(accuracy_score(y,baseline>=.5)); b=float(accuracy_score(y,candidate>=.5))
            rows.append(dict(seed=seed,baseline_accuracy=a,candidate_accuracy=b,gain=b-a,
                corrections=int(((candidate>=.5)==y)[(baseline>=.5)!=y].sum()),
                regressions=int(((candidate>=.5)!=y)[(baseline>=.5)==y].sum())))
    passed=bool(len(rows)==2 and all(r['gain']>0 for r in rows) and np.mean([r['gain'] for r in rows])>=.001
                and np.mean([r['candidate_accuracy'] for r in rows]) > MIN_LOCAL_ACCURACY)
    group_result=None
    if passed:
        groups=train.PassengerId.iloc[dev].str.split('_').str[0]
        group_prob={candidate:np.zeros(len(dev)) for candidate in [BASELINE,name]}
        for fold,(a,b) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=SEEDS[0]).split(dev,y,groups)):
            assert not set(groups.iloc[a]) & set(groups.iloc[b])
            for candidate in [BASELINE,name]:
                components=BLEND if candidate=='v15_context_blend' else [candidate]
                preds=[fit(n,matrices,dev[a],y[a],dev[b],SEEDS[0]+fold)[0] for n in components]
                group_prob[candidate][b]=np.mean(preds,axis=0)
            print('Group confirmation fold',fold+1,'completed',flush=True)
        group_result={candidate:float(accuracy_score(y,prob>=.5)) for candidate,prob in group_prob.items()}
        passed=bool(group_result[name]>MIN_LOCAL_ACCURACY and group_result[name]>group_result[BASELINE])
    gate=dict(passed=passed,minimum_local_accuracy_exclusive=MIN_LOCAL_ACCURACY,screen_accuracy=screen_accuracy,confirmation=rows,group_confirmation=group_result,
              note='Repeated CV on the same development rows: stability evidence, not an independent holdout or a probability of leaderboard success.')
    (out/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    print('GATE',json.dumps(gate),flush=True)


def export(out):
    assert_frozen(out)
    if not json.loads((out/'gate.json').read_text())['passed']:
        raise RuntimeError('Candidate failed local gate; do not submit.')
    if (out/'submission.csv').exists():
        raise RuntimeError('Submission already exported.')
    train,test,matrices,_=dataset()
    selection=json.loads((out/'selection.json').read_text())
    preds=[]
    for name in selection['components']:
        prob,model=fit(name,matrices,np.arange(len(train)),train.Transported.astype(int),
                       np.arange(len(train),len(train)+len(test)),SEEDS[0])
        preds.append(prob)
        if isinstance(model,CatBoostClassifier):
            model.save_model(str(out/(name+'.cbm')))
        else:
            joblib.dump(model,out/(name+'.joblib'))
    prob=np.mean(preds,axis=0)
    pd.DataFrame({'PassengerId':test.PassengerId,'Transported':prob>=.5}).to_csv(out/'submission.csv',index=False)
    pd.DataFrame({'PassengerId':test.PassengerId,'probability':prob}).to_csv(out/'test_probabilities.csv',index=False)
    print('Exported',out/'submission.csv',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['screen','confirm','export'])
    parser.add_argument('--run',default='run_003'); args=parser.parse_args()
    {'screen':screen,'confirm':confirm,'export':export}[args.stage](ROOT/'artifacts'/args.run)
