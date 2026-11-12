"""Reproducible development CV, one-time holdout audit, and Kaggle prediction."""
import argparse
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SPEND = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
SEED = 20261112
CONFIGS = {
    'v0_logistic': dict(features='raw', model='logistic'),
    'v1_catboost_raw': dict(features='raw', model='catboost', depth=6, iterations=700, learning_rate=.045, l2_leaf_reg=5),
    'v2_features': dict(features='engineered', model='catboost', depth=6, iterations=700, learning_rate=.045, l2_leaf_reg=5),
    'v3_regularized': dict(features='engineered', model='catboost', depth=5, iterations=1100, learning_rate=.035, l2_leaf_reg=8),
    'v4_deeper': dict(features='engineered', model='catboost', depth=7, iterations=900, learning_rate=.035, l2_leaf_reg=8),
    'v6_no_id_features': dict(features='engineered_no_ids', model='catboost', depth=6, iterations=700, learning_rate=.045, l2_leaf_reg=5),
    'v7_no_spend_features': dict(features='engineered_no_spend', model='catboost', depth=6, iterations=700, learning_rate=.045, l2_leaf_reg=5),
    'v8_cryo_rules': dict(features='engineered_rules', model='catboost', depth=6, iterations=700, learning_rate=.045, l2_leaf_reg=5),
}


def load_data():
    train = pd.read_csv(ROOT/'data/raw/train.csv')
    test = pd.read_csv(ROOT/'data/raw/test.csv')
    sample = pd.read_csv(ROOT/'data/raw/sample_submission.csv')
    assert train.shape == (8693, 14) and test.shape == (4277, 13)
    assert set(train.columns) - set(test.columns) == {'Transported'}
    assert train.Transported.notna().all() and set(train.Transported.unique()) == {False, True}
    assert train.PassengerId.is_unique and test.PassengerId.is_unique
    assert not set(train.PassengerId) & set(test.PassengerId)
    assert sample.PassengerId.equals(test.PassengerId)
    return train, test, sample


def features(df, mode):
    # All transformations are row-local: no target, test labels, or fitted statistics.
    original = df
    if mode == 'engineered_rules':
        df = df.copy()
        asleep = df.CryoSleep.eq(True)
        df.loc[asleep, SPEND] = df.loc[asleep, SPEND].fillna(0)
        paid = df[SPEND].gt(0).any(axis=1)
        df.loc[df.CryoSleep.isna() & paid, 'CryoSleep'] = False
    x = df.drop(columns=['Transported', 'PassengerId', 'Name'], errors='ignore').copy()
    if mode.startswith('engineered'):
        cabin = df.Cabin.str.split('/', expand=True).reindex(columns=[0, 1, 2])
        x['Deck'] = cabin[0]
        x['CabinNumber'] = pd.to_numeric(cabin[1], errors='coerce').astype(float)
        x['Side'] = cabin[2]
        x = x.drop(columns='Cabin')
        passenger = df.PassengerId.str.split('_', expand=True)
        x['GroupNumber'] = passenger[0].astype(int)
        x['GroupMember'] = passenger[1].astype(int)
        x['MissingCount'] = df.drop(columns=['Transported'], errors='ignore').isna().sum(axis=1)
        # Keep all-missing spending unknown rather than silently making it zero.
        total = df[SPEND].sum(axis=1, min_count=1)
        x['TotalSpend'] = total
        x['AllSpendingZero'] = ((total == 0) & df[SPEND].notna().all(axis=1)).astype(int)
        x['SpendMissingCount'] = df[SPEND].isna().sum(axis=1)
        x['PaidServices'] = (df[SPEND] > 0).sum(axis=1)
        x['LuxurySpend'] = df[['Spa', 'VRDeck', 'RoomService']].sum(axis=1, min_count=1)
        for col in SPEND:
            x[col+'_log1p'] = np.log1p(df[col])
            x[col+'_missing'] = df[col].isna().astype(int)
        x['Age_missing'] = df.Age.isna().astype(int)
        if mode == 'engineered_no_ids':
            x = x.drop(columns=['GroupNumber', 'GroupMember'])
        if mode == 'engineered_no_spend':
            x = x.drop(columns=['TotalSpend', 'AllSpendingZero', 'SpendMissingCount',
                'PaidServices', 'LuxurySpend'] + [c+s for c in SPEND for s in ['_log1p', '_missing']])
        if mode == 'engineered_rules':
            # Preserve the original missingness so rules do not hide data quality.
            x['MissingCount'] = original.drop(columns=['Transported'], errors='ignore').isna().sum(axis=1)
            x['SpendMissingCount'] = original[SPEND].isna().sum(axis=1)
            for c in SPEND:
                x[c+'_missing'] = original[c].isna().astype(int)
    cats = x.select_dtypes(include=['object', 'str', 'bool']).columns.tolist()
    for col in cats:
        x[col] = x[col].fillna('__MISSING__').astype(str)
    return x, cats


def fit_predict(config, train, targets, predict_frames, seed=SEED):
    x, cats = features(train, config['features'])
    frames = [features(frame, config['features'])[0] for frame in predict_frames]
    if config['model'] == 'logistic':
        nums = [c for c in x if c not in cats]
        prep = ColumnTransformer([
            ('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), nums),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cats),
        ])
        model = make_pipeline(prep, LogisticRegression(C=1.0, max_iter=2500, random_state=seed))
    else:
        params = {k: v for k, v in config.items() if k not in ['features', 'model']}
        model = CatBoostClassifier(**params, cat_features=cats, loss_function='Logloss',
                                   random_seed=seed, thread_count=4, verbose=False,
                                   allow_writing_files=False)
    # Fixed iteration counts: outer validation labels are never used for early stopping.
    model.fit(x, targets)
    return [model.predict_proba(f)[:, 1] for f in frames], model, list(x.columns)


def metrics(y, prob):
    return dict(accuracy=float(accuracy_score(y, prob >= .5)),
                log_loss=float(log_loss(y, prob)),
                balanced_accuracy=float(balanced_accuracy_score(y, prob >= .5)))


def initialize(out):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/'experiment_source.py')
    shutil.copy2(ROOT/'data/raw/manifest.json', out/'data_manifest.json')
    shutil.copy2(ROOT/'requirements.lock.txt', out/'requirements.lock.txt')
    train, test, _ = load_data()
    dev, hold = train_test_split(np.arange(len(train)), test_size=.2, stratify=train.Transported,
                                random_state=SEED)
    dev, hold = np.sort(dev), np.sort(hold)
    np.savez(out/'split_indices.npz', development=dev, holdout=hold)
    pd.DataFrame({'PassengerId': train.PassengerId,
                  'partition': np.where(np.isin(np.arange(len(train)), hold), 'holdout', 'development')}).to_csv(out/'partitions.csv', index=False)
    # Exploratory statistics use development data only.
    train.iloc[dev].isna().mean().sort_values(ascending=False).to_csv(out/'missing_rates_development.csv', header=['missing_rate'])
    (out/'protocol.json').write_text(json.dumps(dict(seed=SEED, threshold=.5, folds=3,
        development_rows=len(dev), holdout_rows=len(hold), selection='maximum development OOF accuracy; log loss breaks ties',
        holdout_policy='One audit after candidate selection; no further tuning on holdout results.',
        configs=CONFIGS, python=platform.python_version(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
    print(f'Initialized {out.name}: development={len(dev)}, sealed holdout={len(hold)}', flush=True)


def develop(out):
    train, _, _ = load_data()
    dev = np.load(out/'split_indices.npz')['development']
    frame = train.iloc[dev].reset_index(drop=True)
    y = frame.Transported.astype(int).to_numpy()
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(frame, y))
    rows, predictions = [], {}
    for name, config in CONFIGS.items():
        cache = out/(name+'_oof.npz')
        if cache.exists():
            result = np.load(cache)
            prob, scores, seconds = result['prob'], result['fold_accuracy'], float(result['seconds'])
        else:
            started = time.monotonic()
            prob, scores = np.zeros(len(y)), []
            for fold, (a, b) in enumerate(folds):
                pred, _, _ = fit_predict(config, frame.iloc[a], y[a], [frame.iloc[b]], SEED+fold)
                prob[b] = pred[0]
                score = accuracy_score(y[b], pred[0] >= .5)
                scores.append(score)
                print(f'{name} fold {fold+1}: {score:.5f}', flush=True)
            seconds = time.monotonic()-started
            np.savez(cache, prob=prob, fold_accuracy=scores, seconds=seconds)
        predictions[name] = prob
        row = dict(version=name, **metrics(y, prob), fold_std=float(np.std(scores)), seconds=seconds)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out/'development_results.csv', index=False)
        print(json.dumps(row), flush=True)
    # Predeclared simple blend, no weight search or threshold optimization.
    blend_names = ['v2_features', 'v3_regularized', 'v4_deeper']
    blend = np.mean([predictions[n] for n in blend_names], axis=0)
    scores = [accuracy_score(y[b], blend[b] >= .5) for _, b in folds]
    rows.append(dict(version='v5_equal_blend', **metrics(y, blend), fold_std=float(np.std(scores)), seconds=0))
    np.savez(out/'v5_equal_blend_oof.npz', prob=blend, fold_accuracy=scores)
    results = pd.DataFrame(rows).sort_values(['accuracy', 'log_loss'], ascending=[False, True])
    results.to_csv(out/'development_results.csv', index=False)
    winner = results.iloc[0].version
    selected = blend_names if winner == 'v5_equal_blend' else [winner]
    (out/'selection.json').write_text(json.dumps(dict(version=winner, components=selected,
        threshold=.5, development_accuracy=float(results.iloc[0].accuracy)), indent=2)+'\n')
    print('SELECTED '+winner, flush=True)


def group_audit(out):
    if (out/'group_audit.json').exists():
        print('Group audit already exists.'); return
    train, _, _ = load_data()
    dev = np.load(out/'split_indices.npz')['development']
    frame = train.iloc[dev].reset_index(drop=True)
    y = frame.Transported.astype(int).to_numpy()
    groups = frame.PassengerId.str.split('_').str[0]
    selected = json.loads((out/'selection.json').read_text())['components']
    prob, scores = np.zeros(len(y)), []
    for fold, (a, b) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(frame, y, groups)):
        assert not set(groups.iloc[a]) & set(groups.iloc[b])
        preds = [fit_predict(CONFIGS[n], frame.iloc[a], y[a], [frame.iloc[b]], SEED+fold)[0][0] for n in selected]
        prob[b] = np.mean(preds, axis=0)
        scores.append(accuracy_score(y[b], prob[b] >= .5))
        print(f'Group audit fold {fold+1}: {scores[-1]:.5f}', flush=True)
    (out/'group_audit.json').write_text(json.dumps(dict(**metrics(y, prob), fold_accuracy=scores,
        interpretation='Development-only stress test on unseen passenger groups; not the competition score.'), indent=2)+'\n')


def finalize(out):
    if (out/'holdout_metrics.json').exists():
        raise RuntimeError('Holdout already evaluated. Do not reuse it to tune versions.')
    train, test, sample = load_data()
    split = np.load(out/'split_indices.npz')
    dev, hold = split['development'], split['holdout']
    selection = json.loads((out/'selection.json').read_text())
    selected = selection['components']
    hold_probs = []
    for name in selected:
        predictions, _, _ = fit_predict(CONFIGS[name], train.iloc[dev], train.Transported.iloc[dev].astype(int), [train.iloc[hold]])
        hold_probs.append(predictions[0])
    probs = np.mean(hold_probs, axis=0)
    y = train.Transported.iloc[hold].astype(int)
    result = dict(version=selection['version'], **metrics(y, probs), n=len(hold),
                  confusion_matrix=confusion_matrix(y, probs >= .5).tolist(),
                  warning='Local holdout accuracy, NOT Kaggle score or ranking.')
    (out/'holdout_metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    pd.DataFrame(dict(PassengerId=train.PassengerId.iloc[hold], target=y,
                      probability=probs, prediction=probs >= .5)).to_csv(out/'holdout_predictions.csv', index=False)
    print('HOLDOUT '+json.dumps(result), flush=True)
    test_probs = []
    for name in selected:
        pred, model, cols = fit_predict(CONFIGS[name], train, train.Transported.astype(int), [test])
        test_probs.append(pred[0])
        if isinstance(model, CatBoostClassifier):
            model.save_model(str(out/(name+'.cbm')))
            pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False).to_csv(out/(name+'_feature_importance.csv'), header=['importance'])
    test_prob = np.mean(test_probs, axis=0)
    submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': test_prob >= .5})
    assert submission.shape == (4277, 2) and submission.PassengerId.equals(sample.PassengerId)
    assert not submission.isna().any().any()
    submission.to_csv(out/'submission.csv', index=False)
    pd.DataFrame({'PassengerId': test.PassengerId, 'probability': test_prob}).to_csv(out/'test_probabilities.csv', index=False)
    print('Submission ready: '+str(out/'submission.csv'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['init', 'develop', 'group-audit', 'finalize'])
    parser.add_argument('--run', default='run_001')
    args = parser.parse_args()
    out = ROOT/'artifacts'/args.run
    {'init': initialize, 'develop': develop, 'group-audit': group_audit, 'finalize': finalize}[args.stage](out)
