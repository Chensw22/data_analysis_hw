"""Repeated-development reference audit; never an independent holdout or gate.

Run each reference in a fresh process and its historical environment:
  .venv/bin/python audit_top5_references.py v44
  .venv-mitra/bin/python audit_top5_references.py mitra
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import shutil
import signal
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT/'artifacts/top5_reference_audit'
REPORT_ROOT = ROOT/'reports/top5_reference_audit'
DEV_HASH = '617a0c9b92c1cc61f803ae60719c48e840fb3985c370a4ba7207a4e6c97aa071'
REFERENCES = {'v44': ('run_019', 'v44_seed_ensemble'), 'mitra': ('run_038', 'v71_Mitra2_zeroshot')}
SPLITS = [('random', 20261113), ('random', 20261114), ('group', 20261112)]
DISCLOSURE = ('Repeated cross-validation on the same previously used6954 development rows; '
              'stability evidence, not independent holdout evidence, model selection, or a submission gate.')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_hash(rows):
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()


def string_hash(values):
    return hashlib.sha256(json.dumps(list(map(str, values)), separators=(',', ':')).encode()).hexdigest()


def json_write(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')
    temporary.replace(path)


def directory_hashes(directory):
    return {str(path.relative_to(directory)): file_hash(path) for path in sorted(Path(directory).rglob('*')) if path.is_file()}


def ensure_history_unchanged(directory, expected):
    if directory_hashes(directory) != expected:
        raise RuntimeError('Historical run changed during audit; do not treat audit as isolated.')


def assert_fresh_model_process(modules=None):
    modules = sys.modules if modules is None else modules
    foreign = {'seed_ensemble', 'contrast_features', 'spatial_expert', 'mitra_candidate',
               'autogluon_candidate', 'model_families'} & set(modules)
    if foreign:
        raise RuntimeError(f'Use a fresh process per reference; existing wrappers: {sorted(foreign)}')


def check_historical_sources(protocol):
    for relative, expected in protocol['sources'].items():
        if file_hash(ROOT/relative) != expected:
            raise RuntimeError(f'Historical frozen source mismatch: {relative}')


def check_v44_dependencies():
    # Exact versions of every originally locked dependency; extra packages do
    # not substitute any locked dependency. Record the full current environment.
    for line in (ROOT/'artifacts/run_019/requirements.lock.txt').read_text().splitlines():
        name, version = line.split('==')
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f'v44 historical dependency differs: {name}')


def initialize_reference(reference, out):
    assert_fresh_model_process()
    run, version = REFERENCES[reference]
    history = ROOT/'artifacts'/run
    protocol = json.loads((history/'protocol.json').read_text())
    check_historical_sources(protocol)
    if reference == 'v44':
        check_v44_dependencies()
        # Execute the actual frozen top-level wrapper without creating pyc in
        # the historical run. Its dependencies are hash-identical current files.
        source = history/'seed_ensemble.py'
        if file_hash(source) != protocol['sources']['seed_ensemble.py']:
            raise RuntimeError('Frozen v44 snapshot differs from its manifest.')
        wrapper = types.ModuleType('seed_ensemble')
        wrapper.__file__ = str(source)
        wrapper.__package__ = ''
        sys.modules['seed_ensemble'] = wrapper
        exec(compile(source.read_text(), str(source), 'exec'), wrapper.__dict__)
        framework = sys.modules['iterate']
        framework.assert_frozen(history)
        fit = wrapper.fit
    else:
        wrapper = importlib.import_module('mitra_candidate')
        # Reuse source, environment and runtime validation only. Never call
        # historical confirm(), which is a model-selection gate, not this audit.
        wrapper.configure_runtime(SimpleNamespace(stage='confirm', run='run_038',
            device='mps', query_chunk=None, checkpoint_dir=wrapper.CHECKPOINT_DIR))
        wrapper.RUN_OUT = out  # Redirect every training log before the first fit.
        wrapper.DEADLINE = None
        wrapper.torch.set_num_threads(4)
        if not wrapper.torch.backends.mps.is_available():
            raise RuntimeError('Frozen Mitra MPS backend unavailable; no CPU fallback.')
        wrapper.torch.mps.set_per_process_memory_fraction(wrapper.SETTINGS['mps_memory_fraction'])
        framework, fit = wrapper.FRAMEWORK, wrapper.fit
    return framework, wrapper, fit, protocol, version


def read_development_only(train_path, test_path, development):
    """Read all permitted attributes but parse target values only for dev rows."""
    development = np.asarray(development, dtype=np.int64)
    if np.any(np.diff(development) <= 0):
        raise ValueError('Development row indices must be sorted and unique.')
    selected = set(map(int, development))
    train = pd.read_csv(train_path, usecols=lambda name: name != 'Transported')
    test = pd.read_csv(test_path, usecols=lambda name: name != 'Transported')
    targets = pd.read_csv(train_path, usecols=['PassengerId', 'Transported'],
        skiprows=lambda line_number: line_number > 0 and line_number-1 not in selected)
    if len(targets) != len(development) or not np.array_equal(
            targets.PassengerId.astype(str), train.PassengerId.iloc[development].astype(str)):
        raise RuntimeError('Development target row alignment failed.')
    y = targets.Transported.astype(int).to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Development labels must be binary.')
    return train, test, y


def load_audit_data(framework):
    split_file = ROOT/'artifacts/run_002/split_indices.npz'
    with np.load(split_file) as split:
        # Never read the holdout label array or reconstruct the partition using
        # full-train target stratification. The frozen dev membership is enough.
        dev = np.sort(split['development']).astype(np.int64)
    if len(dev) != 6954 or row_hash(dev) != DEV_HASH:
        raise RuntimeError('Development membership differs from historical6954 rows.')
    train, test, y = read_development_only(ROOT/'data/raw/train.csv', ROOT/'data/raw/test.csv', dev)
    roster = pd.concat([train, test], ignore_index=True)
    matrices = {False: framework.build_features(roster, False)}
    if len(matrices[False][0].columns) != 30:
        raise RuntimeError('Original engineered_no_ids input changed.')
    groups = train.PassengerId.iloc[dev].astype(str).str.split('_').str[0].to_numpy()
    passengers = train.PassengerId.iloc[dev].astype(str).to_numpy()
    return matrices, dev, y, groups, passengers


def audit_splits(kind, seed, y, groups):
    if kind == 'random':
        folds = StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    elif kind == 'group':
        folds = StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups)
    else:
        raise ValueError(kind)
    covered = np.zeros(len(y), dtype=int)
    result = []
    for a, b in folds:
        if np.intersect1d(a, b).size or len(a)+len(b) != len(y):
            raise RuntimeError('Invalid fold membership.')
        if kind == 'group' and set(groups[a]) & set(groups[b]):
            raise RuntimeError('Passenger group crosses training/validation boundary.')
        covered[b] += 1
        result.append((a, b))
    if not np.all(covered == 1):
        raise RuntimeError('OOF rows must be covered exactly once.')
    return result


def original_screen_reference(reference, version):
    run = REFERENCES[reference][0]
    folder = ROOT/'artifacts'/run
    table_path = folder/'screen_results.csv'
    table = pd.read_csv(table_path)
    row = table.loc[table.version.eq(version)].iloc[0].to_dict()
    oof = folder/f'{version}_seed20261112.npz'
    with np.load(oof) as stored:
        if stored['prob'].shape != (6954,):
            raise RuntimeError('Historical screen OOF length mismatch.')
        folds = stored['fold_accuracy'].tolist()
    return dict(kind='historical_screen_reference_not_recomputed', seed=20261112,
        recorded_metrics=row, recorded_fold_accuracy=folds,
        csv_path=str(table_path.relative_to(ROOT)), csv_sha256=file_hash(table_path),
        oof_path=str(oof.relative_to(ROOT)), oof_sha256=file_hash(oof))


def execute_audit(reference, budget_seconds):
    run, version = REFERENCES[reference]
    out, report = ARTIFACT_ROOT/reference, REPORT_ROOT/reference
    history = ROOT/'artifacts'/run
    before = directory_hashes(history)
    out.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def timeout(*_):
        raise TimeoutError('Reference audit budget exceeded; incomplete stability evidence.')
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, budget_seconds)
    try:
        framework, wrapper, fit, protocol, version = initialize_reference(reference, out)
        matrices, dev, y, groups, passengers = load_audit_data(framework)
        environment = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}
        source_hashes = {name: file_hash(ROOT/name) for name in
                         ['audit_top5_references.py', 'test_audit_top5_references.py']}
        own_sources = out/'source'
        own_sources.mkdir()
        for name in source_hashes:
            shutil.copy2(ROOT/name, own_sources/name)
        frozen = dict(reference=reference, version=version, historical_run=run,
            historical_protocol_sha256=file_hash(history/'protocol.json'),
            historical_sources=protocol['sources'], historical_configs=protocol['configs'],
            audit_sources=source_hashes, environment=environment, python=sys.version,
            split_recipe=SPLITS, fold_fit_seed='split_seed+zero_based_fold; frozen model seed offsets preserved',
            development_rows=len(dev), development_ids_sha256=row_hash(dev), group_ids_sha256=string_hash(groups),
            passenger_ids_sha256=string_hash(passengers), budget_seconds=budget_seconds,
            disclosure=DISCLOSURE, target_policy='Only frozen development target rows parsed; old holdout attributes only.',
            log_output=str(out.relative_to(ROOT)))
        if reference == 'mitra':
            frozen.update(checkpoint_sha256=wrapper.WEIGHT_SHA, runtime=protocol['mitra_runtime'])
        json_write(out/'protocol.json', frozen)
        pd.DataFrame(dict(row_index=dev, PassengerId=passengers, GroupId=groups)).to_csv(out/'development_rows.csv', index=False)
        original = original_screen_reference(reference, version)
        rows = []
        for kind, seed in SPLITS:
            key = f'{kind}_{seed}'
            prob, assignment = np.empty(len(y)), np.empty(len(y), dtype=np.int8)
            fold_logs = []
            split_start = time.monotonic()
            for fold, (a, b) in enumerate(audit_splits(kind, seed, y, groups)):
                tick = time.monotonic()
                train_ids, predict_ids = dev[a], dev[b]
                if not set(train_ids).issubset(set(dev)) or np.intersect1d(train_ids, predict_ids).size:
                    raise RuntimeError('Training boundary violation.')
                prediction, model = fit(version, matrices, train_ids, y[a], predict_ids, seed+fold)
                prediction = np.asarray(prediction)
                if prediction.shape != (len(b),) or not np.isfinite(prediction).all():
                    raise RuntimeError('Invalid fold predictions.')
                prob[b], assignment[b] = prediction, fold
                metadata = dict(kind=kind, split_seed=seed, fold=fold,
                    model_fit_seed=seed+fold, train_rows=len(a), validation_rows=len(b),
                    train_ids_sha256=row_hash(train_ids), validation_ids_sha256=row_hash(predict_ids),
                    group_overlap_count=len(set(groups[a]) & set(groups[b])),
                    accuracy=float(accuracy_score(y[b], prediction >= .5)),
                    log_loss=float(log_loss(y[b], prediction, labels=[0,1])),
                    fit_and_predict_seconds=time.monotonic()-tick)
                if reference == 'v44':
                    metadata.update(component_seeds=[seed+fold+offset for offset in [0,100,200]],
                        component_catboost=[dict(tree_count=component.model.base_model.tree_count_,
                            params=component.model.base_model.get_all_params()) for component in model.models],
                        timing_note='Frozen API returns fit+prediction together; no separate prediction timing available.')
                else:
                    metadata['training_metadata'] = model.metadata
                    model.release()
                del model
                fold_logs.append(metadata)
                json_write(out/f'{key}_fold{fold}.json', metadata)
                np.savez(out/f'{key}_fold{fold}_rows.npz', train_ids=train_ids, validation_ids=predict_ids)
                print(f'{reference} {key} fold{fold+1} accuracy={metadata["accuracy"]:.8f} seconds={metadata["fit_and_predict_seconds"]:.2f}', flush=True)
            np.savez(out/f'{key}.npz', prob=prob, fold_id=assignment, dev_ids=dev,
                     group_ids=groups.astype(str), passenger_ids=passengers.astype(str),
                     fold_accuracy=np.array([row['accuracy'] for row in fold_logs]),
                     seconds=time.monotonic()-split_start)
            rows.append(dict(kind=kind, seed=seed, accuracy=float(accuracy_score(y, prob>=.5)),
                log_loss=float(log_loss(y, prob, labels=[0,1])),
                fold_accuracy=[row['accuracy'] for row in fold_logs],
                seconds=time.monotonic()-split_start, oof_path=str((out/f'{key}.npz').relative_to(ROOT)),
                oof_sha256=file_hash(out/f'{key}.npz')))
            json_write(out/'progress.json', dict(status='incomplete', completed_splits=rows, disclosure=DISCLOSURE))
        ensure_history_unchanged(history, before)
        summary = dict(status='complete', reference=reference, version=version,
            disclosure=DISCLOSURE, development_ids_sha256=row_hash(dev), group_ids_sha256=string_hash(groups),
            original_screen=original, audits=rows, total_seconds=time.monotonic()-started,
            source_manifest=str((out/'protocol.json').relative_to(ROOT)),
            historical_run_unchanged=True, no_old_holdout_targets_used=True)
        json_write(out/'summary.json', summary)
        json_write(report/'summary.json', summary)
        pd.DataFrame([{k:v for k,v in row.items() if k!='fold_accuracy'} for row in rows]).to_csv(report/'metrics.csv', index=False)
        print(json.dumps(summary), flush=True)
    except Exception as error:
        json_write(out/'failure.json', dict(status='failed_inconclusive', error=repr(error),
            seconds=time.monotonic()-started, disclosure=DISCLOSURE))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        ensure_history_unchanged(history, before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', choices=REFERENCES)
    parser.add_argument('--budget-seconds', type=int, default=2700)
    args = parser.parse_args()
    if args.budget_seconds < 1:
        parser.error('Positive audit budget required.')
    execute_audit(args.reference, args.budget_seconds)


if __name__ == '__main__':
    main()
