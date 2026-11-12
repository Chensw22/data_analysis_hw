"""Prepare the three unsubmitted A-tier representatives authorized by the user.

Each process imports one original frozen recipe. This never uploads to Kaggle
and never changes the historical screening gates or parameters.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
AUTH = ROOT/'reports/a_tier_batch_20261112/authorization.json'
SPECS = {
    'pfn8': ('run_008', 'TabPFN_v2_compact8', 'tabpfn_refine', 'requirements-tabpfn.lock.txt'),
    'linear_leaf': ('run_027', 'v56_nested_linear_leaf', 'linear_leaf_candidate', None),
    'tabicl': ('run_014', 'v37_TabICL_engineered4', 'tabicl_candidate', 'requirements-tabicl.lock.txt'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_locked_packages(path):
    checked = {}
    for line in Path(path).read_text().splitlines():
        if '==' not in line:
            continue
        name, expected = line.split('==', 1)
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f'Frozen dependency changed: {name}: {expected} -> {actual}')
        checked[name] = actual
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=SPECS)
    args = parser.parse_args()
    source_run, version, module, lock = SPECS[args.model]
    authorization = json.loads(AUTH.read_text())
    if version not in authorization['versions'] or authorization['maximum_upload_attempts'] != 3:
        raise RuntimeError('Outside the current three-file batch authorization.')
    proposal = f'proposal_a_{args.model}'
    out, report = ROOT/'artifacts'/proposal, ROOT/'reports'/proposal
    if out.exists() or report.exists():
        raise RuntimeError('Proposal already exists; do not overwrite or automatically retry.')
    candidate = importlib.import_module(module)
    import iterate
    source = ROOT/'artifacts'/source_run
    iterate.assert_frozen(source)
    protocol = json.loads((source/'protocol.json').read_text())
    if args.model == 'linear_leaf':
        candidate.check_environment(source)
        if json.loads((source/'screen_status.json').read_text())['status'] != 'completed':
            raise RuntimeError('Original complete screening is required.')
        candidate.RUN_OUT = out
        candidate.DEADLINE = time.monotonic()+candidate.STAGE_BUDGET_SECONDS
        environment_evidence = {'full_environment_sha256': protocol['linear_leaf_environment_sha256']}
    else:
        import torch
        torch.set_num_threads(4)
        if not torch.backends.mps.is_available():
            raise RuntimeError('The frozen MPS recipe requires local MPS.')
        environment_evidence = verify_locked_packages(ROOT/lock)
        checkpoint = candidate.candidate.WEIGHTS if args.model == 'pfn8' else candidate.WEIGHTS
        if sha(checkpoint) != protocol['checkpoint_sha256']:
            raise RuntimeError('Frozen checkpoint hash mismatch.')
    score = pd.read_csv(ROOT/'reports'/source_run/'screen_results.csv')
    local = float(score.loc[score.version.eq(version), 'accuracy'].iloc[0])
    out.mkdir(parents=True)
    train, test, matrices, _ = iterate.dataset()
    train_ids, test_ids = np.arange(len(train)), np.arange(len(train), len(train)+len(test))
    y = train.Transported.astype(int).to_numpy()
    print(f'Preparing {version}: full{len(train)} training rows, {len(test)} predictions', flush=True)
    started = time.monotonic()
    probability, model = candidate.fit(version, matrices, train_ids, y, test_ids, iterate.SEEDS[0])
    probability = np.asarray(probability, dtype=float)
    if probability.shape != (len(test),) or not np.isfinite(probability).all() or not ((probability>=0)&(probability<=1)).all():
        raise RuntimeError('Incomplete or invalid probabilities.')
    pd.DataFrame({'PassengerId':test.PassengerId, 'Transported':probability>=.5}).to_csv(out/'submission.csv', index=False)
    pd.DataFrame({'PassengerId':test.PassengerId, 'probability':probability}).to_csv(out/'test_probabilities.csv', index=False)
    if args.model == 'linear_leaf':
        model_path = out/'model.joblib'
        joblib.dump(model, model_path)
        np.testing.assert_allclose(joblib.load(model_path).predict_proba(matrices[False][0].iloc[test_ids])[:,1], probability, rtol=0, atol=1e-12)
        persistence = 'fitted_model_with_verified_reload_predictions'
    else:
        # Foundation inference is reproducible from checkpoint + context + frozen
        # recipe. Persist CPU data, without pickling transient MPS engines.
        model_path = out/'training_context.joblib'
        joblib.dump(dict(x=matrices[False][0].iloc[train_ids], y=y,
            categorical_columns=matrices[False][1], version=version, seed=iterate.SEEDS[0],
            config=iterate.CONFIGS[version], checkpoint_sha256=protocol['checkpoint_sha256']), model_path)
        persistence = 'CPU_training_context_and_frozen_checkpoint_recipe; not_a_serialized_fitted_engine'
    from kaggle_workflow import validate_submission
    validation = validate_submission(out/'submission.csv', ROOT/'data/raw/test.csv')
    selection = dict(version=version, components=[version], source_run=source_run,
        screen_accuracy=local, stage='explicitly_authorized_remaining_A_representatives')
    manifest = dict(**validation, selection=selection, fit_rows=len(train), threshold=.5,
        seconds=time.monotonic()-started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        original_gate=json.loads((source/'gate.json').read_text()),
        explicit_batch_exception=True, authorization_sha256=sha(AUTH),
        prepared_from_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        preparation_source_sha256=sha(__file__), source_protocol_sha256=sha(source/'protocol.json'),
        persistence=persistence, persisted_artifact_sha256=sha(model_path),
        environment_evidence=environment_evidence, upload_attempted=False)
    bound_auth = dict(authorization, scope=f'One upload of {proposal}/{version}',
        file_sha256=validation['sha256'], maximum_upload_attempts_for_this_file=1)
    report.mkdir()
    for filename, payload in [('selection.json',selection),('manifest.json',manifest),('authorization.json',bound_auth)]:
        (out/filename).write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
        shutil.copy2(out/filename,report/filename)
    for path in out.glob('v56*.json'):
        shutil.copy2(path,report/path.name)
    print(json.dumps({'proposal':proposal,'version':version,'seconds':manifest['seconds'],**validation}),flush=True)


if __name__ == '__main__':
    main()
