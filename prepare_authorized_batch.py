"""Export three explicitly authorized below-gate models without changing CV gates.

Run each model in its own frozen environment; GPU exports must be sequential.
This script prepares files only. kaggle_workflow.py performs a single upload.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SPECS = {
    'v44': ('run_019', 'v44_seed_ensemble'),
    'pfn32': ('run_033', 'v65_TabPFN_default32'),
    'mitra2': ('run_038', 'v71_Mitra2_zeroshot'),
}
AUTHORIZATION = ROOT/'reports/online_batch_20261112/authorization.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=SPECS)
    args = parser.parse_args()
    source_run, version = SPECS[args.model]
    authorization = json.loads(AUTHORIZATION.read_text())
    if version not in authorization['versions'] or authorization['maximum_upload_attempts'] != 3:
        raise RuntimeError('Model is outside this explicit three-submission authorization.')
    name = f'proposal_batch_{args.model}'
    out, report = ROOT/'artifacts'/name, ROOT/'reports'/name
    if out.exists() or report.exists():
        raise RuntimeError('Proposal already exists; never overwrite or silently retry.')

    if args.model == 'v44':
        import seed_ensemble as candidate
        import iterate
        iterate.assert_frozen(ROOT/'artifacts'/source_run)
    elif args.model == 'pfn32':
        import tabpfn_views as candidate
        import iterate
        candidate.torch.set_num_threads(4)
        candidate.check_environment(ROOT/'artifacts'/source_run)
        iterate.assert_frozen(ROOT/'artifacts'/source_run)
        candidate.RUN_OUT = out
    else:
        import mitra_candidate as candidate
        candidate.torch.set_num_threads(4)
        budget = candidate.configure_runtime(SimpleNamespace(stage='export', run=source_run,
            query_chunk=None, device='mps', checkpoint_dir=candidate.CHECKPOINT_DIR))
        import iterate
        candidate.torch.mps.set_per_process_memory_fraction(candidate.SETTINGS['mps_memory_fraction'])
        candidate.DEADLINE = time.monotonic()+budget
        candidate.RUN_OUT = out

    # Frozen CV sources/configurations stay untouched and their gate remains false.
    original_gate = json.loads((ROOT/'artifacts'/source_run/'gate.json').read_text())
    screen = pd.read_csv(ROOT/'reports'/source_run/'screen_results.csv')
    local_accuracy = float(screen.loc[screen.version.eq(version), 'accuracy'].iloc[0])
    out.mkdir(parents=True)
    shutil.copy2(AUTHORIZATION, out/'batch_authorization.json')
    train, test, matrices, _ = iterate.dataset()
    train_ids = np.arange(len(train))
    test_ids = np.arange(len(train), len(train)+len(test))
    started = time.monotonic()
    probability, model = candidate.fit(version, matrices, train_ids,
        train.Transported.astype(int).to_numpy(), test_ids, iterate.SEEDS[0])
    probability = np.asarray(probability, dtype=float)
    if probability.shape != (len(test),) or not np.isfinite(probability).all() or not ((probability>=0)&(probability<=1)).all():
        raise RuntimeError('Invalid complete-model prediction.')
    joblib.dump(model, out/'model.joblib')
    # Avoid another full foundation-model inference; their reload contract was
    # tested before CV. Never claim this export has had a second inference pass.
    reload_verified = False
    if args.model == 'v44':
        restored = joblib.load(out/'model.joblib')
        np.testing.assert_allclose(restored.predict_proba(matrices[False][0].iloc[test_ids])[:,1],
                                   probability, atol=1e-12, rtol=0)
        reload_verified = True
    pd.DataFrame({'PassengerId':test.PassengerId, 'Transported':probability>=.5}).to_csv(out/'submission.csv', index=False)
    pd.DataFrame({'PassengerId':test.PassengerId, 'probability':probability}).to_csv(out/'test_probabilities.csv', index=False)
    from kaggle_workflow import validate_submission
    check = validate_submission(out/'submission.csv', ROOT/'data/raw/test.csv')
    selection = dict(version=version, components=[version], source_run=source_run,
        screen_accuracy=local_accuracy, stage='user_authorized_batch_exception')
    (out/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    manifest = dict(**check, selection=selection, fit_rows=len(train), threshold=.5,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-started,
        original_local_gate=original_gate, gate_overridden_by_explicit_batch_authorization=True,
        authorization_sha256=sha(AUTHORIZATION), prepared_from_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        preparation_source_sha256=sha(__file__), source_protocol_sha256=sha(ROOT/'artifacts'/source_run/'protocol.json'),
        model_sha256=sha(out/'model.joblib'), export_reload_inference_verified=reload_verified,
        complete_test_predictions=True, upload_attempted=False)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    bound_authorization = dict(authorization, scope=f'One upload of {name}/{version}',
        file_sha256=check['sha256'], maximum_upload_attempts_for_this_file=1)
    (out/'authorization.json').write_text(json.dumps(bound_authorization, indent=2, ensure_ascii=False)+'\n')
    report.mkdir()
    for filename in ['manifest.json','selection.json','authorization.json']:
        shutil.copy2(out/filename, report/filename)
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
