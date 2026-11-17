"""Ready-only v102: original raw model plus two fixed zero-shot support routes."""
import os
import sys
if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None, 'FALSE'):
    raise RuntimeError('Fresh FALSE-only process required.')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'FALSE'

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import time

import joblib
import numpy as np
import pandas as pd
import torch

import audit_top5_references as audit
import conditional_context_error_probe as diagnostic
import error_rescue_feature_probe as previous
import exaone_validation_data as data
import mitra_candidate as base
import mitra_counts_fixed15_online_first as counts
import prepare_v84_online_first as common

ROOT = Path(__file__).resolve().parent
NAME = 'v102_Mitra_conditional_context_ready'
PROPOSAL = 'proposal_ready_conditional_context'
POLICY = 'reports/error_targeted_20261117/context_export_policy.json'
SEED, BUDGET = 20261112, 1200
sha, read, write = common.sha, common.read, common.write
if __name__ == '__main__':
    sys.modules['prepare_context_ready'] = sys.modules[__name__]


class ConditionalContextPredictor:
    """CPU-persisted components, with at most one live MPS engine."""
    def __init__(self, raw, locals_, metadata):
        if set(locals_) != set(diagnostic.STATES):
            raise ValueError('Both fixed routes required.')
        self.raw, self.locals, self.metadata = raw, locals_, metadata
        self.device = 'cpu'
        self.release()

    def release(self):
        for model in [self.raw, *self.locals.values()]:
            model.release()
        base.release_memory()

    def predict_proba(self, x):
        base.check_deadline()
        try:
            self.raw.device = self.device
            p = self.raw.predict_proba(x)[:, 1].copy()
        finally:
            self.raw.release()
            base.release_memory()
        for state in diagnostic.STATES:
            mask = diagnostic.route(x, state)
            if mask.any():
                try:
                    self.locals[state].device = self.device
                    p[mask] = self.locals[state].predict_proba(x.loc[mask])[:, 1]
                finally:
                    self.locals[state].release()
                    base.release_memory()
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise RuntimeError('Invalid combined probability.')
        return np.column_stack([1-p, p])

    def __getstate__(self):
        self.release()
        state = dict(self.__dict__)
        state['device'] = 'cpu'
        return state


ConditionalContextPredictor.__module__ = 'prepare_context_ready'


def verify_model(model, x, y, ids):
    if (model._engine is not None or model.device != 'cpu' or model.query_chunk != 2318
            or model.preprocessor.columns != base.COMPACT
            or not np.array_equal(model.labels, y) or len(model.support) != len(ids)
            or model.metadata.get('checkpoint_sha256') != base.WEIGHT_SHA
            or model.metadata.get('seed') != SEED
            or model.metadata.get('training_ids_sha256') != base.ids_hash(ids)
            or model.metadata.get('selected_steps') != 0):
        raise RuntimeError('Portable zero-shot full-support component identity differs.')
    np.testing.assert_array_equal(model.support, model.preprocessor.transform(x))



def raw_evidence(policy):
    folder = ROOT/policy['raw_proposal_directory']
    public = ROOT/'reports'/folder.name/'manifest.json'
    local = folder/'manifest.json'
    previous.committed([str(public.relative_to(ROOT))])
    if sha(public) != sha(local):
        raise RuntimeError('Raw producer completed manifest/report mismatch.')
    manifest = read(local)
    component = manifest.get('raw_component', {})
    expected = {'raw_model.joblib','raw_probabilities.npz','raw_reload_probabilities.npz','raw_metadata.json'}
    if (manifest.get('status') != 'complete' or component.get('training_rows') != 8693
            or component.get('rows') != 4277 or component.get('raw_reload_max_abs_diff') != 0
            or set(component.get('files', {})) != expected):
        raise RuntimeError('Completed full raw component and exact reload evidence required.')
    files = {str(p.relative_to(ROOT)):sha(p) for p in (local, public)}
    for name, digest in component['files'].items():
        path = folder/name
        if sha(path) != digest:
            raise RuntimeError('Raw component changed: '+name)
        files[str(path.relative_to(ROOT))] = digest
    for name, digest in manifest['source_hashes'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Raw producer source/evidence changed: '+name)
        files[name] = digest
    return folder, manifest, files


def export():
    out, report = ROOT/'artifacts'/PROPOSAL, ROOT/'reports'/PROPOSAL
    if out.exists() or report.exists():
        raise RuntimeError('No overwrite or resume.')
    policy = read(ROOT/POLICY)
    if (policy['version'] != NAME or policy['allow_full_export'] is not True
            or policy['allow_direct_submit'] is not False or policy['seed'] != SEED):
        raise RuntimeError('Exact ready-only policy required.')
    original, lock = counts.verify_reference()
    raw_dir, raw_manifest, raw_files = raw_evidence(policy)
    names = sorted(set(previous.OWN + [Path(__file__).name, POLICY,
        'conditional_context_error_probe.py', diagnostic.PLAN, 'kaggle_workflow.py',
        'reports/error_targeted_20261117/context_result.json',
        'reports/error_targeted_20261117/context_root_audit.json']
        + [n for n in original['source_hashes'] if n.startswith('third_party/mitra/')]))
    commit = previous.committed(names)
    sources = {n:sha(ROOT/n) for n in names}
    sources.update(raw_files)
    for path in ('data/raw/train.csv', 'data/raw/test.csv', 'data/raw/sample_submission.csv'):
        sources[path] = sha(ROOT/path)
    out.mkdir(parents=True); report.mkdir(parents=True)
    start = time.monotonic(); base.DEADLINE = start+BUDGET
    def timeout(*_):
        raise TimeoutError('Conditional-context ready export exceeded1200s.')
    handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, BUDGET)
    raw = combined = restored = active = None
    locals_ = {}
    try:
        protocol = dict(version=NAME, source_commit=commit, source_hashes=sources,
            policy=policy, checkpoint_sha256=base.WEIGHT_SHA,
            environment_sha256=original['environment_sha256'], budget_seconds=BUDGET)
        for directory in (out, report):
            write(directory/'protocol.json', protocol)
        (out/'environment.lock.txt').write_text(lock)
        counts.require_false()
        if not torch.backends.mps.is_available():
            raise RuntimeError('Original MPS required; no fallback.')
        torch.set_num_threads(4); torch.mps.set_per_process_memory_fraction(.65)
        # Full training labels are authorized for this final export only.
        train = pd.read_csv(ROOT/'data/raw/train.csv')
        test = pd.read_csv(ROOT/'data/raw/test.csv')
        if len(train) != 8693 or len(test) != 4277:
            raise RuntimeError('Full8693/4277 required.')
        roster = pd.concat([train.drop(columns='Transported'), test], ignore_index=True)
        frame = data.compact_features(roster)
        x, q = frame.iloc[:8693], frame.iloc[8693:]
        y, ids = train.Transported.astype(np.int64).to_numpy(), np.arange(8693, dtype=np.int64)
        expected_ids = test.PassengerId.astype(str).to_numpy()
        with np.load(raw_dir/'raw_probabilities.npz', allow_pickle=False) as cached:
            np.testing.assert_array_equal(cached['train_ids'], ids)
            np.testing.assert_array_equal(cached['test_ids'], q.index.to_numpy())
            raw_probability = cached['test_raw'].copy()
        with np.load(raw_dir/'raw_reload_probabilities.npz', allow_pickle=False) as cached_reload:
            np.testing.assert_array_equal(raw_probability, cached_reload['test_raw'])
        first = raw_probability.copy()
        if first.shape != (4277,) or not np.isfinite(first).all() or np.any((first<0)|(first>1)):
            raise RuntimeError('Invalid complete raw cache.')
        raw = joblib.load(raw_dir/'raw_model.joblib')
        verify_model(raw, x, y, ids)
        route_meta = {}; routed = np.zeros(4277, dtype=bool)
        for state in diagnostic.STATES:
            sm, qm = diagnostic.route(x, state), diagnostic.route(q, state)
            if not sm.any() or not qm.any() or len(np.unique(y[sm])) != 2:
                raise RuntimeError('Both complete fixed routes required; no fallback.')
            sid = ids[sm]
            active = base.fit_mitra(x.loc[sid], y[sm], sid, SEED, 'mps', recipe='zero-shot',
                query_chunk=2318, log_path=out/f'local_{state}_fit.json')
            p = active.predict_proba(q.loc[qm])[:,1]
            first[qm] = p; routed |= qm
            active.release(); active.device = 'cpu'; base.release_memory()
            joblib.dump(active, out/f'local_{state}.joblib')
            active = None; base.release_memory()
            active = joblib.load(out/f'local_{state}.joblib')
            verify_model(active, x.loc[sid], y[sm], sid)
            active.device = 'mps'
            again_local = active.predict_proba(q.loc[qm])[:,1]
            if not np.array_equal(p, again_local):
                raise RuntimeError('Local complete-route reload differs; no tolerance rescue.')
            active.release(); active.device = 'cpu'; base.release_memory()
            route_meta[state] = dict(training_rows=len(sid), query_rows=int(qm.sum()),
                training_ids_sha256=base.ids_hash(sid), labels_sha256=base.ids_hash(y[sm]),
                query_ids_sha256=base.ids_hash(q.index[qm].to_numpy()),
                feature_count=active.preprocessor.n_features,
                model_sha256=sha(out/f'local_{state}.joblib'), reload_max_absolute_difference=0.)
            locals_[state] = active; active = None
        np.testing.assert_array_equal(first[~routed], raw_probability[~routed])
        metadata = dict(version=NAME, training_rows=8693, seed=SEED, routes=route_meta,
            raw_model_sha256=sha(raw_dir/'raw_model.joblib'), checkpoint_sha256=base.WEIGHT_SHA,
            training_ids_sha256=base.ids_hash(ids), labels_sha256=base.ids_hash(y),
            prediction_passenger_ids=expected_ids.tolist(), conditional_routes=list(diagnostic.STATES))
        combined = ConditionalContextPredictor(raw, locals_, metadata)
        raw = None; locals_ = {}
        joblib.dump(combined, out/'model.joblib')
        combined.release(); combined = None; base.release_memory()
        restored = joblib.load(out/'model.joblib')
        if restored.device != 'cpu' or set(restored.locals) != set(diagnostic.STATES):
            raise RuntimeError('Combined portable recipe differs.')
        verify_model(restored.raw, x, y, ids)
        for state in diagnostic.STATES:
            sm = diagnostic.route(x, state)
            verify_model(restored.locals[state], x.loc[ids[sm]], y[sm], ids[sm])
        restored.device = 'mps'
        again = restored.predict_proba(q)[:,1]
        difference = common.compare(first.tolist(), again.tolist())
        if difference != 0:
            raise RuntimeError('Complete4277 combined reload not bitwise equal.')
        restored.release(); restored = None; base.release_memory()
        pd.DataFrame(dict(PassengerId=expected_ids, probability=first, reload_probability=again)).to_csv(out/'test_probabilities.csv', index=False)
        pd.DataFrame(dict(PassengerId=expected_ids, Transported=first>=.5)).to_csv(out/'submission.csv', index=False)
        import kaggle_workflow
        check = kaggle_workflow.validate_submission(out/'submission.csv', ROOT/'data/raw/test.csv')
        for path in (ROOT/'reports').rglob('online_attempts/*.json'):
            if read(path).get('file_sha256') == check['sha256']:
                raise RuntimeError('Identical CSV already attempted.')
        for path, expected in sources.items():
            if sha(ROOT/path) != expected:
                raise RuntimeError('Frozen evidence/source changed: '+path)
        if base.environment_lock() != lock:
            raise RuntimeError('Environment changed.')
        counts.require_false(); base.verify_checkpoint(); base.check_deadline()
        selection = dict(version=NAME, components=['raw_elsewhere','Earth_Cryo_True','Earth_Cryo_False'],
            threshold=.5, stage='ready_pending_primary_account_quota', A_eligible=None, S_supported=False,
            optimizer_steps=0, state_selection=False, online_submission_authorized=False)
        manifest = dict(**check, status='complete', version=NAME, proposal=PROPOSAL, training_rows=8693,
            source_commit=commit, source_hashes=sources, policy_path=POLICY, policy_sha256=sha(ROOT/POLICY),
            environment_sha256=original['environment_sha256'], checkpoint_sha256=base.WEIGHT_SHA,
            model_sha256=sha(out/'model.joblib'), test_probabilities_sha256=sha(out/'test_probabilities.csv'),
            export_reload_inference_verified=True, reload_rows=4277, reload_max_absolute_difference=difference,
            reload_threshold_decisions_equal=True, conditional_routes=route_meta,
            reused_raw_model=True, unchanged_nonroute_rows=int((~routed).sum()),
            seconds=time.monotonic()-start, completed_at_utc=datetime.now(timezone.utc).isoformat())
        write(out/'reload_metadata.json', dict(rows=4277, max_absolute_difference=difference, threshold_equal=True))
        for name, value in [('selection.json',selection), ('manifest.json',manifest)]:
            for directory in (out,report):write(directory/name,value)
        print(json.dumps(dict(status='ready_not_upload_authorized', proposal=PROPOSAL, reload_difference=difference)), flush=True)
    except BaseException as error:
        for directory in (out,report):
            for name in ('manifest.json','selection.json','submission.csv'):(directory/name).unlink(missing_ok=True)
            write(directory/'failure.json',dict(status='failed_not_upload_eligible', error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        for model in [active, raw, combined, restored, *locals_.values()]:
            if model is not None:model.release()
        base.release_memory()
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)


if __name__ == '__main__':
    export()
