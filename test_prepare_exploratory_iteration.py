import copy
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
import pandas as pd

import prepare_exploratory_iteration as exporter


class Predictor:
    previous = None
    altered = False
    def __init__(self):
        self.labels = np.zeros(8693, dtype=int)
        self.metadata = {'complete_context': True}
        self.device = 'mps'
    def release(self):
        pass
    def __getstate__(self):
        return {**self.__dict__, 'device': 'cpu'}
    def predict_proba(self, query, **kwargs):
        if self.previous is not None and self.previous() is not None:
            raise AssertionError('First predictor still resident during reload.')
        if self.device != 'mps' or len(query) != 4277:
            raise AssertionError('Full reload inference contract failed.')
        p = np.full(len(query), .6)
        if self.altered:
            p[0] = .4
        return np.column_stack([1-p, p])


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patch_root = patch.object(exporter, 'ROOT', self.root)
        self.patch_root.start()
        self.module = 'mitra_mirror_ensemble'
        self.version = 'v74_Mitra_mirror4'
        self.run = 'run_040'
        self.protocol = f'artifacts/{self.run}/protocol.json'
        self.screen = f'artifacts/{self.run}/screen_results.csv'
        self.path = 'reports/online_iteration_20261116/mirror_decision.json'
        for filename in exporter.OWN_SOURCES+[self.module+'.py']:
            (self.root/filename).write_text('# fixture\n')
        exporter.write(self.root/exporter.AUTH, dict(batch_id=exporter.BATCH,
            competition='spaceship-titanic', maximum_upload_attempts=10,
            weekly_reserve_percent=30, no_duplicate_or_blind_retry_upload=True))
        exporter.write(self.root/self.protocol, {'sources': {
            self.module+'.py': exporter.sha(self.root/(self.module+'.py'))}})
        pd.DataFrame([dict(version=self.version, accuracy=.82)]).to_csv(self.root/self.screen, index=False)
        self.decision = dict(batch_id=exporter.BATCH,
            batch_authorization_sha256=exporter.sha(self.root/exporter.AUTH),
            module=self.module, version=self.version, run=self.run, proposal='proposal_fixture',
            source_protocol_sha256=exporter.sha(self.root/self.protocol),
            evidence_files={self.screen: exporter.sha(self.root/self.screen)},
            exploratory=True, prior_A_eligible=False, prior_S_supported=False,
            rationale='Distinct recipe with practically comparable local evidence.',
            screen_accuracy=.82, threshold=.5, maximum_attempts=1)
        exporter.write(self.root/self.path, self.decision)
        Predictor.previous = None
        Predictor.altered = False

    def tearDown(self):
        self.patch_root.stop()
        self.tmp.cleanup()
        Predictor.previous = None

    def mocked_runtime(self):
        train = pd.DataFrame({'Transported': np.arange(8693)%2})
        test = pd.DataFrame({'PassengerId': [f'{i:04d}_01' for i in range(4277)]})
        features = pd.DataFrame({'Age': np.arange(12970)})
        data = SimpleNamespace(SEEDS=[20261112], dataset=Mock(return_value=(train,test,{False:(features,[])},None)))
        core = SimpleNamespace(environment_lock=lambda: 'fixture==1\n', release_memory=lambda: None, DEADLINE=None)
        candidate = SimpleNamespace(NAME=self.version, mitra=core,
            configure_runtime=Mock(), assert_frozen=Mock())
        def fit(name, matrices, a, y, b, seed):
            self.assertEqual((name,len(a),len(y),len(b)), (self.version,8693,8693,4277))
            self.assertFalse(np.intersect1d(a,b).size)
            model = Predictor()
            Predictor.previous = weakref.ref(model)
            return np.full(4277,.6), model
        candidate.fit = Mock(side_effect=fit)
        return candidate,data

    def execute(self, candidate, data, duplicate=False):
        def validate(path):
            frame = pd.read_csv(path, dtype=str)
            self.assertEqual(frame.shape,(4277,2))
            self.assertTrue(frame.Transported.isin(['True','False']).all())
            return dict(rows=4277,sha256=exporter.sha(path),valid=True)
        with ExitStack() as stack:
            stack.enter_context(patch.object(exporter,'require_committed',return_value='a'*40))
            stack.enter_context(patch.object(exporter.importlib,'import_module',return_value=candidate))
            stack.enter_context(patch.dict(sys.modules,{'iterate':data}))
            stack.enter_context(patch.object(exporter.signal,'signal',return_value=None))
            stack.enter_context(patch.object(exporter.signal,'setitimer'))
            stack.enter_context(patch.object(exporter,'validate_csv',side_effect=validate))
            stack.enter_context(patch.object(exporter,'duplicate_attempts',return_value=['old'] if duplicate else []))
            return exporter.prepare(self.path)

    def test_export_preserves_failed_gate_and_reloads_full_context(self):
        candidate,data = self.mocked_runtime()
        result = self.execute(candidate,data)
        self.assertFalse(result['selection']['A_eligible'])
        self.assertFalse(result['selection']['S_supported'])
        self.assertTrue(result['export_reload_inference_verified'])
        self.assertEqual(result['reload_max_absolute_difference'],0)
        self.assertEqual(result['authorization']['batch_id'],exporter.BATCH)
        self.assertEqual(result['provenance']['decision_sha256'],exporter.sha(self.root/self.path))
        self.assertEqual(exporter.read(self.root/'artifacts/proposal_fixture/authorization.json'),
                         exporter.read(self.root/'reports/proposal_fixture/authorization.json'))
        self.assertEqual(candidate.fit.call_count,1)

    def test_bad_decisions_rejected_before_full_labels(self):
        for change in [dict(prior_A_eligible=True),dict(prior_A_eligible=0),dict(threshold=.49),
                       dict(screen_accuracy=.83),dict(batch_authorization_sha256='changed'),
                       dict(evidence_files={}),dict(exploratory=False)]:
            with self.subTest(change=change):
                exporter.write(self.root/self.path,{**self.decision,**change})
                candidate,data = self.mocked_runtime()
                with self.assertRaises((RuntimeError,ValueError)):
                    self.execute(candidate,data)
                data.dataset.assert_not_called()
                candidate.fit.assert_not_called()

    def test_failed_reload_and_duplicate_remove_upload_artifacts(self):
        for altered,duplicate in [(True,False),(False,True)]:
            with self.subTest(altered=altered,duplicate=duplicate):
                proposal = f'proposal_{altered}_{duplicate}'
                exporter.write(self.root/self.path,{**self.decision,'proposal':proposal})
                Predictor.altered = altered
                candidate,data = self.mocked_runtime()
                with self.assertRaises(RuntimeError):
                    self.execute(candidate,data,duplicate)
                for category in ['reports','artifacts']:
                    out = self.root/category/proposal
                    self.assertEqual(exporter.read(out/'failure.json')['status'], 'failed_incomplete_not_upload_eligible')
                    for filename in ['manifest.json','authorization.json','submission.csv']:
                        self.assertFalse((out/filename).exists())

    def test_source_mismatch_and_uncommitted_decision_fail_closed(self):
        with patch.object(exporter,'require_committed',side_effect=RuntimeError('uncommitted')):
            with self.assertRaisesRegex(RuntimeError,'uncommitted'):
                exporter.decision_evidence(self.path)
        (self.root/(self.module+'.py')).write_text('modified')
        with patch.object(exporter,'require_committed',return_value='a'*40):
            with self.assertRaisesRegex(RuntimeError,'Bound source'):
                exporter.decision_evidence(self.path)


if __name__ == '__main__':
    unittest.main()
