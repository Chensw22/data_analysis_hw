import signal
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import joblib

import nested_conditional_fusion as candidate


class FixedMeta:
    def __init__(self, p):
        self.p = p
    def predict_proba(self, x):
        return np.tile([1-self.p, self.p], (len(x), 1))


class FakeBase:
    def __init__(self, p):
        self.p = p
    def predict_proba(self, x):
        return np.tile([1-self.p, self.p], (len(x), 1))


class NestedFusionTests(unittest.TestCase):
    def setUp(self):
        candidate.begin_stage('test-run')

    def test_inner_oof_is_disjoint_complete_and_shared_between_metas(self):
        x = pd.DataFrame({'CryoSleep': ['True', 'False', '__MISSING__']*12})
        matrices = {False: (x, ['CryoSleep'])}
        train, query = np.arange(24), np.arange(24, 36)
        labels = np.arange(24) % 2
        calls = []
        def fake_pair(matrices, train_idx, y, predict_idx, seed, keep=False):
            train_idx, predict_idx = np.asarray(train_idx), np.asarray(predict_idx)
            self.assertFalse(np.intersect1d(train_idx, predict_idx).size)
            self.assertFalse(np.intersect1d(train_idx, query).size)
            np.testing.assert_array_equal(y, train_idx % 2)
            calls.append((train_idx.copy(), predict_idx.copy(), keep))
            p = np.column_stack([.2+.01*predict_idx, .7-.01*predict_idx])
            return p, ('cat', 'pfn') if keep else None
        with patch.object(candidate, 'pair_fit', side_effect=fake_pair):
            first = candidate.nested_payload(matrices, train, labels, query, 17)
            second = candidate.nested_payload(matrices, train, labels, query, 17)
        self.assertIs(first, second)
        self.assertEqual(len(calls), 4)
        covered = np.concatenate([entry[1] for entry in calls[:3]])
        np.testing.assert_array_equal(np.sort(covered), train)
        np.testing.assert_allclose(first['oof'][:, 0], .2+.01*train)
        self.assertEqual(first['global_meta'][0].n_samples_seen_, 24)
        self.assertEqual(first['conditional_meta'][0].n_samples_seen_, 24)

    def test_fingerprint_rejects_overlap_and_changes_with_labels_seed_query_run(self):
        x = pd.DataFrame({'CryoSleep': ['True']*8})
        matrices = {False: (x, ['CryoSleep'])}
        original = candidate.fingerprint(matrices, [0, 1], [0, 1], [2, 3], 7)
        for indices, labels, query, seed in [([0, 1], [1, 0], [2, 3], 7),
                ([0, 1], [0, 1], [2, 3], 8), ([0, 1], [0, 1], [4, 5], 7),
                ([0, 2], [0, 1], [4, 5], 7)]:
            self.assertNotEqual(original, candidate.fingerprint(matrices, indices, labels, query, seed))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            candidate.fingerprint(matrices, [0, 1], [0, 1], [1, 2], 7)
        changed = x.copy()
        changed.loc[2, 'CryoSleep'] = 'False'
        self.assertNotEqual(original, candidate.fingerprint(
            {False: (changed, ['CryoSleep'])}, [0, 1], [0, 1], [2, 3], 7))
        candidate.begin_stage('other-run')
        self.assertNotEqual(original, candidate.fingerprint(matrices, [0, 1], [0, 1], [2, 3], 7))

    def test_signed_disagreement_and_exact_unknown_fallback(self):
        raw = pd.DataFrame({'CryoSleep': ['True', 'False', '__MISSING__', None]})
        p = np.tile([.25, .75], (4, 1))
        features = candidate.meta_features(p, raw, True)
        np.testing.assert_allclose(features[:, 2], [2*np.log(3), -2*np.log(3), 0., 0.])
        result = candidate.combine(FixedMeta(.123456789), FixedMeta(.8), p, raw, True)
        np.testing.assert_array_equal(result[2:], FixedMeta(.123456789).predict_proba(p[2:]))
        np.testing.assert_array_equal(result[:2, 1], [.8, .8])

    def test_budget_expiry_rejects_work_and_alarm_interrupts(self):
        candidate.begin_stage('expired', seconds=-1)
        with self.assertRaises(candidate.StageTimeout):
            candidate.check_budget()
        old = signal.signal(signal.SIGALRM, candidate.alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, .02)
        try:
            with self.assertRaises(candidate.StageTimeout):
                time.sleep(.1)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    def test_full_fusion_artifact_roundtrip_with_real_fitted_metas(self):
        raw = pd.DataFrame({'CryoSleep': ['True', 'False', '__MISSING__', None]*6})
        p = np.column_stack([np.linspace(.1, .9, 24), np.linspace(.8, .2, 24)])
        y = np.arange(24) % 2
        payload = dict(models=(FakeBase(.2), FakeBase(.7)),
            global_meta=candidate.fit_meta(p, raw, y, False, 7),
            conditional_meta=candidate.fit_meta(p, raw, y, True, 7))
        model = candidate.FusionPredictor(payload, True)
        expected = model.predict_proba(raw)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'fusion.joblib'
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(restored.predict_proba(raw), expected)

    def test_failed_stage_removes_only_new_outputs_and_closes_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            for name in ['submission.csv', 'test_probabilities.csv', 'selection.json']:
                (out/name).write_text('test')
            candidate.invalidate_stage(out, 'export', {'selection.json', 'submission.csv'},
                                       candidate.StageTimeout('budget'))
            self.assertTrue((out/'selection.json').exists())
            self.assertTrue((out/'submission.csv').exists())
            self.assertFalse((out/'test_probabilities.csv').exists())
            self.assertFalse(json.loads((out/'gate.json').read_text())['passed'])

    def test_repeated_confirm_preserves_preexisting_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root/'artifacts'/'existing'
            out.mkdir(parents=True)
            gate = out/'gate.json'
            gate.write_text('{"passed": true}')
            with patch.object(candidate.iterate, 'ROOT', root), patch.object(
                    candidate.sys, 'argv', ['candidate', 'confirm', '--run', 'existing']):
                with self.assertRaisesRegex(RuntimeError, 'already recorded'):
                    candidate.main()
            self.assertEqual(gate.read_text(), '{"passed": true}')
            self.assertFalse((out/'nested_fusion_stage_status.json').exists())


if __name__ == '__main__':
    unittest.main()
