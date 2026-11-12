import dataclasses
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import tabpfn_views as candidate


@dataclasses.dataclass
class FakeView:
    preprocess_config: dict
    class_permutation: np.ndarray


class ViewTests(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame(np.zeros((4, 13)), columns=candidate.COMPACT)

    def test_independent_budgets_diagnostics_and_roundtrip(self):
        calls = []
        class FakeModel:
            def __init__(self, spec, n):
                self.n_estimators_ = n
                self.configs_ = [SimpleNamespace(**spec)]
                self.ensemble_configs_ = [FakeView({'name': 'test'}, np.array([0, 1])) for _ in range(n)]
                self.fit_mode = 'fit_preprocessors'
            def fit(self, x, y):
                calls.append(('fit', self.n_estimators_, len(y)))
            def predict_proba(self, query):
                return np.tile([.4, .6], (len(query), 1))
        with patch.object(candidate.architecture, 'verify_checkpoint'), \
             patch.object(candidate.architecture, 'make_model',
                          side_effect=lambda spec, seed, n, cats, flags: FakeModel(spec, n)), \
             patch.object(candidate.architecture, 'release_memory') as cleanup, \
             patch.object(candidate.torch.mps, 'driver_allocated_memory', return_value=0):
            for components in [candidate.CONTROL_COMPONENTS, candidate.COMPONENTS]:
                predictor = candidate.ViewPredictor(self.x, [0, 1, 0, 1], [], 7, components)
                p = predictor.predict_proba(self.x)
                n = components[0]['n_estimators']
                self.assertEqual(predictor.last_resources[0]['actual_resolved_views'], n)
                self.assertEqual(len(predictor.last_resources[0]['ensemble_configs']), n)
                self.assertFalse(any(isinstance(v, FakeModel) for v in vars(predictor).values()))
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder)/'model.joblib'
                    joblib.dump(predictor, path)
                    restored = joblib.load(path)
                    np.testing.assert_array_equal(restored.predict_proba(self.x), p)
                    self.assertEqual(restored.components, components)
            self.assertEqual(calls, [('fit', 8, 4), ('fit', 8, 4), ('fit', 32, 4), ('fit', 32, 4)])
            self.assertEqual(cleanup.call_count, 4)

    def test_resolved_budget_mismatch_fails_closed(self):
        fake = SimpleNamespace(configs_=[SimpleNamespace(**candidate.architecture.CHECKPOINTS['default'])],
                               n_estimators_=8, ensemble_configs_=[None]*8,
                               fit=lambda *args: None)
        with patch.object(candidate.architecture, 'verify_checkpoint'), \
             patch.object(candidate.architecture, 'make_model', return_value=fake), \
             patch.object(candidate.architecture, 'release_memory') as cleanup:
            predictor = candidate.ViewPredictor(self.x, [0, 1, 0, 1], [], 7, candidate.COMPONENTS)
            with self.assertRaisesRegex(RuntimeError, 'Resolved view count'):
                predictor.predict_proba(self.x)
            self.assertFalse(hasattr(predictor, 'last_resources'))
            cleanup.assert_called_once()

    def test_original_v6_dispatch_and_grid(self):
        with patch.object(candidate, 'BASE_FIT', return_value='original') as base:
            self.assertEqual(candidate.fit(candidate.iterate.BASELINE, {}, [], [], [], 7), 'original')
            base.assert_called_once()
        self.assertEqual(set(candidate.iterate.CONFIGS),
                         {candidate.iterate.BASELINE, candidate.CONTROL, candidate.NAME})
        self.assertEqual(candidate.iterate.CONFIGS[candidate.iterate.BASELINE],
                         dict(context=False, model='catboost', **candidate.iterate.CAT))

    def test_protocol_frozen_before_baseline(self):
        order = []
        with patch.object(candidate, 'record_protocol', side_effect=lambda out: order.append('freeze')), \
             patch.object(candidate, 'BASE_EVALUATE', side_effect=lambda *args: order.append('evaluate')):
            candidate.evaluate(candidate.iterate.BASELINE, {}, None, [], 7, Path('/unused'))
        self.assertEqual(order, ['freeze', 'evaluate'])


if __name__ == '__main__':
    unittest.main()
