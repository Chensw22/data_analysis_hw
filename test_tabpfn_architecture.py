import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import tabpfn_architecture as candidate


class ArchitectureTests(unittest.TestCase):
    def test_checkpoint_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'fake.ckpt'
            path.write_bytes(b'wrong')
            with patch.object(candidate, 'checkpoint_path', return_value=path):
                with self.assertRaisesRegex(RuntimeError, 'SHA256'):
                    candidate.verify_checkpoint(dict(filename='fake.ckpt', bytes=5, sha256='bad'))

    def test_family_weights_sequential_execution_and_reload(self):
        x = pd.DataFrame(np.zeros((4, 13)), columns=candidate.COMPACT)
        calls = []
        probs = {'default': .2, 'gn2p4bpt': .4, 'vutqq28w': .6, 'znskzxi4': .8}
        class FakeModel:
            def __init__(self, spec):
                self.spec = spec
                self.configs_ = [SimpleNamespace(**spec)]
            def fit(self, context, y):
                self.rows = len(y)
            def predict_proba(self, query):
                key = next(k for k, v in candidate.CHECKPOINTS.items() if v['sha256'] == self.spec['sha256'])
                calls.append(key)
                return np.tile([1-probs[key], probs[key]], (len(query), 1))
        with patch.object(candidate, 'verify_checkpoint'), \
             patch.object(candidate, 'make_model', side_effect=lambda spec, *args: FakeModel(spec)), \
             patch.object(candidate, 'release_memory', side_effect=lambda: calls.append('release')), \
             patch.object(candidate.torch.mps, 'driver_allocated_memory', return_value=0):
            predictor = candidate.ArchitecturePredictor(x, [0, 1, 0, 1], [], 7, candidate.ENSEMBLE)
            p = predictor.predict_proba(x)
            np.testing.assert_allclose(p[:, 1], .4)  # .5*.2 + (.4+.6+.8)/6
            self.assertEqual(calls, ['default', 'release', 'gn2p4bpt', 'release',
                                     'vutqq28w', 'release', 'znskzxi4', 'release'])
            self.assertFalse(any(isinstance(v, FakeModel) for v in vars(predictor).values()))
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'model.joblib'
                joblib.dump(predictor, path)
                restored = joblib.load(path)
                np.testing.assert_array_equal(restored.predict_proba(x), p)

    def test_failed_component_aborts_without_reweighting(self):
        x = pd.DataFrame(np.zeros((4, 13)), columns=candidate.COMPACT)
        with patch.object(candidate, 'verify_checkpoint'), \
             patch.object(candidate, 'make_model', side_effect=RuntimeError('component failed')), \
             patch.object(candidate, 'release_memory') as cleanup:
            predictor = candidate.ArchitecturePredictor(x, [0, 1, 0, 1], [], 7, candidate.ENSEMBLE)
            with self.assertRaisesRegex(RuntimeError, 'component failed'):
                predictor.predict_proba(x)
            cleanup.assert_called_once()


if __name__ == '__main__':
    unittest.main()
