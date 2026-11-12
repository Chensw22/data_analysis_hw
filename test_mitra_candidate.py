import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import torch

import mitra_candidate as candidate


TINY = dict(dim=8, dim_output=10, n_layers=1, n_heads=2, task='CLASSIFICATION')


class MitraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(31)
        return candidate.make_model(TINY)

    def test_train_only_schema_unknown_missing_and_all_missing(self):
        x = candidate.synthetic_frame(24)
        x['Spa'] = np.nan
        original = x.copy(deep=True)
        prep = candidate.MitraPreprocessor().fit(x, np.arange(24)%2)
        query = candidate.synthetic_frame(3, 5)
        query['HomePlanet'] = ['unseen', None, x.HomePlanet.dropna().iloc[0]]
        encoded = prep.encode(query)
        col = prep.columns.index('HomePlanet')
        np.testing.assert_array_equal(encoded[:2, col], [0, 1])
        means = prep.upstream.pre_nan_mean.copy()
        vocab = dict(prep.vocab['HomePlanet'])
        first = prep.transform(query)
        query['Age'] = 1e10
        second = prep.transform(query)
        np.testing.assert_array_equal(means, prep.upstream.pre_nan_mean)
        self.assertEqual(vocab, prep.vocab['HomePlanet'])
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(first).all() and np.isfinite(second).all())
        self.assertTrue(x.equals(original))
        self.assertTrue(prep.all_missing[prep.columns.index('Spa')])

    def test_query_chunk_invariance_and_context_is_not_mutated(self):
        rng = np.random.default_rng(5)
        support = torch.from_numpy(rng.normal(size=(12, 3)).astype(np.float32))
        labels = torch.arange(12)%2
        query = torch.from_numpy(rng.normal(size=(7, 3)).astype(np.float32))
        before = support.clone()
        model = self.model()
        full = candidate.predict_arrays(model, support, labels, query, 7)
        chunked = candidate.predict_arrays(model, support, labels, query, 2)
        np.testing.assert_allclose(full, chunked, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(full.sum(axis=1), 1., atol=1e-6)
        torch.testing.assert_close(support, before, atol=0, rtol=0)

    def test_episode_disjoint_and_query_labels_not_a_forward_argument(self):
        generator = torch.Generator().manual_seed(19)
        support_ids, query_ids = candidate.episode_indices(24, 5, generator)
        self.assertFalse(set(support_ids.tolist()) & set(query_ids.tolist()))
        self.assertEqual(len(set(support_ids.tolist()) | set(query_ids.tolist())), 24)
        x, y = torch.randn(24, 3), torch.arange(24)%2
        model = self.model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        record = candidate.backward_probe(model, x, y, 5)
        self.assertEqual(record['optimizer_steps'], 1)
        self.assertTrue(record['finite_gradients'])
        self.assertTrue(record['weight_update_confirmed'])
        self.assertGreater(record['optimizer_state_tensor_bytes'], 0)
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_resource_episode_can_omit_capped_query_remainder(self):
        model = self.model()
        record = candidate.backward_probe(model, torch.randn(25, 3), torch.arange(25)%2,
                                          query_rows=3, support_rows=20)
        self.assertEqual((record['support_rows'], record['query_rows']), (20, 3))
        self.assertTrue(record['disjoint'])

    def test_full_context_has_no_8192_limit(self):
        class RecordingModel(torch.nn.Module):
            def forward(self, support, labels, query, *masks):
                self.support_rows = support.shape[1]
                self.labels_rows = labels.shape[1]
                return torch.zeros((1, query.shape[1], 10))
        model = RecordingModel()
        candidate.predict_arrays(model, torch.zeros(8693, 2), torch.zeros(8693, dtype=torch.long),
                                 torch.ones(3, 2), 2)
        self.assertEqual(model.support_rows, 8693)
        self.assertEqual(model.labels_rows, 8693)

    def test_cpu_state_context_roundtrip_and_meta_rebuild(self):
        x = candidate.synthetic_frame(20)
        y = np.arange(20)%2
        prep = candidate.MitraPreprocessor().fit(x, y)
        model = self.model()
        predictor = candidate.MitraPredictor(prep, TINY, model.state_dict(), prep.transform(x), y)
        query = candidate.synthetic_frame(5, 11)
        expected = predictor.predict_proba(query)
        self.assertIsNotNone(predictor._engine)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mitra.joblib'
            joblib.dump(predictor, path)
            restored = joblib.load(path)
            self.assertEqual(restored.device, 'cpu')
            self.assertIsNone(restored._engine)
            np.testing.assert_array_equal(restored.support, predictor.support)
            np.testing.assert_array_equal(restored.labels, y)
            np.testing.assert_array_equal(restored.predict_proba(query), expected)
            self.assertTrue(all(v.device.type == 'cpu' for v in restored.state.values()))

    def test_no_checkpoint_download_and_no_empty_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'no automatic download'):
                candidate.verify_checkpoint(directory)
        with self.assertRaises(ValueError):
            candidate.episode_indices(5, 5, torch.Generator())
        with self.assertRaises(ValueError):
            candidate.binary_labels([0, 2], 2)


if __name__ == '__main__':
    unittest.main()
