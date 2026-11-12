import tempfile
import unittest
import copy
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

    def test_chunked_attention_official_forward_input_and_parameter_gradients(self):
        # Real frozen long-sequence threshold, including Q << K cross attention.
        for query_rows, key_rows, self_attention in [(513, 513, True), (7, 517, False)]:
            with self.subTest(query_rows=query_rows, key_rows=key_rows):
                torch.manual_seed(17)
                reference = candidate.UpstreamAttention(8, 2, False).double()
                adapted = candidate.install_attention_backend(copy.deepcopy(reference))
                self.assertEqual(list(reference.state_dict()), list(adapted.state_dict()))
                q = torch.randn(3, query_rows, 8, dtype=torch.float64, requires_grad=True)
                k = q if self_attention else torch.randn(3, key_rows, 8, dtype=torch.float64, requires_grad=True)
                v = q if self_attention else torch.randn(3, key_rows, 8, dtype=torch.float64, requires_grad=True)
                q2 = q.detach().clone().requires_grad_()
                k2 = q2 if self_attention else k.detach().clone().requires_grad_()
                v2 = q2 if self_attention else v.detach().clone().requires_grad_()
                expected = reference(q, k, v)
                with patch.object(candidate, 'checkpoint', wraps=candidate.checkpoint) as checkpoints:
                    actual = adapted(q2, k2, v2)
                    self.assertEqual(checkpoints.call_count, 3)
                    self.assertTrue(all(call.kwargs['use_reentrant'] is False for call in checkpoints.call_args_list))
                    actual.square().sum().backward()
                expected.square().sum().backward()
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
                for first, second in zip([q, k, v], [q2, k2, v2]):
                    self.assertIsNotNone(first.grad)
                    torch.testing.assert_close(second.grad, first.grad, rtol=1e-8, atol=1e-10)
                for (name, parameter), (name2, parameter2) in zip(reference.named_parameters(), adapted.named_parameters()):
                    self.assertEqual(name, name2)
                    self.assertIsNotNone(parameter.grad)
                    torch.testing.assert_close(parameter2.grad, parameter.grad, rtol=1e-8, atol=1e-10)

    def test_attention_threshold_keeps_short_sequence_vectorized_and_full_keys(self):
        model = candidate.install_attention_backend(candidate.UpstreamAttention(8, 2, False))
        with torch.no_grad(), patch.object(candidate.F, 'scaled_dot_product_attention',
                wraps=candidate.F.scaled_dot_product_attention) as sdpa:
            model(torch.randn(20, 14, 8), torch.randn(20, 14, 8), torch.randn(20, 14, 8))
            self.assertEqual(sdpa.call_count, 1)
            self.assertEqual(sdpa.call_args.args[0].shape[:3], (20, 2, 14))
            sdpa.reset_mock()
            model(torch.randn(3, 7, 8), torch.randn(3, 513, 8), torch.randn(3, 513, 8))
            self.assertEqual(sdpa.call_count, 3)
            for call in sdpa.call_args_list:
                self.assertEqual(call.args[0].shape, (1, 2, 7, 4))
                self.assertEqual(call.args[1].shape, (1, 2, 513, 4))
                self.assertEqual(call.args[2].shape, (1, 2, 513, 4))

    def test_nested_upstream_layer_checkpoint_preserves_model_gradients(self):
        config = dict(TINY, n_layers=2)
        torch.manual_seed(11)
        reference = candidate.make_model(config, memory_backend=False)
        adapted = candidate.make_model(config, reference.state_dict())
        candidate.install_attention_backend(adapted, sequence_threshold=8)
        support, query = torch.randn(20, 3), torch.randn(7, 3)
        labels = torch.arange(20)%2
        expected = candidate.model_forward(reference, support, labels, query)
        actual = candidate.model_forward(adapted, support, labels, query)
        target = torch.arange(7)%2
        candidate.F.cross_entropy(expected, target).backward()
        candidate.F.cross_entropy(actual, target).backward()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        for (name, p), (name2, p2) in zip(reference.named_parameters(), adapted.named_parameters()):
            self.assertEqual(name, name2)
            self.assertEqual(p.grad is None, p2.grad is None)
            if p.grad is not None:
                torch.testing.assert_close(p2.grad, p.grad, atol=2e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
