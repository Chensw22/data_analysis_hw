import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import torch
from torch.nn import functional as F

import mitra_lastblock as candidate


TINY = dict(dim=8, dim_output=10, n_layers=2, n_heads=2, task='CLASSIFICATION')


class LastBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setup_pair(self):
        torch.manual_seed(71)
        original = candidate.configure(candidate.base.make_model(TINY))
        return original, copy.deepcopy(original)

    def data(self):
        generator = torch.Generator().manual_seed(9)
        return (torch.randn(19, 3, generator=generator), torch.arange(19)%2,
                torch.randn(7, 3, generator=generator), torch.arange(7)%2)

    def test_outputs_gradients_and_adamw_match_original(self):
        original, split = self.setup_pair()
        support, labels, query, targets = self.data()
        full = candidate.base.model_forward(original, support, labels, query)
        actual = candidate.split_forward(split, support, labels, query, 3)
        torch.testing.assert_close(full, actual, atol=1e-6, rtol=1e-5)
        F.cross_entropy(full, targets).backward()
        record = candidate.backward_episode(split, support, labels, query, targets,
                                             np.arange(19), np.arange(19, 26), 3)
        self.assertEqual(record['microbatches'], 3)
        self.assertAlmostEqual(record['training_full_head_ce'],
                               float(F.cross_entropy(full, targets).detach()), places=6)
        for (name, a), (other, b) in zip(original.named_parameters(), split.named_parameters()):
            self.assertEqual(name, other)
            if not a.requires_grad:
                self.assertIsNone(a.grad)
                self.assertIsNone(b.grad)
            else:
                self.assertEqual(a.grad is None, b.grad is None, name)
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5, msg=name)
        for model in [original, split]:
            candidate.base.optimizer_for(model).step()
        for name, parameter in original.state_dict().items():
            torch.testing.assert_close(parameter, split.state_dict()[name], atol=2e-6, rtol=2e-5, msg=name)

    def test_support_normalization_kv_gradients_and_prefix_freeze(self):
        _, model = self.setup_pair()
        support, labels, query, targets = self.data()
        digest = candidate.frozen_digest(model)
        optimizer = candidate.base.optimizer_for(model)
        with patch.object(candidate, 'frozen_prefix', wraps=candidate.frozen_prefix) as prefix:
            record = candidate.optimizer_step(model, optimizer, support, labels, query, targets,
                                               np.arange(19), np.arange(19, 26), 1, 3)
        self.assertEqual(prefix.call_count, 1)
        self.assertEqual(digest, candidate.frozen_digest(model))
        self.assertTrue(record['finite_gradients'])
        self.assertGreater(record['optimizer_state_tensor_bytes'], 0)
        parameters = dict(model.named_parameters())
        for name in ['layers.1.layer_norm1.weight', 'layers.1.attention1.k.weight',
                     'layers.1.attention1.v.weight']:
            self.assertIsNotNone(parameters[name].grad)
            self.assertGreater(float(parameters[name].grad.abs().max()), 0, name)
        for parameter in model.parameters():
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad)

    def test_full_vs_nondivisible_microbatch_gradient_weighting(self):
        a, b = self.setup_pair()
        support, labels, query, targets = self.data()
        for model, chunk in [(a, 7), (b, 3)]:
            candidate.backward_episode(model, support, labels, query, targets,
                                         np.arange(19), np.arange(19, 26), chunk)
        for (name, x), (_, y) in zip(a.named_parameters(), b.named_parameters()):
            if x.grad is not None:
                torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-5, msg=name)

    def test_overlap_rejected_before_prefix_and_forward_has_no_query_targets(self):
        _, model = self.setup_pair()
        support, labels, query, targets = self.data()
        with patch.object(candidate, 'frozen_prefix', side_effect=AssertionError('must not run')):
            with self.assertRaises(ValueError):
                candidate.backward_episode(model, support, labels, query, targets,
                                             np.arange(19), np.arange(7), 3)
        import inspect
        self.assertNotIn('query_labels', inspect.signature(candidate.frozen_prefix).parameters)
        self.assertNotIn('query_labels', inspect.signature(candidate.query_tail).parameters)

    def test_portable_state_context_roundtrip_original_architecture(self):
        _, model = self.setup_pair()
        frame = candidate.base.synthetic_frame(35, 5)
        labels = np.arange(28)%2
        prep = candidate.base.MitraPreprocessor().fit(frame.iloc[:28], labels, 3)
        support = torch.from_numpy(prep.transform(frame.iloc[:28]))
        query = torch.from_numpy(prep.transform(frame.iloc[28:]))
        y = torch.from_numpy(labels)
        optimizer = candidate.base.optimizer_for(model)
        candidate.optimizer_step(model, optimizer, support[:21], y[:21], support[21:], y[21:],
                                 np.arange(21), np.arange(21, 28), 1, 3)
        with torch.no_grad():
            expected = candidate.split_forward(model, support, y, query, 3)[:, :2].softmax(-1).numpy()
        artifact = candidate.base.MitraPredictor(prep, TINY, model.state_dict(), support.numpy(), labels)
        original_path = artifact.predict_proba(frame.iloc[28:])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.joblib'
            joblib.dump(artifact, path)
            restored = joblib.load(path)
            self.assertEqual(restored.device, 'cpu')
            actual = restored.predict_proba(frame.iloc[28:])
            np.testing.assert_allclose(expected, actual, atol=2e-6, rtol=2e-5)
            np.testing.assert_array_equal(original_path, actual)


if __name__ == '__main__':
    unittest.main()
