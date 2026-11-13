import unittest
from unittest.mock import patch
import io
import contextlib
import tempfile
from pathlib import Path
import joblib
import numpy as np
import torch
import mitra_full_episode_probe as probe


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(3, 4), torch.nn.Linear(4, 10)])

    def forward(self, x):
        return self.layers[1](torch.tanh(self.layers[0](x)))


class ResourceProbeTests(unittest.TestCase):
    def test_existing_probe_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory)/'result.json'
            result.write_text('previous evidence')
            with self.assertRaises(FileExistsError):
                probe.run(directory)
            self.assertEqual(result.read_text(), 'previous evidence')

    def test_fresh_disjoint_sampler(self):
        rng = np.random.default_rng(42)
        a, b = probe.sample_episode(8693, rng)
        c, d = probe.sample_episode(8693, rng)
        self.assertEqual((len(a), len(b)), (1024, 256))
        self.assertEqual(len(np.unique(np.r_[a, b])), 1280)
        self.assertFalse(np.array_equal(a, c))
        np.testing.assert_array_equal(a, probe.sample_episode(8693, np.random.default_rng(42))[0])
        with self.assertRaises(ValueError):
            probe.sample_episode(100, rng)

    def test_persistent_native_head_adamw_updates(self):
        torch.manual_seed(5)
        model = Tiny()
        data = torch.randn(20, 3)
        labels = torch.arange(20) % 2
        optimizer = probe.mitra.optimizer_for(model)
        def forward(net, support, y, query):
            self.assertEqual(len(support), 8)
            self.assertEqual(len(query), 4)
            return net(query)
        with patch.object(probe.mitra, 'model_forward', forward), contextlib.redirect_stdout(io.StringIO()):
            result = probe.two_updates(model, optimizer, data, labels, np.random.default_rng(2), 8, 4)
        self.assertEqual([item['optimizer_state_steps'] for item in result], [[1], [2]])
        self.assertGreater(result[0]['optimizer_state_tensor_bytes'], 0)
        self.assertEqual(result[0]['optimizer_state_tensor_bytes'], result[1]['optimizer_state_tensor_bytes'])
        for item in result:
            self.assertTrue(all(item['updated_blocks'].values()))
            self.assertTrue(item['disjoint'])
        self.assertAlmostEqual(result[1]['lr'], 5.7e-7)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_portable_recipe_and_reload_rejects_partial_or_changed(self):
        predictor = probe.mitra.MitraPredictor(None, {'test': True}, {'weight': torch.ones(2)},
            np.zeros((8693, 13), np.float32), np.arange(8693) % 2, 'mps',
            metadata={'recipe': probe.RECIPE}, query_chunk=2318)
        stream = io.BytesIO()
        joblib.dump(predictor, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        self.assertEqual(restored.device, 'cpu')
        self.assertEqual(restored.query_chunk, 2318)
        self.assertEqual(restored.metadata['recipe'], probe.RECIPE)
        np.testing.assert_array_equal(restored.labels, predictor.labels)
        probabilities = np.tile([.6, .4], (2318, 1))
        self.assertEqual(probe.check_reload(probabilities, probabilities.copy()), 0)
        with self.assertRaises(ValueError):
            probe.check_reload(probabilities[:128], probabilities[:128])
        changed = probabilities.copy()
        changed[0] = [.4, .6]
        with self.assertRaises(RuntimeError):
            probe.check_reload(probabilities, changed)


if __name__ == '__main__':
    unittest.main()
