import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import mitra_full_episode as candidate
from mitra_full_episode_probe import sample_episode


class Prep:
    fitted = []
    n_features = 3

    def fit(self, x, y, seed):
        self.fit_ids = x.index.to_numpy().copy()
        type(self).fitted.append(self.fit_ids)
        return self

    def transform(self, x):
        return x.to_numpy(np.float32)


class FullEpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_native_ce_persistent_optimizer_and_validation_memory_boundary(self):
        torch.manual_seed(8)
        model = torch.nn.Linear(3, 10)
        data = torch.randn(20, 3)
        targets = torch.arange(20) % 2
        ids = np.arange(20)*3+101
        optimizer = candidate.base.optimizer_for(model)
        expected_losses = []
        def forward(net, support, y, query):
            result = net(query)
            self.assertEqual(result.shape[1], 10)
            return result
        rng = np.random.default_rng(4)
        repeat_rng = np.random.default_rng(4)
        with patch.object(candidate, 'sample_episode', lambda n, r: sample_episode(n, r, 8, 4)), \
             patch.object(candidate.base, 'model_forward', forward):
            for step in (1, 2):
                _, query = sample_episode(len(data), repeat_rng, 8, 4)
                expected = float(F.cross_entropy(model(data[query]), targets[query]).detach())
                row = candidate.episode_step(model, optimizer, data, targets, ids, rng, step)
                self.assertAlmostEqual(row['training_full_head_ce'], expected, places=6)
                self.assertEqual(row['cumulative_optimizer_steps'], step)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertTrue(all(int(s['step']) == step for s in optimizer.state.values()))
                self.assertGreater(row['optimizer_state_tensor_bytes'], 0)

    def test_training_validation_isolation_zero_fallback_and_fresh_refit(self):
        x = pd.DataFrame(np.arange(120).reshape(40, 3)/120, index=np.arange(40)*3+101)
        y = np.arange(40)%2
        ids = x.index.to_numpy()
        for selected in (0, 1):
            with self.subTest(selected=selected):
                Prep.fitted = []
                models, initial, validation_rows, episodes = [], [], [], []
                def fresh(device):
                    model = torch.nn.Linear(3, 10)
                    models.append(model)
                    initial.append({k: v.detach().clone() for k, v in model.state_dict().items()})
                    return model, {'toy': True}
                def forward(model, support, labels, query):
                    # The input row's first feature uniquely identifies it.
                    self.assertFalse(set(support[:, 0].tolist()) & set(query[:, 0].tolist()))
                    episodes.append((len(models), support[:, 0].tolist(), query[:, 0].tolist()))
                    return model(query)
                def predict(model, data, targets, valid, chunk):
                    self.assertEqual(chunk, 2318)
                    self.assertTrue(all(p.grad is None for p in model.parameters()))
                    self.assertFalse(set(data[:, 0].tolist()) & set(valid[:, 0].tolist()))
                    validation_rows.append((len(data), len(valid)))
                    return np.full((len(valid), 2), .5)
                losses = [1.] * 51 if selected == 0 else [1.] + [.9]*50
                with patch.object(candidate.base, 'MitraPreprocessor', Prep), \
                     patch.object(candidate, 'fresh_model', fresh), \
                     patch.object(candidate, 'sample_episode', lambda n, r: sample_episode(n, r, 8, 4)), \
                     patch.object(candidate.base, 'model_forward', forward), \
                     patch.object(candidate.base, 'predict_arrays', predict), \
                     patch.object(candidate, 'log_loss', side_effect=losses), \
                     contextlib.redirect_stdout(io.StringIO()):
                    artifact = candidate.fit_full_episode(x, y, ids, 9, device='cpu')
                self.assertEqual(len(models), 2)
                self.assertEqual(len(Prep.fitted), 2)
                self.assertEqual(len(Prep.fitted[0]), 32)
                np.testing.assert_array_equal(Prep.fitted[1], ids)
                self.assertEqual(validation_rows, [(32, 8)]*51)
                self.assertEqual(sum(stage == 1 for stage, _, _ in episodes), 50)
                self.assertEqual(sum(stage == 2 for stage, _, _ in episodes), selected)
                self.assertEqual(artifact.metadata['selected_steps'], selected)
                self.assertEqual(len(artifact.labels), 40)
                self.assertEqual(artifact.query_chunk, 2318)
                for key in initial[0]:
                    torch.testing.assert_close(initial[0][key], initial[1][key], atol=0, rtol=0)
                    if selected == 0:
                        torch.testing.assert_close(artifact.state[key], initial[1][key], atol=0, rtol=0)
                if selected:
                    self.assertTrue(any(not torch.equal(artifact.state[k], initial[1][k]) for k in initial[1]))
                inner_values = set(x.loc[Prep.fitted[0]].iloc[:, 0].astype(np.float32))
                for stage, support, query in episodes:
                    if stage == 1:
                        self.assertTrue(set(support+query) <= inner_values)

    def test_fixed_budget_and_unique_ids_fail_before_model(self):
        x = pd.DataFrame(np.zeros((20, 3)))
        with patch.object(candidate, 'fresh_model', side_effect=AssertionError('must not load')):
            with self.assertRaises(ValueError):
                candidate.fit_full_episode(x, np.arange(20)%2, np.arange(20), 7, max_steps=100)
            with self.assertRaises(ValueError):
                candidate.fit_full_episode(x, np.arange(20)%2, np.zeros(20), 7)


if __name__ == '__main__':
    unittest.main()
