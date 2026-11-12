"""CPU-only checks for the FT-Transformer training contract, never development CV."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch

import ft_transformer_candidate as candidate


def frame(n=40):
    x = pd.DataFrame({c: np.arange(n, dtype=float) for c in candidate.NUMERIC})
    for col in candidate.CATEGORICAL:
        x[col] = np.where(np.arange(n) % 2, 'A', 'B')
    x.loc[0, 'Spa'] = np.nan
    return x


class TinyModel(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = torch.nn.Linear(width, 1)

    def forward(self, numeric, categorical):
        return self.linear(numeric)

    def make_default_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


class FTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_train_only_preprocessing_seven_binary_category_flags(self):
        x = frame()
        x['Age'] = 7.
        x['RoomService'] = np.nan
        prep = candidate.FTPreprocessor().fit(x)
        self.assertNotIn('Age', prep.numeric_columns)
        self.assertNotIn('RoomService', prep.numeric_columns)
        self.assertEqual(prep.medians['Spa'], 20.)
        self.assertEqual(prep.medians['RoomService'], 0.)
        before = prep.scaler.mean_.copy()
        held = x.iloc[:2].copy()
        held['HomePlanet'] = ['unseen', None]
        held['FoodCourt'] = [1e9, 2e9]
        numeric, categorical = prep.transform(held)
        np.testing.assert_array_equal(prep.scaler.mean_, before)
        self.assertEqual(categorical.shape, (2, 13))
        self.assertEqual(prep.cat_cardinalities[-7:], [2]*7)
        self.assertEqual(categorical[0, 0], 0)
        self.assertEqual(categorical[1, 0], 1)
        self.assertEqual(categorical[0, 6+candidate.NUMERIC.index('Spa')], 1)
        self.assertEqual(numeric.dtype, np.float32)
        self.assertTrue(np.isfinite(numeric).all())
        for j, cardinality in enumerate(prep.cat_cardinalities):
            self.assertTrue(((categorical[:, j] >= 0) & (categorical[:, j] < cardinality)).all())

    def test_inner_selection_fresh_refit_and_step_counts(self):
        x = frame()
        y = np.arange(len(x)) % 2
        fitted_preprocessors, models = [], []
        def make(prep, seed, device):
            fitted_preprocessors.append(prep)
            model = TinyModel(len(prep.numeric_columns))
            models.append(model)
            return model
        settings = dict(candidate.SETTINGS, max_epochs=5, patience=2)
        with patch.object(candidate, 'SETTINGS', settings), \
             patch.object(candidate, 'make_model', side_effect=make), \
             patch.object(candidate, 'train_epoch', return_value={'training_bce': .6, 'optimizer_steps': 1}), \
             patch.object(candidate, 'log_loss', side_effect=[.6, .5, .51, .52]):
            fitted = candidate.fit_ft(x, y, 13, 'cpu')
        self.assertEqual([int(p.scaler.n_samples_seen_) for p in fitted_preprocessors], [32, 40])
        self.assertIsNot(models[0], models[1])
        self.assertIs(fitted.model, models[1])
        self.assertEqual(fitted.metadata['parameters'], sum(p.numel() for p in models[1].parameters()))
        self.assertEqual(fitted.selected_epochs, 2)
        self.assertEqual(fitted.metadata['inner_optimizer_steps'], 4)
        self.assertEqual(fitted.metadata['refit_optimizer_steps'], 2)
        self.assertEqual(len(fitted.metadata['inner_history']), 4)
        self.assertEqual(fitted.metadata['stop_reason'], 'patience')
        self.assertFalse(fitted.metadata['budget_capped'])

    def test_cap_is_reported_and_deadline_fails_closed(self):
        x = frame()
        settings = dict(candidate.SETTINGS, max_epochs=2, patience=40)
        with patch.object(candidate, 'SETTINGS', settings), \
             patch.object(candidate, 'make_model', side_effect=lambda p, s, d: TinyModel(len(p.numeric_columns))), \
             patch.object(candidate, 'train_epoch', return_value={'training_bce': .6, 'optimizer_steps': 1}), \
             patch.object(candidate, 'log_loss', side_effect=[.6, .5]):
            epoch, metadata = candidate.select_epochs(x, np.arange(len(x)) % 2, 7, 'cpu')
        self.assertEqual(epoch, 2)
        self.assertTrue(metadata['budget_capped'])
        with patch.object(candidate, 'DEADLINE', time.monotonic()-1):
            with self.assertRaisesRegex(TimeoutError, 'incomplete candidate'):
                candidate.check_deadline()

    def test_real_official_cpu_backward_optimizer_groups_and_serialization(self):
        x = frame()
        y = np.arange(len(x)) % 2
        with patch.object(candidate, 'select_epochs', return_value=(1, {'test_selection': True})):
            fitted = candidate.fit_ft(x, y, 13, 'cpu')
        optimizer = fitted.model.make_default_optimizer()
        self.assertEqual({g['lr'] for g in optimizer.param_groups}, {1e-4})
        self.assertEqual({g['weight_decay'] for g in optimizer.param_groups}, {0., 1e-5})
        no_decay = {id(p) for g in optimizer.param_groups if g['weight_decay'] == 0 for p in g['params']}
        self.assertTrue(all(id(p) in no_decay for p in fitted.model.cont_embeddings.parameters()))
        self.assertTrue(all(p.device.type == 'cpu' for p in fitted.model.parameters()))
        before = fitted.predict_proba(x.iloc[:3])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.joblib'
            joblib.dump(fitted, path)
            restored = joblib.load(path)
            np.testing.assert_allclose(restored.predict_proba(x.iloc[:3]), before, atol=1e-7)
        self.assertTrue(np.isfinite(before).all())
        np.testing.assert_allclose(before.sum(axis=1), 1.)

    def test_protocol_matches_and_freezes_before_baseline(self):
        protocol = json.loads((candidate.iterate.ROOT/candidate.PROTOCOL_FILE).read_text())
        self.assertEqual(protocol['settings'], candidate.SETTINGS)
        order = []
        with patch.object(candidate, 'record_protocol', side_effect=lambda out: order.append('freeze')), \
             patch.object(candidate, 'BASE_EVALUATE', side_effect=lambda *args: order.append('evaluate')):
            candidate.evaluate(candidate.iterate.BASELINE, {}, None, [], 7, Path('/unused'))
        self.assertEqual(order, ['freeze', 'evaluate'])
        with patch.object(candidate, 'BASE_FIT', return_value='v6') as fit:
            self.assertEqual(candidate.fit(candidate.iterate.BASELINE, {}, [], [], [], 7), 'v6')
            fit.assert_called_once()


if __name__ == '__main__':
    unittest.main()
