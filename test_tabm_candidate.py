"""Small checks for TabM preprocessing, member objective and portable export."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch

import tabm_candidate as candidate


def frame(n=40):
    x = pd.DataFrame({c: np.arange(n, dtype=float) for c in candidate.NUMERIC})
    for col in candidate.CATEGORICAL:
        x[col] = np.where(np.arange(n) % 2, 'A', 'B')
    x.loc[0, 'Spa'] = np.nan
    return x


class TabMTests(unittest.TestCase):
    def test_training_only_preprocessing_and_unknown_categories(self):
        x = frame()
        x['Age'] = 7.
        prep = candidate.TabMPreprocessor().fit(x)
        self.assertNotIn('Age', prep.numeric_columns)
        self.assertEqual(prep.medians['Spa'], 20.)
        before = prep.scaler.mean_.copy()
        held = x.iloc[:2].copy()
        held['HomePlanet'] = ['unseen', None]
        held['FoodCourt'] = [1e9, 2e9]
        nums, cats = prep.transform(held)
        np.testing.assert_array_equal(before, prep.scaler.mean_)
        self.assertEqual(cats[0, 0], 0)
        self.assertEqual(cats[1, 0], 1)
        self.assertEqual(cats[0, len(candidate.CATEGORICAL) + candidate.NUMERIC.index('Spa')], 1)
        self.assertEqual(nums.dtype, np.float32)
        self.assertTrue(np.isfinite(nums).all())
        for j, cardinality in enumerate(prep.cat_cardinalities):
            self.assertTrue(((cats[:, j] >= 0) & (cats[:, j] < cardinality)).all())

    def test_each_member_has_its_own_loss_and_probability(self):
        logits = torch.tensor([[[8.], [-8.]]], requires_grad=True)
        targets = torch.ones(1)
        loss = candidate.member_loss(logits, targets)
        loss.backward()
        self.assertGreater(float(loss.detach()), 3.9)
        self.assertGreater(abs(float(logits.grad[0, 1, 0])), .49)
        self.assertLess(abs(float(logits.grad[0, 0, 0])), .001)
        torch.testing.assert_close(candidate.member_probabilities(logits), torch.tensor([.5]))

    def test_cpu_forward_pickle_and_full_refit_epoch_contract(self):
        x = frame()
        y = np.arange(len(x)) % 2
        # Stub only epoch selection so this test never runs an early-stop search.
        with patch.object(candidate, 'select_epochs', return_value=(1, {'test_selection': True})):
            fitted = candidate.fit_tabm(x, y, 13, 'cpu')
        self.assertEqual(fitted.selected_epochs, 1)
        self.assertEqual(fitted.metadata['outer_train_rows'], len(x))
        self.assertEqual(fitted.preprocessor.scaler.n_samples_seen_, len(x))
        self.assertTrue(all(p.device.type == 'cpu' for p in fitted.model.parameters()))
        before = fitted.predict_proba(x.iloc[:3])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            joblib.dump(fitted, path)
            restored = joblib.load(path)
            np.testing.assert_allclose(restored.predict_proba(x.iloc[:3]), before, atol=1e-7)
        self.assertTrue(np.isfinite(before).all())
        np.testing.assert_allclose(before.sum(axis=1), 1.)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
