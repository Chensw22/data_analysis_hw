import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch

import tabr_candidate as candidate
from third_party.tabr.model import TabRModel


def frame(n=32):
    x = pd.DataFrame({col: np.arange(n, dtype=float)+1 for col in candidate.NUMERIC})
    for col in candidate.CATEGORICAL:
        x[col] = ['A', 'B']*(n//2)
    x.loc[0, 'Age'] = np.nan
    return x


class TabRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_self_label_is_invisible_but_duplicate_neighbor_is_allowed(self):
        torch.manual_seed(3)
        model = TabRModel(3, d_main=8, context_size=2, context_dropout=0., dropout0=0.)
        model.eval()
        library = torch.tensor([[1., 2., 3.], [1., 2., 3.], [8., 6., 9.], [7., 9., 2.]])
        labels, ids = torch.tensor([0, 1, 0, 1]), torch.tensor([91, 92, 93, 94])
        query = library[:1]
        _, keys = model.encode(library)
        neighbors = model.retrieve(keys[:1], keys, ids[:1], ids)
        self.assertNotIn(0, neighbors[0].tolist())
        self.assertIn(1, neighbors[0].tolist())
        first = model(query, library, labels, ids, ids[:1])
        labels[0] = 1
        torch.testing.assert_close(first, model(query, library, labels, ids, ids[:1]), rtol=0, atol=0)
        labels[1] = 0
        self.assertFalse(torch.equal(first, model(query, library, labels, ids, ids[:1])))

    def test_selected_candidate_keys_and_labels_receive_gradients(self):
        torch.manual_seed(7)
        model = TabRModel(3, d_main=8, context_size=3, context_dropout=0., dropout0=0.)
        library = torch.randn(8, 3, requires_grad=True)
        query = library[:2].detach().clone().requires_grad_()
        labels, ids = torch.arange(8)%2, torch.arange(8)
        model(query, library, labels, ids, ids[:2]).square().mean().backward()
        for gradient in [library.grad, query.grad, model.K.weight.grad,
                         model.label_encoder.weight.grad, model.T[0].weight.grad]:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_preprocessor_train_only_unknown_missing_and_flags(self):
        x = frame()
        prep = candidate.TabRPreprocessor().fit(x.iloc[:16])
        median = prep.medians['Age']
        query = x.iloc[[16]].copy()
        query['Age'] = 1e6
        query['HomePlanet'] = 'never-in-training'
        first = prep.transform(query)
        query['HomePlanet'] = None
        second = prep.transform(query)
        self.assertEqual(prep.medians['Age'], median)
        start = int(prep.keep.sum())
        np.testing.assert_array_equal(first[0, start:start+2], [1., 0.])
        np.testing.assert_array_equal(second[0, start:start+2], [0., 1.])
        self.assertEqual(len(prep.cardinalities), 13)  # Six original categories + seven flags.
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(first).all())

    def test_inner_library_is_disjoint_and_full_refit_is_fresh(self):
        x, y, ids = frame(), np.arange(32)%2, np.arange(100, 132, dtype=np.int64)
        created, libraries = [], []
        original_make, original_predict = candidate.make_model, candidate.predict_arrays
        def tracked_make(prep, seed, device):
            model = original_make(prep, seed, device)
            created.append((prep, model))
            return model
        def tracked_predict(model, query, library, labels, lib_ids, query_ids=None, batch_size=256):
            if query_ids is not None:
                self.assertFalse(set(lib_ids.tolist()) & set(query_ids.tolist()))
            libraries.append(set(lib_ids.tolist()))
            return original_predict(model, query, library, labels, lib_ids, query_ids, batch_size)
        with patch.dict(candidate.SETTINGS, d_main=8, context_size=3, max_epochs=2,
                        patience=1, batch_size=8), patch.object(candidate, 'make_model',
                        side_effect=tracked_make), patch.object(candidate, 'predict_arrays',
                        side_effect=tracked_predict):
            result = candidate.fit_tabr(x, y, ids, 13, 'cpu')
        self.assertEqual(len(created), 2)
        self.assertIsNot(created[0][0], created[1][0])
        self.assertIsNot(created[0][1], created[1][1])
        self.assertTrue(all(len(library) == 25 for library in libraries))
        np.testing.assert_array_equal(result.ids.numpy(), ids)
        self.assertEqual(result.metadata['outer_training_rows'], 32)
        self.assertEqual(result.metadata['inner_validation_rows'], 7)
        self.assertEqual(result.metadata['inner_optimizer_steps'],
                         4*len(result.metadata['inner_epoch_log']))
        self.assertEqual(result.metadata['refit_optimizer_steps'],
                         4*result.metadata['selected_epochs'])
        self.assertTrue(all(np.isfinite(row['training_bce']) and row['seconds'] >= 0
                            for row in result.metadata['inner_epoch_log']))
        self.assertGreater(result.metadata['parameters'], 0)

    def test_serialization_preserves_library_and_predictions(self):
        x, y, ids = frame(), np.arange(32)%2, np.arange(32, dtype=np.int64)
        prep = candidate.TabRPreprocessor().fit(x)
        model = TabRModel(prep.n_features, d_main=8, context_size=3, context_dropout=0., dropout0=0.)
        predictor = candidate.TabRPredictor(prep, model, torch.from_numpy(prep.transform(x)),
            torch.from_numpy(y), torch.from_numpy(ids), 'cpu', {'selected_epochs': 0})
        expected = predictor.predict_proba(x.iloc[:3], query_ids=ids[:3])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tabr.joblib'
            joblib.dump(predictor, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(restored.predict_proba(x.iloc[:3], query_ids=ids[:3]), expected)
            np.testing.assert_array_equal(restored.labels.numpy(), y)
            np.testing.assert_array_equal(restored.ids.numpy(), ids)

    def test_outer_overlap_rejected_before_any_training(self):
        with patch.object(candidate, 'fit_tabr') as fit:
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                candidate.fit(candidate.NAME, {False: (frame(), [])}, [0, 1], [0, 1], [1, 2], 7)
            fit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
