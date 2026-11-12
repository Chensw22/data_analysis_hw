"""Synthetic CPU checks only; no project data or MPS training."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch

import scarf_candidate as candidate


def synthetic(n=80):
    rng = np.random.default_rng(7)
    raw = pd.DataFrame({col: rng.uniform(1, 20, n) for col in candidate.NUMERIC})
    for col in candidate.CATEGORICAL:
        raw[col] = np.resize(['A', 'B', '__MISSING__'], n)
    raw.loc[::7, 'Spa'] = np.nan
    raw.loc[60:, 'HomePlanet'] = 'test_only_category'
    return raw


SMALL = dict(width=8, encoder_layers=2, batch_size=16, pretrain_max_epochs=1,
             pretrain_patience=1, pretrain_min_epochs=1, supervised_max_epochs=1,
             supervised_patience=1, supervised_min_epochs=1, validation_corruption_passes=2)


class SCARFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_training_only_preprocessor_and_atomic_corruption(self):
        raw = synthetic()
        prep = candidate.Preprocessor().fit(raw.iloc[:40])
        self.assertNotIn('test_only_category', prep.vocabularies['HomePlanet'])
        x = prep.transform(raw)
        home_group = prep.groups[7]
        self.assertTrue((x[60:, home_group[0]] == 1).all())
        original = x.copy()
        changed = candidate.corrupt(x[:20], x[20:40], prep.groups, np.random.default_rng(3), .6)
        for row in range(len(changed)):
            for group in prep.groups:
                allowed = np.concatenate([x[row:row+1, group], x[20:40][:, group]])
                self.assertTrue(np.any(np.all(allowed == changed[row, group], axis=1)))
        for group in prep.groups[7:]:
            np.testing.assert_array_equal(changed[:, group].sum(axis=1), np.ones(len(changed)))
        np.testing.assert_array_equal(x, original)

    def test_matching_encoder_initialization_and_fixed_validation(self):
        raw = synthetic()
        prep = candidate.Preprocessor().fit(raw.iloc[:40])
        x = prep.transform(raw)
        with patch.dict(candidate.SETTINGS, SMALL):
            contrastive = candidate.make_contrastive(x.shape[1], 13, 'cpu')
            classifier = candidate.make_classifier(x.shape[1], 13, 'cpu')
            for key, value in contrastive.encoder.state_dict().items():
                self.assertTrue(torch.equal(value, classifier.encoder.state_dict()[key]))
            pairs = candidate.static_validation(x[40:48], x[:40], prep.groups, 3)
            before = [(a.copy(), b.copy()) for a, b in pairs]
            first = candidate.contrastive_validation(contrastive, pairs, 'cpu')
            second = candidate.contrastive_validation(contrastive, pairs, 'cpu')
            self.assertEqual(first, second)
            for (a, b), (c, d) in zip(pairs, before):
                np.testing.assert_array_equal(a, c)
                np.testing.assert_array_equal(b, d)

    def test_real_synthetic_two_recipes_refit_boundaries_and_serialization(self):
        raw = synthetic()
        matrices = {False: (raw, candidate.CATEGORICAL)}
        y = np.arange(40) % 2
        seen_prep = []
        original_fit = candidate.Preprocessor.fit
        def record_fit(prep, x):
            seen_prep.append(set(x.index))
            return original_fit(prep, x)
        with patch.dict(candidate.SETTINGS, SMALL), patch.object(candidate, 'DEVICE', 'cpu'), \
             patch.object(candidate, 'TEST_ROWS', np.arange(60, 80)), \
             patch.object(candidate.Preprocessor, 'fit', new=record_fit):
            results = {}
            for name in [candidate.SCRATCH, candidate.SCARF]:
                p, model = candidate.fit(name, matrices, np.arange(40), y, np.arange(40, 60), 11)
                self.assertTrue(np.isfinite(p).all())
                self.assertEqual(len(model.metadata['phases']), 2 if name == candidate.SCRATCH else 5)
                self.assertEqual(model.metadata['inner_training_rows'], 32)
                self.assertEqual(model.metadata['inner_validation_rows'], 8)
                self.assertEqual(model.metadata['actual_test_attribute_rows'], 20)
                self.assertTrue(all(t.device.type == 'cpu' for t in model.state.values()))
                self.assertNotIn('test_only_category', model.preprocessor.vocabularies['HomePlanet'])
                if name == candidate.SCARF:
                    phases = model.metadata['phases']
                    self.assertEqual([phases[i]['pool_rows'] for i in [0, 1, 3]], [52, 52, 60])
                    self.assertTrue(all(phases[i]['first_encoder_parameter_max_change'] > 0 for i in [0, 1, 3]))
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder)/'model.joblib'
                    joblib.dump(model, path)
                    restored = joblib.load(path)
                    np.testing.assert_array_equal(p, restored.predict_proba(raw.iloc[40:60])[:, 1])
                    restored.device = 'mps'
                    with patch.object(candidate.torch.backends.mps, 'is_available', return_value=False):
                        np.testing.assert_array_equal(p, restored.predict_proba(raw.iloc[40:60])[:, 1])
                    self.assertEqual(restored.last_prediction_device, 'cpu')
                results[name] = model.metadata
            self.assertEqual(seen_prep[0], seen_prep[2])
            self.assertEqual(seen_prep[1], set(range(40)))
            self.assertEqual(seen_prep[3], set(range(40)))
            self.assertTrue(all(not rows & set(range(40, 80)) for rows in seen_prep))
            self.assertEqual(results[candidate.SCRATCH]['inner_train_index_sha256'],
                             results[candidate.SCARF]['inner_train_index_sha256'])

    def test_reject_labelled_test_or_overlapping_prediction_rows(self):
        raw = synthetic()
        with patch.object(candidate, 'TEST_ROWS', np.arange(60, 80)):
            for train, query in [(np.arange(40), np.arange(30, 50)),
                                 (np.arange(50, 70), np.arange(40, 50))]:
                with self.assertRaisesRegex(ValueError, 'Training labels'):
                    candidate.fit(candidate.SCARF, {False: (raw, [])}, train,
                                  np.arange(len(train)) % 2, query, 7)

    def test_freeze_precedes_first_model(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out/'protocol.json').write_text(json.dumps({'sources': {}}))
            def model(*args):
                protocol = json.loads((out/'protocol.json').read_text())
                self.assertIn('scarf_candidate.py', protocol['sources'])
                self.assertIn('reports/scarf_protocol.json', protocol['sources'])
                self.assertIn('scarf_environment_sha256', protocol)
                return 'ready'
            with patch.object(candidate, 'BASE_EVALUATE', side_effect=model):
                self.assertEqual(candidate.evaluate(candidate.iterate.BASELINE, None, None, None, 7, out), 'ready')


if __name__ == '__main__':
    unittest.main()
