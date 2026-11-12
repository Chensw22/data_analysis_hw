"""Small boundary and real CatBoost API tests; never run development CV."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import nested_structure_search as candidate


def frame(n=90):
    rng = np.random.default_rng(123)
    x = pd.DataFrame(dict(HomePlanet=['Earth'] * n, CryoSleep=['True'] * n,
                          Side=np.where(np.arange(n) % 2, 'P', 'S'),
                          CabinNumber=np.arange(n, dtype=float), Age=rng.uniform(12, 70, n)))
    for col in candidate.iterate.base.SPEND:
        x[col] = rng.uniform(0, 100, n)
    return x


class SearchTests(unittest.TestCase):
    def tearDown(self):
        candidate.DEADLINE = None
        candidate.RUN_OUT = None

    def test_frozen_space_contains_exact_original_v34_cat(self):
        self.assertEqual(len(candidate.PATHS), 6)
        self.assertEqual(len(candidate.CANDIDATES), 18)
        self.assertEqual(candidate.CANDIDATES[1], dict(
            grow_policy='SymmetricTree', depth=6, l2_leaf_reg=5, iterations=700))
        self.assertEqual(candidate.SETTINGS['learning_rate'], .045)
        for config in candidate.CANDIDATES:
            if config['grow_policy'] == 'SymmetricTree':
                self.assertNotIn('min_data_in_leaf', config)
                self.assertNotIn('max_leaves', config)

    def test_real_policies_checkpoint_and_serialization(self):
        raw = frame()
        x = candidate.add_contrasts(raw)
        y = np.arange(len(raw)) % 2
        cats = ['HomePlanet', 'CryoSleep', 'Side']
        for structure in candidate.STRUCTURES:
            with self.subTest(policy=structure['grow_policy']):
                model = candidate.train_cat(x, y, cats, 7,
                    {**structure, 'l2_leaf_reg': 5, 'iterations': 8})
                for count in [3, 7]:
                    prefix = model.predict_proba(x, ntree_end=count)
                    shortened = candidate.train_cat(x, y, cats, 7,
                        {**structure, 'l2_leaf_reg': 5, 'iterations': count})
                    np.testing.assert_allclose(prefix, shortened.predict_proba(x), rtol=0, atol=1e-14)
                spatial = candidate.fit_spatial(x, y, 7)
                spatial.base_model = model
                predictor = candidate.NestedStructurePredictor(spatial, {'refit_tree_count': 8})
                p = predictor.predict_proba(raw)
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(((p >= 0) & (p <= 1)).all())
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / 'predictor.joblib'
                    joblib.dump(predictor, path)
                    restored = joblib.load(path)
                    np.testing.assert_array_equal(p, restored.predict_proba(raw))
                    self.assertEqual(restored.spatial_model.base_model.tree_count_, 8)

    def test_inner_boundaries_pooled_selection_and_fresh_exact_refit(self):
        raw = frame(75)
        y = np.arange(60) % 2
        models, spatial_training_rows = [], []

        class RecordingCat:
            def __init__(self, config):
                self.config = config

            def fit(self, x, labels, **kwargs):
                self.rows = set(x.index)
                self.kwargs = kwargs
                return self

            def predict_proba(self, x, ntree_end=None):
                # All 18 candidates tie, verifying the fixed final tie break.
                self.last_prediction_rows = set(x.index)
                if len(self.rows) < 60:
                    assert not self.rows & self.last_prediction_rows
                p = .2 + .6 * (x.index.to_numpy() % 2)
                return np.column_stack([1 - p, p])

        class RecordingSpatial:
            def predict_proba(self, x):
                return self.base_model.predict_proba(x)

        def create(cats, seed, config):
            model = RecordingCat(config)
            models.append(model)
            return model

        def make_spatial(x, labels, seed):
            spatial_training_rows.append(set(x.index))
            return RecordingSpatial()

        with patch.object(candidate, 'new_cat', side_effect=create), \
             patch.object(candidate, 'fit_spatial', side_effect=make_spatial), \
             patch.object(candidate, 'spatial_values', side_effect=lambda s, x: (np.zeros(len(x), dtype=bool), np.empty(0))):
            p, predictor = candidate.fit(candidate.NAME, {False: (raw, [])},
                                         np.arange(60), y, np.arange(60, 75), 7)
        self.assertEqual(len(models), 19)  # 3 inner folds x 6 paths + fresh refit.
        self.assertTrue(all(len(m.rows) == 40 and m.rows <= set(range(60)) for m in models[:-1]))
        self.assertTrue(all('eval_set' not in m.kwargs for m in models))
        self.assertEqual(spatial_training_rows[-1], set(range(60)))
        self.assertEqual([len(rows) for rows in spatial_training_rows], [40, 40, 40, 60])
        for inner in range(3):
            self.assertEqual(spatial_training_rows[inner], models[inner * 6].rows)
        self.assertEqual(models[-1].rows, set(range(60)))
        self.assertEqual(models[-1].config['iterations'], 350)
        self.assertEqual(predictor.metadata['selected_configuration_index'], 0)
        self.assertEqual(len(predictor.metadata['inner_table']), 18)
        self.assertTrue(all(row['correct_count'] == 60 for row in predictor.metadata['inner_table']))
        self.assertEqual(p.shape, (15,))

    def test_outer_overlap_and_expired_budget_fail_before_selection(self):
        raw = frame(10)
        with patch.object(candidate, 'select_configuration') as selector:
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                candidate.fit(candidate.NAME, {False: (raw, [])}, np.arange(8), np.arange(8) % 2,
                              np.arange(7, 10), 7)
            selector.assert_not_called()
        candidate.DEADLINE = time.monotonic() - 1
        with patch.object(candidate, 'new_cat') as constructor:
            with self.assertRaises(candidate.StageBudgetExceeded):
                candidate.train_cat(raw, np.arange(10) % 2, [], 7, {})
            constructor.assert_not_called()

    def test_timeout_invalidates_stage_winner_and_records_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'new_run'

            def interrupted_screen(path):
                path.mkdir()
                (path / 'selection.json').write_text('{}')
                raise candidate.StageBudgetExceeded('simulated deadline')

            with patch.object(candidate.iterate, 'screen', side_effect=interrupted_screen):
                with self.assertRaises(candidate.StageBudgetExceeded):
                    candidate.run_stage('screen', out)
            self.assertFalse((out / 'selection.json').exists())
            status = json.loads((out / 'screen_status.json').read_text())
            self.assertEqual(status['status'], 'incomplete_timeout')
            self.assertFalse(status['partial_results_eligible'])

    def test_wrapped_deadline_is_also_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'new_run'

            def wrapped_failure(path):
                path.mkdir()
                candidate.DEADLINE = time.monotonic() - 1
                raise RuntimeError('native learner wrapped a Python timeout')

            with patch.object(candidate.iterate, 'screen', side_effect=wrapped_failure):
                with self.assertRaises(candidate.StageBudgetExceeded):
                    candidate.run_stage('screen', out)
            self.assertEqual(json.loads((out / 'screen_status.json').read_text())['status'], 'incomplete_timeout')

    def test_protocol_frozen_before_first_model(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / 'protocol.json').write_text(json.dumps({'sources': {}}))

            def first_model(*args):
                protocol = json.loads((out / 'protocol.json').read_text())
                self.assertEqual(len(protocol['nested_configurations_in_tie_order']), 18)
                self.assertIn('nested_structure_search.py', protocol['sources'])
                self.assertTrue((out / 'nested_structure_search.py').exists())
                return 'evaluated'

            with patch.object(candidate, 'BASE_EVALUATE', side_effect=first_model):
                self.assertEqual(candidate.evaluate(candidate.iterate.BASELINE, None, None, None, 7, out), 'evaluated')


if __name__ == '__main__':
    unittest.main()
