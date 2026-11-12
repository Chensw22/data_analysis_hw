"""Small API and leakage-boundary checks; never run full development CV."""
import model_families
if __name__ == '__main__':
    model_families.ensure_openmp()

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import linear_leaf_candidate as candidate


def frame(n=120):
    rng = np.random.default_rng(23)
    x = pd.DataFrame(dict(HomePlanet=['Earth'] * n, CryoSleep=['True'] * n,
                          Side=np.where(np.arange(n) % 2, 'P', 'S'),
                          CabinNumber=np.arange(n, dtype=float), Age=rng.uniform(12, 70, n)))
    for col in candidate.iterate.base.SPEND:
        x[col] = rng.uniform(0, 100, n)
    x.loc[::13, 'Age'] = np.nan
    x.loc[::17, 'Spa'] = np.nan
    return x


class LinearLeafTests(unittest.TestCase):
    def tearDown(self):
        candidate.DEADLINE = None
        candidate.RUN_OUT = None

    def test_parameter_budget_and_helper_restoration(self):
        self.assertEqual(candidate.PARAMS['n_estimators'], 700)
        self.assertEqual(candidate.PARAMS['learning_rate'], .035)
        self.assertEqual(candidate.PARAMS['num_leaves'], 15)
        for key, value in candidate.model_families.LGB.items():
            self.assertEqual(candidate.PARAMS[key], value)
        self.assertIs(candidate.BASE_FIT, candidate.model_families.ORIGINAL_FIT)
        self.assertIs(candidate.spatial_expert.ORIGINAL_FIT, candidate.BASE_FIT)
        self.assertEqual(candidate.OPTIONS, (dict(linear_tree=False),
            dict(linear_tree=True, linear_lambda=1.), dict(linear_tree=True, linear_lambda=10.),
            dict(linear_tree=True, linear_lambda=100.)))

    def test_training_only_scaler_categories_and_nan_preservation(self):
        x = candidate.contrast_features.add_contrasts(frame())
        x.loc[100:, 'HomePlanet'] = 'UnknownPlanet'
        cats = ['HomePlanet', 'CryoSleep', 'Side']
        prep = candidate.TrainingPreprocessor().fit(x.iloc[:90], cats)
        transformed = prep.transform(x)
        self.assertNotIn('UnknownPlanet', prep.vocabularies['HomePlanet'])
        self.assertTrue(transformed.HomePlanet.iloc[100:].isna().all())
        np.testing.assert_array_equal(transformed[prep.numeric].isna(), x[prep.numeric].isna())
        np.testing.assert_allclose(prep.scaler.mean_, np.nanmean(x.iloc[:90][prep.numeric], axis=0))
        self.assertEqual(transformed.columns.tolist(), x.columns.tolist())

    def test_original_cat_controls_reproduce_exactly(self):
        x = frame()
        cats = ['HomePlanet', 'CryoSleep', 'Side']
        matrices = {False: (x, cats)}
        a, b = np.arange(90), np.arange(90, 120)
        y = np.arange(90) % 2
        p, _ = candidate.fit(candidate.iterate.BASELINE, matrices, a, y, b, 7)
        expected, _ = candidate.BASE_FIT(candidate.iterate.BASELINE, matrices, a, y, b, 7)
        np.testing.assert_array_equal(p, expected)
        p, _ = candidate.fit('v34_contrasts_spatial', matrices, a, y, b, 7)
        with patch.dict(candidate.iterate.CONFIGS, {'v25_spatial_blend': dict(
                context=False, model='spatial', **candidate.SPATIAL)}):
            expected, _ = candidate.contrast_features.fit('v34_contrasts_spatial', matrices, a, y, b, 7)
        np.testing.assert_array_equal(p, expected)

    def test_real_linear_leaf_predictions_and_serialization(self):
        raw = frame()
        y = np.arange(90) % 2
        cats = ['HomePlanet', 'CryoSleep', 'Side']
        matrices = {False: (raw, cats)}
        with patch.dict(candidate.PARAMS, {'n_estimators': 20}):
            for option in [candidate.OPTIONS[0], candidate.OPTIONS[2]]:
                with self.subTest(option=option), patch.object(candidate, 'select_option',
                    return_value=(option, {'selected_option': option})):
                    p, model = candidate.fit(candidate.NAME, matrices, np.arange(90), y, np.arange(90, 120), 7)
                    self.assertTrue(np.isfinite(p).all())
                    self.assertTrue(((p >= 0) & (p <= 1)).all())
                    self.assertEqual(model.model.get_params()['linear_tree'], option['linear_tree'])
                    with tempfile.TemporaryDirectory() as folder:
                        path = Path(folder) / 'model.joblib'
                        joblib.dump(model, path)
                        np.testing.assert_array_equal(p, joblib.load(path).predict_proba(raw.iloc[90:])[:, 1])

    def test_inner_all_pipelines_are_training_only_and_refit_is_fresh(self):
        raw = frame(75)
        y = np.arange(60) % 2
        cats = ['HomePlanet', 'CryoSleep', 'Side']
        models, spatial_rows, prep_rows = [], [], []
        original_prep_fit = candidate.TrainingPreprocessor.fit

        class RecordingModel:
            def predict_proba(self, x):
                assert not self.rows & set(x.index)
                p = .2 + .6 * (x.index.to_numpy() % 2)
                return np.column_stack([1 - p, p])

        def train(x, y, option, seed):
            model = RecordingModel()
            model.rows, model.option = set(x.index), option
            models.append(model)
            return model

        def prep_fit(self, x, cats):
            prep_rows.append(set(x.index))
            return original_prep_fit(self, x, cats)

        def spatial_fit(x, y, seed):
            spatial_rows.append(set(x.index))
            return object()

        with patch.object(candidate, 'fit_model', side_effect=train), \
             patch.object(candidate.TrainingPreprocessor, 'fit', new=prep_fit), \
             patch.object(candidate, 'fit_spatial', side_effect=spatial_fit), \
             patch.object(candidate, 'spatial_predictions', side_effect=lambda s, x: (np.zeros(len(x), dtype=bool), np.empty(0))):
            p, predictor = candidate.fit(candidate.NAME, {False: (raw, cats)},
                np.arange(60), y, np.arange(60, 75), 7)
        self.assertEqual(len(models), 13)
        self.assertEqual([len(v) for v in prep_rows], [40, 40, 40, 60])
        self.assertEqual(prep_rows, spatial_rows)
        for fold in range(3):
            for model in models[fold * 4:fold * 4 + 4]:
                self.assertEqual(model.rows, prep_rows[fold])
        self.assertEqual(models[-1].rows, set(range(60)))
        self.assertEqual(models[-1].option, candidate.OPTIONS[0])
        self.assertEqual(predictor.metadata['selected_configuration_index'], 0)
        self.assertTrue(all(row['correct_count'] == 60 for row in predictor.metadata['inner_table']))

    def test_outer_overlap_rejected_and_expired_search_cannot_select(self):
        raw = frame(30)
        with patch.object(candidate, 'select_option') as selector:
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                candidate.fit(candidate.NAME, {False: (raw, [])}, np.arange(20), np.arange(20) % 2,
                              np.arange(19, 30), 7)
            selector.assert_not_called()
        candidate.DEADLINE = time.monotonic() - 1
        with patch.object(candidate, 'new_model') as constructor:
            with self.assertRaises(candidate.StageBudgetExceeded):
                candidate.fit_model(raw, np.arange(30) % 2, candidate.OPTIONS[0], 7)
            constructor.assert_not_called()

    def test_timeout_or_search_failure_leaves_no_partial_scores(self):
        for exception in [candidate.StageBudgetExceeded('deadline'), RuntimeError('failed option')]:
            with self.subTest(exception=type(exception).__name__), tempfile.TemporaryDirectory() as folder:
                out = Path(folder) / 'run'
                def interrupted(path):
                    path.mkdir()
                    (path / 'selection.json').write_text('{}')
                    (path / 'screen_results.csv').write_text('partial')
                    raise exception
                with patch.object(candidate.iterate, 'screen', side_effect=interrupted):
                    with self.assertRaises(type(exception)):
                        candidate.run_stage('screen', out)
                self.assertFalse((out / 'selection.json').exists())
                self.assertFalse((out / 'screen_results.csv').exists())
                self.assertFalse(json.loads((out / 'screen_status.json').read_text())['partial_results_eligible'])

    def test_sources_and_environment_frozen_before_first_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / 'protocol.json').write_text(json.dumps({'sources': {}}))
            def first_model(*args):
                protocol = json.loads((out / 'protocol.json').read_text())
                self.assertIn('linear_leaf_candidate.py', protocol['sources'])
                self.assertIn('model_families.py', protocol['sources'])
                self.assertIn('linear_leaf_environment_sha256', protocol)
                self.assertTrue((out / 'requirements-linear-leaf.lock.txt').exists())
                return 'ready'
            with patch.object(candidate, 'BASE_EVALUATE', side_effect=first_model):
                self.assertEqual(candidate.evaluate(candidate.iterate.BASELINE, None, None, None, 7, out), 'ready')
            candidate.check_environment(out)
            with patch.object(candidate, 'environment_lock', return_value='changed'):
                with self.assertRaisesRegex(RuntimeError, 'Environment changed'):
                    candidate.check_environment(out)


if __name__ == '__main__':
    unittest.main()
