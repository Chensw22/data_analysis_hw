import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import autogluon_candidate as candidate


class AutoGluonTests(unittest.TestCase):
    def test_original_feature_contract_and_fixed_fit_recipe(self):
        x = candidate.synthetic_features(40)
        candidate.validate_features(x)
        self.assertEqual(len(x.columns), 30)
        self.assertTrue(pd.options.future.infer_string)
        kwargs = candidate.fit_kwargs()
        self.assertEqual(set(kwargs['hyperparameters']), {'CAT','GBM','RF','XT'})
        self.assertEqual((kwargs['num_bag_folds'],kwargs['num_bag_sets'],kwargs['num_stack_levels']), (3,1,1))
        self.assertFalse(kwargs['ag_args_ensemble']['use_child_oof'])
        self.assertTrue(kwargs['ag_args_ensemble']['use_orig_features'])
        for name in ['dynamic_stacking','calibrate','calibrate_decision_threshold','use_bag_holdout']:
            self.assertFalse(kwargs[name])
        self.assertIsNone(kwargs['time_limit'])
        self.assertIsNone(kwargs['hyperparameter_tune_kwargs'])
        self.assertEqual(kwargs['num_gpus'], 0)
        self.assertEqual(kwargs['num_cpus'], 4)
        self.assertTrue(kwargs['fit_full_last_level_weighted_ensemble'])
        for family in ['RF','XT']:
            self.assertEqual(kwargs['hyperparameters'][family]['n_jobs'],4)
        with self.assertRaises(ValueError):
            candidate.validate_features(x.assign(PassengerId='forbidden'))

    def test_outer_validation_labels_and_rows_never_enter_fit(self):
        x = candidate.synthetic_features(30)
        external_targets = np.arange(30)%2
        seen = []
        class Artifact:
            def predict_proba(self, query):
                return np.tile([.4,.6], (len(query),1))
        def fake_fit(frame, y, ids, path, seed):
            seen.append((frame.copy(), np.asarray(y).copy(), np.asarray(ids).copy()))
            return Artifact()
        with tempfile.TemporaryDirectory() as directory, patch.object(candidate,'RUN_OUT',Path(directory)), \
                patch.object(candidate,'fit_outer',side_effect=fake_fit):
            candidate.fit(candidate.NAME,{False:(x,[])},np.arange(20),external_targets[:20],np.arange(20,30),7)
            external_targets[20:] = 1-external_targets[20:]
            candidate.fit(candidate.NAME,{False:(x,[])},np.arange(20),external_targets[:20],np.arange(20,30),7)
            self.assertTrue(seen[0][0].equals(seen[1][0]))
            np.testing.assert_array_equal(seen[0][1],seen[1][1])
            np.testing.assert_array_equal(seen[0][2],np.arange(20))
            self.assertFalse(set(seen[0][2])&set(range(20,30)))
        with patch.object(candidate,'fit_outer') as fit:
            with self.assertRaisesRegex(ValueError,'disjoint'):
                candidate.fit(candidate.NAME,{False:(x,[])},[0,1],[0,1],[1,2],7)
            fit.assert_not_called()

    def test_fit_outer_has_only_one_data_argument_and_audits_membership(self):
        x = candidate.synthetic_features(24)
        y, ids = np.arange(24)%2, np.arange(24)+101
        class FakePredictor:
            def __init__(self, **kwargs):
                self.path = Path(kwargs['path'])
                self.path.mkdir(parents=True)
                self.model_best = 'fake'
            def fit(self, **kwargs):
                self.fit_args = kwargs
                self.data = kwargs['train_data']
                return self
            def load_data_internal(self, data):
                return self.data.drop(columns=candidate.LABEL),self.data[candidate.LABEL]
            def save(self):
                pass
        with tempfile.TemporaryDirectory() as directory, patch.object(candidate,'TabularPredictor',FakePredictor), \
                patch.object(candidate,'assert_coverage',return_value=([],{})):
            artifact = candidate.fit_outer(x,y,ids,Path(directory)/'model',11)
            predictor = artifact._predictor
            self.assertEqual(set(k for k in predictor.fit_args if k.endswith('_data')), {'train_data'})
            np.testing.assert_array_equal(predictor.data.index,ids)
            np.testing.assert_array_equal(predictor.data[candidate.LABEL],y)
            self.assertTrue(x.equals(candidate.synthetic_features(24)))

    def test_original_baseline_function_is_preserved(self):
        captured = candidate.BASE_FIT
        with patch.object(candidate,'BASE_FIT',return_value=('baseline','model')) as baseline:
            result = candidate.fit(candidate.iterate.BASELINE,{},[0],[1],[2],9)
            self.assertEqual(result,('baseline','model'))
            baseline.assert_called_once_with(candidate.iterate.BASELINE,{},[0],[1],[2],9)
        self.assertIs(candidate.BASE_FIT,captured)

    def test_incomplete_family_or_oob_shortcut_is_rejected(self):
        class Incomplete:
            def leaderboard(self, silent):
                return pd.DataFrame({'model':[],'stack_level':[]})
            def info(self):
                return {'model_info':{}}
        with self.assertRaisesRegex(RuntimeError,'coverage'):
            candidate.assert_coverage(Incomplete())


if __name__ == '__main__':
    unittest.main()
