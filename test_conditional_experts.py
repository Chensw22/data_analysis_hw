"""Check state routing and the expert selection/refit validation boundary."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import conditional_experts as candidate


class ConstantPredictor:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, x):
        return np.tile([1 - self.p, self.p], (len(x), 1))


def frame(states):
    x = pd.DataFrame({'CryoSleep': states})
    for col in candidate.iterate.base.SPEND:
        x[col] = 1.
    return x


class ExpertTests(unittest.TestCase):
    def test_unknown_is_exact_global_and_states_are_disjoint(self):
        x = frame(['True', 'False', '__MISSING__', None, 'unexpected', True, False])
        predictor = candidate.ConditionalPredictor(
            ConstantPredictor(.3), {True: ConstantPredictor(.9), False: ConstantPredictor(.1)}, {})
        p = predictor.predict_proba(x)[:, 1]
        np.testing.assert_allclose(p, [.6, .2, .3, .3, .3, .6, .2])
        unknown = ~(candidate.state_mask(x, True) | candidate.state_mask(x, False))
        self.assertFalse((candidate.state_mask(x, True) & candidate.state_mask(x, False)).any())
        np.testing.assert_array_equal(p[unknown], np.full(3, .3))

    def test_inner_selection_and_fresh_state_only_refit(self):
        x = frame(['True'] * 40 + ['False'] * 40)
        x.index = np.arange(100, 180)
        y = np.tile([0, 1], 40)
        models = []

        class RecordingCat:
            def fit(self, features, labels, **kwargs):
                self.rows = set(features.index)
                self.kwargs = kwargs
                return self

            def get_best_iteration(self):
                return 6

            def get_best_score(self):
                return {'validation': {'Logloss': .4}}

        def create(cats, seed, iterations):
            model = RecordingCat()
            model.iterations = iterations
            models.append(model)
            return model

        with patch.object(candidate, 'new_cat', side_effect=create):
            fitted, audit = candidate.fit_expert(x, y, [], 7, True)
        self.assertEqual(len(models), 2)
        self.assertIs(fitted, models[1])
        self.assertEqual(models[0].iterations, 700)
        self.assertEqual(models[1].iterations, 7)
        validation_rows = set(models[0].kwargs['eval_set'][0].index)
        self.assertFalse(models[0].rows & validation_rows)
        self.assertEqual(models[0].rows | validation_rows, set(range(100, 140)))
        self.assertEqual(models[1].rows, set(range(100, 140)))
        self.assertEqual(models[1].kwargs, {})
        self.assertEqual(audit['state_train_rows'], 40)
        self.assertEqual(audit['selected_iterations'], 7)

    def test_one_class_expert_falls_back_without_training(self):
        x = frame(['True'] * 20)
        with patch.object(candidate, 'new_cat') as constructor:
            model, audit = candidate.fit_expert(x, np.ones(20), [], 7, True)
        self.assertIsNone(model)
        self.assertIn('global_only', audit['status'])
        constructor.assert_not_called()
        predictor = candidate.ConditionalPredictor(ConstantPredictor(.3), {True: model}, {})
        np.testing.assert_array_equal(predictor.predict_proba(x), ConstantPredictor(.3).predict_proba(x))


if __name__ == '__main__':
    unittest.main()
