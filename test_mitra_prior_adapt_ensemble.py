import unittest
from unittest.mock import patch

import numpy as np

from mitra_prior_adapt_ensemble import PriorAdaptMeanPredictor, probabilities


class FakeModel:
    resident = 0

    def __init__(self, p, fail=False):
        self.p, self.fail = np.asarray(p), fail
        self.labels = np.array([0, 1])
        self.active = False

    def predict_proba(self, x):
        if FakeModel.resident:
            raise AssertionError('Two engines resident simultaneously.')
        FakeModel.resident += 1
        self.active = True
        if self.fail:
            raise RuntimeError('prediction failed')
        return self.p

    def release(self):
        if self.active:
            FakeModel.resident -= 1
            self.active = False


class Tests(unittest.TestCase):
    def test_equal_mean_and_sequential_release(self):
        a, b = FakeModel([[.8, .2], [.1, .9]]), FakeModel([[.2, .8], [.3, .7]])
        model = PriorAdaptMeanPredictor([a, b])
        with patch('mitra_prior_adapt_ensemble.mitra.check_deadline'):
            np.testing.assert_allclose(model.predict_proba([1, 2]), [[.5, .5], [.2, .8]])
        self.assertEqual(FakeModel.resident, 0)
        model.device = 'mps'
        self.assertEqual(model.__getstate__()['device'], 'cpu')

    def test_prediction_failure_releases_engine(self):
        a, b = FakeModel([[.5, .5]], fail=True), FakeModel([[.5, .5]])
        model = PriorAdaptMeanPredictor([a, b])
        with self.assertRaises(RuntimeError), patch('mitra_prior_adapt_ensemble.mitra.check_deadline'):
            model.predict_proba([1])
        self.assertEqual(FakeModel.resident, 0)

    def test_invalid_probabilities_and_label_misalignment(self):
        for value in ([[np.nan, .5]], [[.2, .2]], [[-.1, 1.1]], [[.5, .5], [.5, .5]]):
            with self.assertRaises(ValueError):
                probabilities(value, 1)
        a, b = FakeModel([[.5, .5]]), FakeModel([[.5, .5]])
        b.labels = b.labels[::-1]
        with self.assertRaises(ValueError):
            PriorAdaptMeanPredictor([a, b])


if __name__ == '__main__':
    unittest.main()
