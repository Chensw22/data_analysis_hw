import unittest

import numpy as np
import pandas as pd

import planet_marginal as candidate


class FakePosterior:
    classes_ = candidate.PLANETS
    def predict_proba(self, x):
        assert 'HomePlanet' not in x and 'Transported' not in x and 'MissingCount' not in x
        return np.tile([.2, .3, .5], (len(x), 1))


class ConstantOriginal:
    def predict_proba(self, x):
        return np.tile([.83, .17], (len(x), 1))


class PlanetConditional:
    def predict_proba(self, x):
        assert x.MissingCount.eq(7).all()
        p = x.HomePlanet.map(dict(Earth=.1, Europa=.5, Mars=.9)).to_numpy()
        return np.column_stack([1-p, p])


def frame():
    x = pd.DataFrame({c: np.ones(4) for c in candidate.Q_FEATURES}, index=[42, 8, 19, 2])
    x['HomePlanet'] = ['Earth', '__MISSING__', 'Mars', '__MISSING__']
    x['MissingCount'] = [0, 7, 2, 7]
    x['Spa_missing'] = [0, 1, 0, 1]
    return x


class PlanetMarginalTests(unittest.TestCase):
    def test_expansion_preserves_row_mass_labels_and_original_missingness(self):
        x = frame()
        before = x.copy(deep=True)
        expanded, positions, weights = candidate.expand_training(x, FakePosterior())
        np.testing.assert_allclose(np.bincount(positions, weights=weights), np.ones(4))
        np.testing.assert_array_equal(np.bincount(positions), [1, 3, 1, 3])
        np.testing.assert_array_equal(expanded.MissingCount, x.MissingCount.iloc[positions])
        np.testing.assert_array_equal(expanded.Spa_missing, x.Spa_missing.iloc[positions])
        for pos in [1, 3]:
            branch = np.flatnonzero(positions == pos)
            self.assertEqual(expanded.HomePlanet.iloc[branch].tolist(), candidate.PLANETS)
            np.testing.assert_allclose(weights[branch], [.2, .3, .5])
        pd.testing.assert_frame_equal(x, before)

    def test_observed_rows_bitwise_unchanged_and_missing_uses_all_branches(self):
        x = frame()
        predictor = candidate.PlanetMarginalPredictor(ConstantOriginal(), PlanetConditional(), FakePosterior(), {})
        p = predictor.predict_proba(x)
        np.testing.assert_array_equal(p[[0, 2]], ConstantOriginal().predict_proba(x.iloc[[0, 2]]))
        np.testing.assert_allclose(p[[1, 3], 1], .62)

    def test_posterior_excludes_target_and_planet_and_missing_count(self):
        x = frame()
        self.assertNotIn('HomePlanet', candidate.posterior_features(x))
        self.assertNotIn('MissingCount', candidate.posterior_features(x))
        with self.assertRaisesRegex(ValueError, 'Transported'):
            candidate.posterior_features(x.assign(Transported=[0, 1, 1, 0]))


if __name__ == '__main__':
    unittest.main()
