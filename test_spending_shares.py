import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import spending_shares as candidate


class SpendingShareTests(unittest.TestCase):
    def test_missing_zero_and_original_values_preserved(self):
        frame = pd.DataFrame(dict(FoodCourt=[0., 4., 0., 4.], ShoppingMall=[0., 0., 5., np.nan],
            RoomService=[1., 0., np.nan, 2.], Spa=[np.nan, 0., 3., 4.],
            VRDeck=[2., 0., 3., 2.], RoomService_missing=[0, 0, 1, 0]), index=[12, 5, 27, 3])
        original = frame.copy(deep=True)
        result = candidate.add_shares(frame)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(result[original.columns], original)
        np.testing.assert_allclose(result.FoodWithinDiningShare, [np.nan, 1., 0., np.nan], equal_nan=True)
        np.testing.assert_allclose(result.RoomWithinLuxuryShare, [np.nan, np.nan, np.nan, .25], equal_nan=True)
        np.testing.assert_allclose(result.SpaWithinSpaVRShare, [np.nan, np.nan, .5, 2/3], equal_nan=True)

    def test_ratio_splits_equal_oblique_planes_and_preserve_scale(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.uniform(.1, 100, size=(50, 5)),
                             columns=['FoodCourt', 'ShoppingMall', 'RoomService', 'Spa', 'VRDeck'])
        shares = candidate.add_shares(frame)
        scaled = candidate.add_shares(frame.mul(np.arange(1, 51), axis=0))
        for name, cols in candidate.SHARES.items():
            a, b = frame[cols[0]], frame[cols[1:]].sum(axis=1)
            for q in [.2, .5, .8]:
                np.testing.assert_array_equal(shares[name] < q, (1-q)*a-q*b < 0)
            np.testing.assert_allclose(shares[name], scaled[name], atol=1e-15)

    def test_model_reload_keeps_transform_columns_and_missing_predictions(self):
        rng = np.random.default_rng(3)
        raw = pd.DataFrame(rng.integers(0, 40, size=(30, 5)).astype(float),
                           columns=['FoodCourt', 'ShoppingMall', 'RoomService', 'Spa', 'VRDeck'])
        raw.loc[0, :] = 0.
        raw.loc[1, 'ShoppingMall'] = np.nan
        x = candidate.add_shares(candidate.add_contrasts(raw))
        model = CatBoostClassifier(iterations=3, depth=2, verbose=False,
                                   allow_writing_files=False, thread_count=1, random_seed=7)
        model.fit(x, np.arange(len(x)) % 2)
        wrapper = candidate.SharePredictor(model, candidate.SHARES)
        before = wrapper.predict_proba(raw)
        np.testing.assert_array_equal(before, model.predict_proba(x))
        with tempfile.TemporaryDirectory(prefix='share-test-') as folder:
            path = Path(folder) / 'model.joblib'
            joblib.dump(wrapper, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(restored.predict_proba(raw), before)


if __name__ == '__main__':
    unittest.main()
