import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import structure_counts as candidate


def roster():
    return pd.DataFrame(dict(
        PassengerId=['0001_01', '0001_02', '0001_03', '0002_01'],
        HomePlanet=['Earth']*4, CryoSleep=[True, False, True, False],
        Cabin=['G/1/P', 'G/1/P', 'G/2/S', None], Destination=['TRAPPIST-1e']*4,
        Age=[20., 30., 40., 50.], VIP=[False]*4,
        RoomService=[0., 1., 2., 3.], FoodCourt=[0., 1., 2., 3.],
        ShoppingMall=[0., 1., 2., 3.], Spa=[0., 1., 2., 3.], VRDeck=[0., 1., 2., 3.],
        Name=['A A', 'B B', 'C C', 'D D'], Transported=[True, False, True, False]))


class CountModel:
    def predict_proba(self, x):
        p = (x.GroupSize + x.CabinOccupants.fillna(0)).to_numpy()/10
        return np.column_stack([1-p, p])


class StructureCountsTests(unittest.TestCase):
    def test_dataset_rejects_changed_frozen_roster(self):
        raw = roster()
        x, cats = candidate.iterate.base.features(raw, 'engineered_no_ids')
        value = (raw.iloc[:3], raw.iloc[3:].drop(columns='Transported'),
                 {False: (x, cats)}, np.arange(3))
        with patch.object(candidate, 'BASE_DATASET', return_value=value), patch.object(
                candidate, 'EXPECTED_ROSTER_HASH', 'not-the-current-roster'):
            with self.assertRaisesRegex(RuntimeError, 'membership changed'):
                candidate.dataset()

    def test_counts_ignore_target_and_preserve_missing_subset_order(self):
        raw = roster()
        lookup = candidate.RosterCounts(raw)
        altered = raw.copy()
        altered['Transported'] = ~altered.Transported
        self.assertEqual(lookup.roster_sha256, candidate.RosterCounts(altered).roster_sha256)
        counts = lookup.transform(raw.iloc[[3, 1, 0]])
        self.assertEqual(list(counts.index), [3, 1, 0])
        np.testing.assert_array_equal(counts.GroupSize, [1., 3., 3.])
        np.testing.assert_allclose(counts.CabinOccupants, [np.nan, 2., 2.], equal_nan=True)
        pd.testing.assert_frame_equal(counts.iloc[[1]], lookup.transform(raw.iloc[[1]]))

    def test_unknown_or_mutated_roster_fails_closed(self):
        raw = roster()
        lookup = candidate.RosterCounts(raw)
        for column, value in [('PassengerId', '9999_01'), ('PassengerId', '0001_04'),
                              ('Cabin', 'G/999/P'), ('Cabin', 'G/2/S')]:
            query = raw.iloc[[0]].copy()
            query.loc[0, column] = value
            with self.assertRaises(ValueError):
                lookup.transform(query)

    def test_raw_prepared_and_serialized_predictions_agree(self):
        raw = roster()
        lookup = candidate.RosterCounts(raw)
        x, _ = candidate.iterate.base.features(raw, 'engineered_no_ids')
        prepared = pd.concat([x, lookup.transform(raw)], axis=1)
        columns = candidate.add_contrasts(prepared).columns
        model = candidate.StructurePredictor(CountModel(), lookup, columns, True, True)
        subset = [2, 0, 3]
        expected = model.predict_proba(prepared.iloc[subset])
        np.testing.assert_array_equal(model.predict_raw_proba(raw.iloc[subset]), expected)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.joblib'
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(restored.predict_raw_proba(raw.iloc[subset]), expected)
            self.assertEqual(restored.lookup.roster_sha256, lookup.roster_sha256)
        with self.assertRaisesRegex(ValueError, 'Prepared prediction'):
            model.predict_proba(x)

    def test_controls_strip_counts_and_candidate_changes_only_two_columns(self):
        raw = roster()
        lookup = candidate.RosterCounts(raw)
        x, cats = candidate.iterate.base.features(raw, 'engineered_no_ids')
        augmented = pd.concat([x, lookup.transform(raw)], axis=1)
        matrices = {False: (augmented, cats)}
        observed = {}
        def fake_fit(name, matrices, train_idx, y, predict_idx, seed):
            observed[name] = matrices[False][0].copy()
            return np.array([.5]), object()
        with patch.object(candidate, 'LOOKUP', lookup), patch.object(
                candidate, 'BASE_FIT', side_effect=fake_fit), patch.object(
                candidate.spatial_expert, 'fit_spatial', side_effect=fake_fit):
            for name in [candidate.iterate.BASELINE, candidate.CONTROL, candidate.NAME]:
                candidate.fit(name, matrices, [0, 1, 2], [1, 0, 1], [3], 7)
        pd.testing.assert_frame_equal(observed[candidate.iterate.BASELINE], x)
        pd.testing.assert_frame_equal(observed[candidate.CONTROL], candidate.add_contrasts(x))
        pd.testing.assert_frame_equal(observed[candidate.NAME].drop(columns=candidate.COUNTS),
                                      observed[candidate.CONTROL])


if __name__ == '__main__':
    unittest.main()
