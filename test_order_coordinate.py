"""Tiny synthetic checks for a strictly isolated spatial-coordinate replacement."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import order_coordinate as candidate


def raw_frame(n=96):
    rng = np.random.default_rng(20261112)
    raw = pd.DataFrame(dict(
        PassengerId=[f'{i * 5 + 1:04d}_01' for i in range(n)],
        HomePlanet=['Earth'] * n, CryoSleep=[True] * n,
        Cabin=[f'G/{i}/{"P" if i % 2 else "S"}' for i in range(n)],
        Destination=['TRAPPIST-1e'] * n, Age=rng.uniform(12, 70, n),
        VIP=[False] * n, Name=[f'First{i} Last{i}' for i in range(n)]))
    for col in candidate.iterate.base.SPEND:
        raw[col] = rng.uniform(0, 100, n)
    raw.loc[::11, 'CryoSleep'] = False
    raw.loc[::13, 'Cabin'] = None
    raw.loc[::17, 'HomePlanet'] = 'Mars'
    return raw


class CoordinateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = raw_frame()
        cls.x, cls.cats = candidate.prepare_raw(cls.raw)
        cls.train_idx = np.arange(72)
        cls.predict_idx = np.arange(72, 96)
        cls.y = (np.arange(72) % 3 == 0).astype(int)
        cls.matrices = {False: (cls.x, cls.cats)}
        cls.p, cls.predictor = candidate.fit(candidate.NAME, cls.matrices,
            cls.train_idx, cls.y, cls.predict_idx, 7)

    def test_v6_and_v34_controls_reproduce_original_paths(self):
        ordinary = candidate.without_coordinate(self.matrices)
        control_p, _ = candidate.fit(candidate.iterate.BASELINE, self.matrices,
                                    self.train_idx, self.y, self.predict_idx, 7)
        original_p, _ = candidate.BASE_FIT(candidate.iterate.BASELINE, ordinary,
                                         self.train_idx, self.y, self.predict_idx, 7)
        np.testing.assert_array_equal(control_p, original_p)
        control_p, control = candidate.fit(candidate.CONTROL, self.matrices,
                                          self.train_idx, self.y, self.predict_idx, 7)
        with patch.dict(candidate.iterate.CONFIGS, {'v25_spatial_blend': dict(
                context=False, model='spatial', **candidate.SPATIAL)}):
            original_p, _ = candidate.contrast_features.fit(candidate.CONTROL, ordinary,
                self.train_idx, self.y, self.predict_idx, 7)
        np.testing.assert_array_equal(control_p, original_p)
        plain = candidate.contrast_features.add_contrasts(ordinary[False][0])
        self.assertEqual(self.predictor.cat_columns, plain.columns.tolist())
        self.assertNotIn(candidate.COORDINATE, self.predictor.base_model.feature_names_)
        np.testing.assert_array_equal(
            self.predictor.base_model.predict_proba(plain.iloc[self.predict_idx]),
            control.model.base_model.predict_proba(plain.iloc[self.predict_idx]))

    def test_original_routing_and_global_predictions_stay_exact(self):
        ordinary = candidate.without_coordinate(self.matrices)[False][0]
        mask = candidate.spatial_expert.SpatialPredictor.eligible(ordinary)
        enriched_mask = candidate.spatial_expert.SpatialPredictor.eligible(self.x)
        np.testing.assert_array_equal(mask, enriched_mask)
        self.assertFalse(mask[self.raw.Cabin.isna()].any())
        valid = self.x.iloc[self.predict_idx]
        eligible = mask.iloc[self.predict_idx].to_numpy()
        base = self.predictor.base_model.predict_proba(
            candidate.contrast_features.add_contrasts(valid)[self.predictor.cat_columns])[:, 1]
        np.testing.assert_array_equal(self.p[~eligible], base[~eligible])
        specialist = self.predictor.expert.predict_proba(self.predictor.design(valid.loc[eligible]))[:, 1]
        np.testing.assert_array_equal(self.p[eligible], .5 * base[eligible] + .5 * specialist)
        altered = valid.copy()
        altered[candidate.COORDINATE] += 10000
        np.testing.assert_array_equal(base, self.predictor.base_model.predict_proba(
            candidate.contrast_features.add_contrasts(altered)[self.predictor.cat_columns])[:, 1])

    def test_raw_serialization_and_new_row_indices(self):
        raw = self.raw.iloc[self.predict_idx].copy()
        raw.index = np.arange(9000, 9000 + len(raw))
        np.testing.assert_array_equal(self.p, self.predictor.predict_proba_raw(raw)[:, 1])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.joblib'
            joblib.dump(self.predictor, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(self.p, restored.predict_proba_raw(raw)[:, 1])
            np.testing.assert_array_equal(self.p, restored.predict_proba(self.x.iloc[self.predict_idx])[:, 1])

    def test_coordinate_does_not_depend_on_labels_or_roster(self):
        raw = self.raw.copy()
        raw['Transported'] = True
        a, cats_a = candidate.prepare_raw(raw)
        raw['Transported'] = False
        b, cats_b = candidate.prepare_raw(raw)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(cats_a, cats_b)
        for index in [4, 29, 81]:
            single, _ = candidate.prepare_raw(raw.iloc[[index]])
            self.assertEqual(single[candidate.COORDINATE].iloc[0], a[candidate.COORDINATE].iloc[index])
        raw.loc[0, 'PassengerId'] = 'invalid'
        with self.assertRaisesRegex(ValueError, 'numeric group_member'):
            candidate.prepare_raw(raw)

    def test_dataset_positional_alignment_and_control_matrix_unchanged(self):
        train = self.raw.iloc[:72].copy()
        train['Transported'] = self.y
        test = self.raw.iloc[72:].copy()
        ordinary, cats = candidate.iterate.base.features(self.raw, 'engineered_no_ids')
        with patch.object(candidate, 'BASE_DATASET', return_value=(train, test,
                {False: (ordinary, cats)}, np.arange(60))):
            _, _, matrices, _ = candidate.dataset()
        pd.testing.assert_frame_equal(matrices[False][0], self.x)
        pd.testing.assert_frame_equal(candidate.without_coordinate(matrices)[False][0], ordinary)
        misaligned = ordinary.copy()
        misaligned.index = np.arange(100, 196)
        with patch.object(candidate, 'BASE_DATASET', return_value=(train, test,
                {False: (misaligned, cats)}, np.arange(60))):
            with self.assertRaisesRegex(ValueError, 'positional'):
                candidate.dataset()

    def test_protocol_is_frozen_before_first_baseline_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / 'protocol.json').write_text(json.dumps({'sources': {}}))

            def first_model(*args):
                protocol = json.loads((out / 'protocol.json').read_text())
                self.assertIn('order_coordinate_policy', protocol)
                self.assertIn('order_coordinate.py', protocol['sources'])
                return 'ready'

            with patch.object(candidate, 'BASE_EVALUATE', side_effect=first_model):
                self.assertEqual(candidate.evaluate(candidate.iterate.BASELINE, None, None, None, 7, out), 'ready')


if __name__ == '__main__':
    unittest.main()
