"""Real small-model checks of the complete raw spending monotonicity contract."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import monotone_spatial as candidate


def raw_frame(n=120):
    rng = np.random.default_rng(47)
    raw = pd.DataFrame(dict(
        PassengerId=[f'{i + 1:04d}_01' for i in range(n)],
        HomePlanet=np.resize(['Earth', 'Europa', 'Mars'], n),
        CryoSleep=np.arange(n) % 4 < 2,
        Cabin=[f'G/{i}/{"P" if (i // 3) % 2 else "S"}' for i in range(n)],
        Destination=['TRAPPIST-1e'] * n, Age=rng.uniform(12, 70, n), VIP=[False] * n,
        Name=[f'First{i} Last{i}' for i in range(n)]))
    for col in candidate.SIGNS:
        raw[col] = rng.uniform(0, 1000, n)
    raw.loc[::11, 'FoodCourt'] = np.nan
    raw.loc[::13, 'Spa'] = np.nan
    raw.loc[::17, 'Cabin'] = None
    raw.loc[::19, 'Age'] = np.nan
    return raw


class MonotoneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = raw_frame()
        cls.x, cls.cats = candidate.prepare_raw(cls.raw)
        cls.matrices = {False: (cls.x, cls.cats)}
        cls.a, cls.b = np.arange(90), np.arange(90, 120)
        cls.y = (np.arange(90) % 5 < 2).astype(int)
        cls.p, cls.predictor = candidate.fit(candidate.NAME, cls.matrices, cls.a, cls.y, cls.b, 7)

    def test_exact_feature_removal_constraints_and_matched_control(self):
        full = candidate.contrast_features.add_contrasts(self.x)
        slim = candidate.candidate_features(self.x)
        self.assertEqual(set(full) - set(slim), set(candidate.DROP_COLUMNS))
        self.assertEqual(len(candidate.CONSTRAINTS), 13)
        self.assertTrue(set(candidate.CONSTRAINTS) <= set(slim))
        self.assertEqual(self.predictor.columns, slim.columns.tolist())
        self.assertEqual(self.predictor.model.base_model.get_param('monotone_constraints'), candidate.CONSTRAINTS)
        _, control = candidate.fit(candidate.CONTROL, self.matrices, self.a, self.y, self.b, 7)
        self.assertEqual(control.columns, self.predictor.columns)
        self.assertEqual(control.constraints, {})
        left = self.predictor.model.base_model.get_params().copy()
        right = control.model.base_model.get_params().copy()
        left.pop('monotone_constraints')
        self.assertEqual(left, right)
        self.assertEqual(left['max_ctr_complexity'], 1)

    def test_original_v6_v34_controls_reproduce(self):
        p, _ = candidate.fit(candidate.iterate.BASELINE, self.matrices, self.a, self.y, self.b, 7)
        expected, _ = candidate.BASE_FIT(candidate.iterate.BASELINE, self.matrices, self.a, self.y, self.b, 7)
        np.testing.assert_array_equal(p, expected)
        p, _ = candidate.fit('v34_contrasts_spatial', self.matrices, self.a, self.y, self.b, 7)
        with patch.dict(candidate.iterate.CONFIGS, {'v25_spatial_blend': dict(
                context=False, model='spatial', **candidate.SPATIAL)}):
            expected, _ = candidate.contrast_features.fit('v34_contrasts_spatial', self.matrices,
                                                         self.a, self.y, self.b, 7)
        np.testing.assert_array_equal(p, expected)

    def test_full_raw_reconstruction_is_monotone_with_missing_patterns(self):
        result = candidate.audit_raw_monotonicity(self.predictor, self.raw)
        self.assertTrue(all(row['violations'] == 0 for row in result['features']))
        for row in result['features']:
            self.assertEqual(row['rows'], int(self.raw[row['feature']].notna().sum()))
        ordinary, _ = candidate.prepare_raw(self.raw)
        altered = self.raw.copy()
        observed = altered.FoodCourt.notna()
        altered.loc[observed, 'FoodCourt'] += 1000
        new, _ = candidate.prepare_raw(altered)
        a, b = candidate.candidate_features(ordinary), candidate.candidate_features(new)
        unconstrained = [col for col in a if col not in candidate.CONSTRAINTS]
        pd.testing.assert_frame_equal(a[unconstrained], b[unconstrained])
        for col, direction in candidate.CONSTRAINTS.items():
            changes = (b[col] - a[col]).dropna() * direction
            self.assertTrue((changes >= -1e-10).all())

    def test_spatial_route_unchanged_and_noneligible_stays_global(self):
        x = candidate.candidate_features(self.x.iloc[self.b])
        mask = candidate.spatial_expert.SpatialPredictor.eligible(x).to_numpy()
        global_p = self.predictor.model.base_model.predict_proba(x)[:, 1]
        np.testing.assert_array_equal(self.p[~mask], global_p[~mask])
        if mask.any():
            expert = self.predictor.model.expert.predict_proba(self.predictor.model.design(x.loc[mask]))[:, 1]
            np.testing.assert_array_equal(self.p[mask], .5 * global_p[mask] + .5 * expert)

    def test_raw_serialization_and_prepared_column_order(self):
        raw = self.raw.iloc[self.b].copy()
        raw.index = np.arange(9000, 9030)
        np.testing.assert_array_equal(self.p, self.predictor.predict_proba_raw(raw)[:, 1])
        reversed_columns = self.x.columns[::-1]
        np.testing.assert_array_equal(self.p, self.predictor.predict_proba(self.x.iloc[self.b][reversed_columns])[:, 1])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.joblib'
            joblib.dump(self.predictor, path)
            model = joblib.load(path)
            self.assertEqual(model.columns, self.predictor.columns)
            self.assertEqual(model.constraints, candidate.CONSTRAINTS)
            np.testing.assert_array_equal(self.p, model.predict_proba_raw(raw)[:, 1])

    def test_monotonicity_probe_rejects_wrong_direction(self):
        class WrongDirection:
            def predict_proba_raw(self, raw):
                p = .8 - .05 * np.log1p(raw.FoodCourt.fillna(0).to_numpy())
                return np.column_stack([1 - p, p])
        with self.assertRaisesRegex(RuntimeError, 'stop the route without encoding repairs'):
            candidate.audit_raw_monotonicity(WrongDirection(), self.raw)

    def test_sources_frozen_before_first_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / 'protocol.json').write_text(json.dumps({'sources': {}}))
            def first_model(*args):
                protocol = json.loads((out / 'protocol.json').read_text())
                self.assertIn('monotone_spatial.py', protocol['sources'])
                self.assertEqual(protocol['monotone_constraints'], candidate.CONSTRAINTS)
                return 'ready'
            with patch.object(candidate, 'BASE_EVALUATE', side_effect=first_model):
                self.assertEqual(candidate.evaluate(candidate.iterate.BASELINE, None, None, None, 7, out), 'ready')


if __name__ == '__main__':
    unittest.main()
