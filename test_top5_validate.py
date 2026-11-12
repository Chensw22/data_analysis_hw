import copy
import unittest
from unittest.mock import patch

import numpy as np

from top5_validate import check_policy, cluster_interval, make_decision


class Top5ValidationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(kind=kind, seed=seed, accuracy=.82,
            vs_mitra=dict(gain=gain), vs_v44=dict(gain=-.002))
            for kind, seed, gain in [('random',20261113,.001),
                                     ('random',20261114,.0012),('group',20261112,0.)]]
        self.intervals = {'random_mean_vs_v44':dict(lower95=-.003),
                          'group_vs_v44':dict(lower95=-.004)}

    def test_small_repeatable_gain_is_only_A_not_S(self):
        result = make_decision(.82,self.rows,self.intervals)
        self.assertTrue(result['A_eligible'])
        self.assertFalse(result['S_supported'])

    def test_random_or_group_reversal_and_material_regression_fail(self):
        for index, reference, gain in [(0,'mitra',-.0001),(2,'mitra',-.0001),(1,'v44',-.0101)]:
            with self.subTest(index=index,reference=reference):
                rows = copy.deepcopy(self.rows)
                rows[index][f'vs_{reference}']['gain'] = gain
                self.assertFalse(make_decision(.82,rows,self.intervals)['A_eligible'])

    def test_tiny_average_gain_is_not_enough(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['vs_mitra']['gain'] = .0001
        rows[1]['vs_mitra']['gain'] = .0002
        self.assertFalse(make_decision(.82,rows,self.intervals)['A_eligible'])

    def test_paired_cluster_constant_difference_has_no_fake_variance(self):
        result = cluster_interval(np.repeat(.002,6),['a','a','a','b','c','c'],replicates=200)
        self.assertAlmostEqual(result['lower95'],.002)
        self.assertAlmostEqual(result['upper95'],.002)
        self.assertEqual(result['groups'],3)

    def test_authorization_is_bound_to_current_frozen_policy(self):
        self.assertIn('A_online_research_gate',check_policy())

    def test_changed_splits_and_duplicate_result_rejected(self):
        with patch('top5_validate.audit.SPLITS', [('random', 20261113)]):
            with self.assertRaises(RuntimeError):
                check_policy()
        rows = copy.deepcopy(self.rows)
        rows[1]['seed'] = 20261113
        with self.assertRaises(RuntimeError):
            make_decision(.82, rows, self.intervals)


if __name__ == '__main__':
    unittest.main()
