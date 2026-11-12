"""Guard against target leakage and incorrect passenger-context aggregation."""
import unittest
import numpy as np
import pandas as pd
import experiment
from iterate import build_features


class GroupFeatureTests(unittest.TestCase):
    def setUp(self):
        train,_,_=experiment.load_data()
        self.frame=train.head(3).copy()
        self.frame['PassengerId']=['0001_01','0001_02','0002_01']
        self.frame['CryoSleep']=[True,False,True]
        self.frame['HomePlanet']=['Europa',None,None]

    def test_targets_cannot_change_features(self):
        a,_=build_features(self.frame,True)
        changed=self.frame.copy(); changed['Transported']=~changed.Transported
        pd.testing.assert_frame_equal(a,build_features(changed,True)[0])
        pd.testing.assert_frame_equal(a,build_features(self.frame.drop(columns='Transported'),True)[0])

    def test_peer_attributes_exclude_self_and_missing_group_is_not_filled(self):
        x,_=build_features(self.frame,True)
        self.assertEqual(x.loc[0,'PeerCryoSleepRate'],0)
        self.assertEqual(x.loc[1,'PeerCryoSleepRate'],1)
        self.assertTrue(np.isnan(x.loc[2,'PeerCryoSleepRate']))
        self.assertEqual(x.loc[1,'HomePlanet'],'Europa')
        self.assertEqual(x.loc[2,'HomePlanet'],'__MISSING__')

    def test_baseline_features_are_unchanged(self):
        pd.testing.assert_frame_equal(build_features(self.frame,False)[0],
                                      experiment.features(self.frame,'engineered_no_ids')[0])


if __name__=='__main__':
    unittest.main()
