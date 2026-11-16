import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tabdpt_resource_probe as probe


class TabDPTProbeTest(unittest.TestCase):
    def test_encoder_uses_training_maps_and_preserves_missing(self):
        train = pd.DataFrame({name:[1.,2.,3.] for name in probe.COMPACT})
        for name in probe.CATEGORICAL:train[name]=['earth','mars',None]
        query=train.iloc[:2].copy();query['HomePlanet']=['unseen','__MISSING__']
        encoder=probe.CompactEncoder().fit(train)
        output=encoder.transform(query)
        self.assertEqual(output.shape,(2,13));self.assertEqual(output.dtype,np.float32)
        self.assertEqual(output[0,0],-1);self.assertTrue(np.isnan(output[1,0]))
        self.assertEqual(encoder.maps['HomePlanet'],{'earth':0,'mars':1})
        self.assertTrue(np.isnan(encoder.transform(train)[2,0]))

    def test_fixed_prediction_settings_and_invalid_output(self):
        class Fake:
            def ensemble_predict_proba(self,x,**kwargs):
                self.kwargs=kwargs
                return np.tile([.4,.6],(len(x),1))
        model=Fake();p=probe.probabilities(model,np.zeros((3,13)))
        self.assertEqual(p.shape,(3,2));self.assertIsNone(model.kwargs['context_size'])
        self.assertEqual(model.kwargs['n_ensembles'],8)
        self.assertTrue(model.kwargs['permute_classes'])
        with patch.object(model,'ensemble_predict_proba',return_value=np.array([[np.nan,1.]])):
            with self.assertRaisesRegex(RuntimeError,'probability'):probe.probabilities(model,np.zeros((1,13)))

    def test_comparison_rejects_threshold_flip_within_tolerance(self):
        first=np.array([[.499999,.500001]])
        second=np.array([[.500001,.499999]])
        with self.assertRaisesRegex(RuntimeError,'differ'):probe.compare(first,second)
        self.assertEqual(probe.compare(first,first.copy()),0.)

    def test_existing_probe_not_overwritten(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(RuntimeError,'exists'):
                probe.run('cpu',Path(path))

    def test_synthetic_dimensions_and_no_real_data_access(self):
        x,y,q=probe.synthetic(20,7,42)
        self.assertEqual(x.shape,(20,13));self.assertEqual(q.shape,(7,13))
        self.assertEqual(set(y),{0,1});self.assertEqual(y.dtype,np.int64)
        np.testing.assert_array_equal(x,probe.synthetic(20,7,42)[0])


if __name__=='__main__':unittest.main()
