import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import weakref

import joblib
import numpy as np

import mitra_missing_flags_probe as probe


class TinyPredictor:
    def __init__(self,x,y):
        self.preprocessor=probe.core.MissingFlagsPreprocessor().fit(x,y,42)
        self.support=self.preprocessor.transform(x);self.labels=y.copy();self.device='mps'
        self._engine=None;self.state={'tiny':probe.base.torch.ones(1)};self.query_chunk=2318
    def predict_proba(self,x):
        p=np.where(probe.core.missing_flags(x)[:,0]>0,.6,.4)
        return np.column_stack([1-p,p])
    def release(self):self._engine=None
    def __getstate__(self):return {**self.__dict__,'device':'cpu','_engine':None}


class ProbeTests(unittest.TestCase):
    def test_real_preprocessing_and_tiny_cpu_predictor_full_release_reload(self):
        actual_tensor=probe.base.torch.as_tensor;actual_load=joblib.load;references=[]
        def fit(x,y,*args):
            model=TinyPredictor(x,y);references.append(weakref.ref(model));return model
        def load(path):
            self.assertIsNone(references[0]())
            return actual_load(path)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(probe.core,'fit_missing_flags',side_effect=fit), \
                patch.object(probe.base.torch,'as_tensor',side_effect=lambda value,device:actual_tensor(value,device='cpu')), \
                patch.object(probe.base,'snapshot',return_value={'cpu_mock':True}), \
                patch.object(probe.base,'release_memory'),patch.object(probe.joblib,'load',side_effect=load):
            result=probe.case(128,Path(folder))
            self.assertEqual(result['status'],'complete');self.assertEqual(result['actual_retained_features'],20)
            self.assertEqual(result['reload_max_difference'],0);self.assertTrue(result['cpu_mps_input_exact'])
            self.assertFalse(list(Path(folder).glob('*.joblib')))

    def test_threshold_equivalence_is_required_even_with_tiny_probability_difference(self):
        with self.assertRaises(RuntimeError):
            probe.same_predictions(np.array([[.4999999,.5000001]]),np.array([[.5000001,.4999999]]))
        with self.assertRaises(RuntimeError):
            probe.same_predictions(np.array([[.3,.7]]),np.array([[.4,.6]]))

    def test_partial_second_shape_failure_does_not_pass_or_overwrite(self):
        contract=dict(sources={},evidence_files={},environment_sha256='fixture',settings=probe.core.SETTINGS)
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'artifact';report=Path(folder)/'report'
            with patch.object(probe,'OUTPUT',out),patch.object(probe,'REPORT',report), \
                    patch.object(probe.wrapper,'check_quota'),patch.object(probe.wrapper,'freeze_contract',return_value=(contract,'fixture')), \
                    patch.object(probe.wrapper,'runtime'),patch.object(probe.wrapper,'assert_frozen'), \
                    patch.object(probe.base,'release_memory'),patch.object(probe,'case',side_effect=[{'status':'complete'},RuntimeError('second shape')]), \
                    patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError,'second shape'):probe.probe('ephemeral-session')
                import json
                result=json.loads((out/'result.json').read_text())
                self.assertEqual(result['status'],'failed_inconclusive');self.assertEqual(len(result['cases']),1)
                with self.assertRaisesRegex(RuntimeError,'no overwrite'):probe.probe('ephemeral-session')


if __name__=='__main__':unittest.main()
