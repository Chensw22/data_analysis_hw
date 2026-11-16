from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import joblib
import numpy as np

import exaone_resource_probe as probe


class TinyPortable:
    def __init__(self):
        self.device='cpu';self._engine=object();self.support=np.zeros((8,13));self.labels=np.arange(8)%2
        self.metadata={'preprocessing':{'fixture':True},'last_prediction':{'forward_records':[]}}
    def release(self):self._engine=None
    def __getstate__(self):return {**self.__dict__,'device':'cpu','_engine':None}
    def predict_proba(self,q):return np.tile([.4,.6],(len(q),1))


class ResourceTests(unittest.TestCase):
    def test_probability_tolerance_and_threshold_are_both_required(self):
        original=np.array([[.2,.8],[.6,.4]])
        self.assertLess(probe.compare(original+np.array([[1e-7,-1e-7],[0,0]]),original,2e-6),2e-6)
        with self.assertRaises(RuntimeError):probe.compare(np.array([[.49999,.50001]]),np.array([[.50001,.49999]]),.002)
        with self.assertRaises(RuntimeError):probe.compare(np.array([[.3,.7]]),np.array([[.2,.8]]),.002)

    def test_portable_context_after_engine_release_and_no_temporary_residue(self):
        model=TinyPortable();q=np.zeros((17,13));p=model.predict_proba(q)
        load=joblib.load
        def checked_load(path):
            self.assertIsNone(model._engine);return load(path)
        with tempfile.TemporaryDirectory() as folder,patch.object(probe.joblib,'load',side_effect=checked_load):
            result=probe.roundtrip(model,q,p,folder,'cpu')
            self.assertEqual(result['max_difference'],0)
            self.assertFalse((Path(folder)/'portable.joblib').exists())

    def test_full_probe_failure_keeps_partial_and_cannot_claim_qualification(self):
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder);stack.enter_context(patch.object(probe,'ROOT',root))
            frozen=dict(sources={},environment_sha256='fixture',settings=probe.core.SETTINGS)
            stack.enter_context(patch.object(probe,'check_quota'))
            stack.enter_context(patch.object(probe,'contract',return_value=(frozen,'fixture')))
            stack.enter_context(patch.object(probe,'assert_frozen'))
            stack.enter_context(patch.object(probe,'cpu_stages',return_value=({'status':'complete'},np.tile([.4,.6],(33,1)))))
            stack.enter_context(patch.object(probe.core.torch.backends.mps,'is_available',return_value=True))
            stack.enter_context(patch.object(probe.core.torch.mps,'set_per_process_memory_fraction'))
            stack.enter_context(patch.object(probe,'mps_case',side_effect=[{'status':'complete','seconds':0},RuntimeError('full shape failed')]))
            stack.enter_context(patch('builtins.print'))
            with self.assertRaisesRegex(RuntimeError,'full shape failed'):probe.run('mps','private-session')
            for folder in [root/'artifacts/probes/exaone_mps',root/'reports/exaone_resources/mps']:
                result=json.loads((folder/'result.json').read_text())
                self.assertEqual(result['status'],'failed_inconclusive');self.assertFalse(result['resource_qualification'])
                self.assertEqual(len(result['stages']),2);self.assertNotIn('private-session',json.dumps(result))
            with self.assertRaisesRegex(RuntimeError,'no overwrite'):probe.run('mps','private-session')


if __name__=='__main__':unittest.main()
