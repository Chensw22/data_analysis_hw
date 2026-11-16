from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import prepare_seed_mean_submission as exporter


class FakePredictor:
    active=0
    calls=[]
    def __init__(self,p,seed):
        self.p=p;self.device='cpu';self._engine=None;self.state={};self.live=False
        self.labels=np.tile([0,1],4347)[:8693]
        self.metadata=dict(seed=seed,recipe=dict(max_steps=50))
    def predict_proba(self,x):
        if FakePredictor.active:raise AssertionError('Concurrent engines')
        FakePredictor.active+=1;self.live=True;FakePredictor.calls.append(self.metadata['seed'])
        return np.tile([1-self.p,self.p],(len(x),1))
    def release(self):
        if self.live:FakePredictor.active-=1;self.live=False
    def __getstate__(self):return {**self.__dict__,'device':'cpu','live':False,'_engine':None}


class Tests(unittest.TestCase):
    def test_failed_continuation_refuses_export(self):
        record=dict(status='complete',primary_candidate=exporter.NAME,
            sole_possible_future_export=exporter.NAME,solo_upload_allowed=False,
            new_folds=[dict(fold=i) for i in range(3)],continuation=dict(passed=False))
        with self.assertRaisesRegex(RuntimeError,'passing'):exporter.require_passing_assessment(record)

    def test_no_decision_means_no_full_labels_or_fit(self):
        with patch.object(exporter.candidate,'check_quota'), \
             patch.object(exporter,'decision_evidence',side_effect=RuntimeError('no committed decision')), \
             patch.object(exporter,'read_full_data') as labels, \
             patch.object(exporter,'fit_new_component') as fit:
            with self.assertRaisesRegex(RuntimeError,'decision'):exporter.prepare(Path('unused'))
            labels.assert_not_called();fit.assert_not_called()

    def test_fit_wrapper_passes_only_new_seed_and_original_fixed_core(self):
        frame=pd.DataFrame(dict(Age=np.zeros(8693)));y=np.tile([0,1],4347)[:8693]
        with patch.object(exporter.core,'fit_full_episode') as fit:
            exporter.fit_new_component(frame,y,Path('fixture.json'))
        args,kw=fit.call_args
        self.assertEqual(args[3],20261212)
        np.testing.assert_array_equal(args[2],np.arange(8693))
        self.assertEqual(kw,dict(device='mps',log_path=Path('fixture.json'),max_steps=50))

    def test_complete_export_fits_once_and_reloads_both_components_sequentially(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory);old=root/'artifacts'/exporter.OLD_PROPOSAL;old.mkdir(parents=True)
            ids=pd.Series([f'{i:04d}_01' for i in range(4277)])
            joblib.dump(FakePredictor(.4,20261112),old/'model.joblib')
            pd.DataFrame(dict(PassengerId=ids,probability=.4)).to_csv(old/'test_probabilities.csv',index=False)
            frame=pd.DataFrame(dict(Age=np.arange(12970)));y=np.tile([0,1],4347)[:8693]
            evidence=dict(decision=dict(screen_accuracy=.82,source_protocol_sha256='protocol'),
                authorization_sha256='authorization',decision_sha256='decision',sources={},evidence_files={},committed_at='a'*40)
            stack.enter_context(patch.object(exporter,'ROOT',root))
            stack.enter_context(patch.object(exporter.candidate,'check_quota'))
            stack.enter_context(patch.object(exporter,'decision_evidence',return_value=evidence))
            stack.enter_context(patch.object(exporter,'configure_runtime',return_value='fixture==1\n'))
            stack.enter_context(patch.object(exporter.base,'environment_lock',return_value='fixture==1\n'))
            stack.enter_context(patch.object(exporter,'read_full_data',return_value=(None,pd.DataFrame(dict(PassengerId=ids)),frame,y)))
            stack.enter_context(patch.object(exporter,'check_component',side_effect=lambda *args:dict(seed=args[-1])))
            fit=stack.enter_context(patch.object(exporter,'fit_new_component',return_value=FakePredictor(.8,20261212)))
            stack.enter_context(patch.object(exporter.base,'release_memory'))
            stack.enter_context(patch.object(exporter.base,'check_deadline'))
            stack.enter_context(patch.object(exporter.signal,'signal',return_value=0))
            stack.enter_context(patch.object(exporter.signal,'setitimer'))
            stack.enter_context(patch.object(exporter.helpers,'duplicate_attempts',return_value=[]))
            stack.enter_context(patch.object(exporter.helpers,'validate_csv',side_effect=lambda p:dict(rows=4277,sha256=exporter.sha(p),valid=True)))
            FakePredictor.active=0;FakePredictor.calls=[]
            manifest=exporter.prepare(Path('unused'))
            self.assertEqual(fit.call_count,1)
            self.assertEqual(FakePredictor.calls,[20261112,20261212,20261212,20261112,20261212])
            self.assertEqual(FakePredictor.active,0)
            self.assertEqual(manifest['new_fullfit_count'],1)
            self.assertEqual(manifest['new_fullfit_seed'],20261212)
            self.assertFalse(manifest['old_component_retrained'])
            self.assertTrue(manifest['export_reload_inference_verified'])
            probabilities=pd.read_csv(root/'artifacts'/exporter.PROPOSAL/'test_probabilities.csv')
            np.testing.assert_allclose(probabilities.probability,.6)
            self.assertTrue((root/'reports'/exporter.PROPOSAL/'manifest.json').exists())


if __name__=='__main__':unittest.main()
