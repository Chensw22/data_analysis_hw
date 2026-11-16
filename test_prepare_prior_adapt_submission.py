from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import prepare_prior_adapt_submission as exporter


class FakePredictor:
    active = 0
    calls = []
    altered = False
    def __init__(self, p):
        self.p=p;self.labels=np.arange(8693)%2;self.device='cpu';self.live=False
    def predict_proba(self,x):
        if self.active or self.device!='mps' or len(x)!=4277:
            raise AssertionError('Must run exactly one full MPS engine.')
        type(self).active+=1;self.live=True;type(self).calls.append(self.p)
        p=np.full(len(x),self.p)
        if self.altered:p[0]=1-p[0]
        return np.column_stack([1-p,p])
    def release(self):
        if self.live:type(self).active-=1;self.live=False
    def __getstate__(self):
        return {**self.__dict__,'device':'cpu','live':False}


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patch=patch.object(exporter,'ROOT',self.root);self.patch.start()
        self.ids=pd.Series([f'{i:04d}_01' for i in range(4277)])
        self.evidence=dict(decision=dict(screen_accuracy=.82,source_protocol_sha256='protocol'),
            decision_path=exporter.DECISION,decision_sha256='decision',authorization_sha256='auth',
            sources={},evidence_files={'evidence':'digest'},committed_at='a'*40,source_protocol={},
            components=[dict(historical_export_reload_inference_verified=False),dict(historical_export_reload_inference_verified=True)])
        for (_,_,proposal),p in zip(exporter.COMPONENTS,[.4,.6]):
            directory=self.root/'artifacts'/proposal;directory.mkdir(parents=True)
            joblib.dump(FakePredictor(p),directory/'model.joblib')
            pd.DataFrame(dict(PassengerId=self.ids,probability=p)).to_csv(directory/'test_probabilities.csv',index=False)
        FakePredictor.active=0;FakePredictor.calls=[];FakePredictor.altered=False

    def tearDown(self):
        self.patch.stop();self.tmp.cleanup()
        FakePredictor.active=0;FakePredictor.altered=False

    def execute(self,duplicate=False):
        def validate(path):
            frame=pd.read_csv(path)
            self.assertEqual(len(frame),4277)
            return dict(rows=4277,sha256=exporter.sha(path),valid=True)
        train=pd.DataFrame({'PassengerId':np.arange(8693)})
        test=pd.DataFrame({'PassengerId':self.ids})
        frame=pd.DataFrame({'Age':np.arange(12970)})
        with ExitStack() as stack:
            stack.enter_context(patch.object(exporter,'decision_evidence',return_value=self.evidence))
            stack.enter_context(patch.object(exporter,'configure_runtime',return_value='fixture==1\n'))
            stack.enter_context(patch.object(exporter,'load_attributes',return_value=(train,test,frame)))
            stack.enter_context(patch.object(exporter,'check_component',side_effect=lambda *args:dict(training_ids_sha256='ids',support_labels_sha256='labels')))
            stack.enter_context(patch.object(exporter.ensemble.mitra,'environment_lock',return_value='fixture==1\n'))
            stack.enter_context(patch.object(exporter.ensemble.mitra,'release_memory'))
            stack.enter_context(patch.object(exporter.helpers,'validate_csv',side_effect=validate))
            stack.enter_context(patch.object(exporter.helpers,'duplicate_attempts',return_value=['attempt'] if duplicate else []))
            stack.enter_context(patch.object(exporter.signal,'signal',return_value=0))
            stack.enter_context(patch.object(exporter.signal,'setitimer'))
            return exporter.prepare()

    def test_sequential_complete_reload_preserves_history_and_provenance(self):
        manifest=self.execute()
        self.assertEqual(FakePredictor.calls,[.4,.6,.4,.6])
        self.assertEqual(FakePredictor.active,0)
        self.assertTrue(manifest['export_reload_inference_verified'])
        self.assertFalse(manifest['new_fit_performed'])
        self.assertFalse(manifest['component_reload_checks'][0]['historical_export_reload_inference_verified'])
        self.assertEqual(manifest['reload_max_absolute_difference'],0)
        out=self.root/'artifacts'/exporter.PROPOSAL
        cache=pd.read_csv(out/'test_probabilities.csv')
        np.testing.assert_array_equal(cache.probability,np.full(4277,.5))
        self.assertEqual(manifest['provenance']['evidence_files'],self.evidence['evidence_files'])
        for name in ['selection.json','manifest.json','authorization.json']:
            self.assertEqual(exporter.read(out/name),exporter.read(self.root/'reports'/exporter.PROPOSAL/name))

    def test_failed_component_or_duplicate_has_no_upload_outputs(self):
        for reason in ['component','duplicate']:
            with self.subTest(reason=reason):
                FakePredictor.altered=reason=='component'
                with self.assertRaises(RuntimeError):self.execute(duplicate=reason=='duplicate')
                for directory in [self.root/'artifacts'/exporter.PROPOSAL,self.root/'reports'/exporter.PROPOSAL]:
                    self.assertTrue((directory/'failure.json').exists())
                    for name in ['manifest.json','authorization.json','selection.json','submission.csv','test_probabilities.csv']:
                        self.assertFalse((directory/name).exists())
                    import shutil
                    shutil.rmtree(directory)

    def test_component_tolerance_cannot_flip_cached_mean_threshold(self):
        directory=self.root/'artifacts'/exporter.COMPONENTS[1][2]
        cache=pd.read_csv(directory/'test_probabilities.csv')
        cache['probability']=.5999998
        cache.to_csv(directory/'test_probabilities.csv',index=False)
        with self.assertRaises(RuntimeError):self.execute()
        self.assertEqual(FakePredictor.calls,[.4,.6])

    def test_decision_failure_precedes_attribute_read_and_model_load(self):
        with patch.object(exporter,'decision_evidence',side_effect=RuntimeError('uncommitted')), \
             patch.object(exporter,'load_attributes') as attrs,patch.object(exporter.joblib,'load') as load:
            with self.assertRaises(RuntimeError):exporter.prepare()
            attrs.assert_not_called();load.assert_not_called()

    def test_exact_test_id_order_required(self):
        directory=self.root/'artifacts'/exporter.COMPONENTS[0][2]
        cache=pd.read_csv(directory/'test_probabilities.csv')
        cache.iloc[::-1].to_csv(directory/'test_probabilities.csv',index=False)
        with self.assertRaisesRegex(RuntimeError,'PassengerIds'):self.execute()
        self.assertEqual(FakePredictor.calls,[])

    def test_no_target_csv_parse(self):
        import sys
        from types import SimpleNamespace
        calls=[]
        def read_csv(path,usecols):
            self.assertFalse(usecols('Transported'));self.assertTrue(usecols('PassengerId'))
            calls.append(path)
            n=8693 if path.name=='train.csv' else 4277
            return pd.DataFrame({'PassengerId':np.arange(n)})
        fake=SimpleNamespace(build_features=lambda roster,context:(pd.DataFrame({'Age':np.arange(len(roster))}),[]))
        with patch.object(exporter.pd,'read_csv',side_effect=read_csv),patch.dict(sys.modules,{'iterate':fake}):
            exporter.load_attributes()
        self.assertEqual(len(calls),2)


if __name__=='__main__':unittest.main()
