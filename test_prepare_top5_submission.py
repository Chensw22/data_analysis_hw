import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import pandas as pd

import prepare_top5_submission as preparation


class FakePredictor:
    altered = False
    original = None
    def __init__(self, rows):
        self.labels = np.arange(rows)%2
        self.device='mps'
        self.metadata={'test_double':True,'full_support':True}
    def release(self):
        pass
    def predict_proba(self, frame, **kwargs):
        if self.original is not None and self.original() is not None:
            raise AssertionError('Original model still resident at reload inference.')
        if self.device != 'mps':
            raise AssertionError('Reload did not restore MPS backend.')
        p=np.full(len(frame),.6)
        if self.altered:
            p[0]=.4
        return np.column_stack([1-p,p])
    def __getstate__(self):
        state=dict(self.__dict__)
        state['device']='cpu'
        return state


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.root_patch=patch.object(preparation,'ROOT',self.root)
        self.root_patch.start()
        self.run='run_fixture'
        self.run_out=self.root/'artifacts'/self.run
        self.run_out.mkdir(parents=True)
        self.name=preparation.SPECS['mitra_mirror_ensemble'][0]
        self.files=['mitra_mirror_ensemble.py','top5_validate.py','audit_top5_references.py',
            'prepare_top5_submission.py','test_prepare_top5_submission.py','kaggle_workflow.py']
        for filename in self.files:
            (self.root/filename).write_text('# isolated fixture\n')
        preparation.write(self.root/preparation.POLICY,{'fixture':'policy'})
        preparation.write(self.root/preparation.AUTH,dict(eligibility_policy=preparation.POLICY,
            eligibility_policy_sha256=preparation.sha(self.root/preparation.POLICY),
            no_duplicate_or_blind_retry_upload=True))
        sources={f:preparation.sha(self.root/f) for f in self.files+[preparation.POLICY,preparation.AUTH]}
        preparation.write(self.run_out/'protocol.json',{'sources':sources})
        self.binding=dict(original_screen_protocol_sha256=preparation.sha(self.run_out/'protocol.json'),
            validation_policy_sha256=preparation.sha(self.root/preparation.POLICY),
            authorization_sha256=preparation.sha(self.root/preparation.AUTH))
        self.eligibility=dict(status='complete',candidate=self.name,A_eligible=True,
            A_checks={'all_conditions':True},screen_accuracy=.82,S_supported=False,**self.binding)
        preparation.write(self.run_out/'stability/eligibility.json',self.eligibility)
        preparation.write(self.run_out/'stability/protocol.json',dict(candidate=self.name,
            own_source_sha256=sources['top5_validate.py'],**self.binding))
        FakePredictor.altered=False
        FakePredictor.original=None

    def tearDown(self):
        self.root_patch.stop()
        self.temporary.cleanup()
        FakePredictor.original=None

    def runtime(self):
        train=pd.DataFrame({'Transported':np.arange(8693)%2==0})
        test=pd.DataFrame({'PassengerId':[f'{i:04d}_01' for i in range(4277)]})
        matrix=pd.DataFrame({'Age':np.arange(8693+4277,dtype=float)})
        data=SimpleNamespace(SEEDS=[20261112],dataset=unittest.mock.Mock(return_value=(
            train,test,{False:(matrix,[])},None)))
        core=SimpleNamespace(environment_lock=lambda:'fake==1\n',release_memory=lambda:None,DEADLINE=None)
        candidate=SimpleNamespace(NAME=self.name,mitra=core,
            configure_runtime=unittest.mock.Mock(),assert_frozen=unittest.mock.Mock())
        def fit(name,matrices,a,y,b,seed):
            self.assertEqual(name,self.name)
            self.assertEqual(len(a),8693)
            self.assertEqual(len(y),8693)
            self.assertEqual(len(b),4277)
            self.assertFalse(np.intersect1d(a,b).size)
            model=FakePredictor(len(a))
            FakePredictor.original=weakref.ref(model)
            return np.full(len(b),.6),model
        candidate.fit=unittest.mock.Mock(side_effect=fit)
        return candidate,data

    def execute(self,candidate,data,proposal='proposal_fixture',duplicate=False):
        def validate(path):
            frame=pd.read_csv(path,dtype=str)
            self.assertEqual(len(frame),4277)
            self.assertEqual(list(frame),['PassengerId','Transported'])
            self.assertTrue(frame.Transported.isin(['True','False']).all())
            return dict(rows=4277,sha256=preparation.sha(path),valid=True)
        real_import=preparation.importlib.import_module
        def load(name,*args,**kwargs):
            return candidate if name=='mitra_mirror_ensemble' else real_import(name,*args,**kwargs)
        with patch.object(preparation.importlib,'import_module',side_effect=load), \
             patch.dict(sys.modules,{'iterate':data}), \
             patch.object(preparation.subprocess,'check_output',return_value='f'*40+'\n'), \
             patch.object(preparation,'validate_csv',side_effect=validate), \
             patch.object(preparation.signal,'signal',return_value=None), \
             patch.object(preparation.signal,'setitimer'), \
             patch.object(preparation,'duplicate_attempts',return_value=['old/attempt.json'] if duplicate else []):
            return preparation.prepare('mitra_mirror_ensemble',self.run,proposal)

    def test_eligibility_requires_exact_status_identity_and_every_binding(self):
        preparation.eligibility_evidence(self.run,self.name)
        for field,value in [('status','partial'),('candidate','other'),('A_eligible',False),
                ('A_eligible',1),('A_checks',{'one':False}),
                ('original_screen_protocol_sha256','wrong'),('validation_policy_sha256','wrong'),
                ('authorization_sha256','wrong')]:
            with self.subTest(field=field,value=value):
                changed=copy.deepcopy(self.eligibility)
                changed[field]=value
                preparation.write(self.run_out/'stability/eligibility.json',changed)
                with self.assertRaises(RuntimeError):
                    preparation.eligibility_evidence(self.run,self.name)
        preparation.write(self.run_out/'stability/eligibility.json',self.eligibility)
        (self.root/'top5_validate.py').write_text('# changed\n')
        with self.assertRaisesRegex(RuntimeError,'source not frozen'):
            preparation.eligibility_evidence(self.run,self.name)

    def test_failed_gate_never_imports_candidate_or_reads_full_labels(self):
        changed={**self.eligibility,'A_eligible':False}
        preparation.write(self.run_out/'stability/eligibility.json',changed)
        candidate,data=self.runtime()
        with self.assertRaisesRegex(RuntimeError,'A-eligible'):
            self.execute(candidate,data)
        candidate.configure_runtime.assert_not_called()
        data.dataset.assert_not_called()
        self.assert_failed()

    def assert_failed(self):
        for area in ['artifacts','reports']:
            folder=self.root/area/'proposal_fixture'
            self.assertTrue((folder/'failure.json').exists())
            for filename in ['manifest.json','authorization.json','submission.csv']:
                self.assertFalse((folder/filename).exists(),filename)

    def test_complete_roundtrip_binds_one_upload_only_and_full_shape(self):
        candidate,data=self.runtime()
        manifest=self.execute(candidate,data)
        self.assertEqual(manifest['status'],'complete')
        self.assertEqual(manifest['fit_rows'],8693)
        self.assertEqual(manifest['reload_rows'],4277)
        self.assertEqual(manifest['reload_max_absolute_difference'],0)
        self.assertTrue(manifest['export_reload_inference_verified'])
        out=self.root/'artifacts/proposal_fixture'
        authorization=preparation.read(out/'authorization.json')
        self.assertEqual(authorization['maximum_upload_attempts_for_this_file'],1)
        self.assertEqual(authorization['file_sha256'],preparation.sha(out/'submission.csv'))
        self.assertEqual(authorization['eligibility_sha256'],preparation.sha(self.run_out/'stability/eligibility.json'))
        candidate.configure_runtime.assert_called_once_with(out/'fit_logs',stage='stability')
        data.dataset.assert_called_once()
        with self.assertRaisesRegex(RuntimeError,'Proposal exists'):
            self.execute(candidate,data)
        self.assertEqual(data.dataset.call_count,1)

    def test_reload_mismatch_and_duplicate_attempts_leave_no_authorization(self):
        for duplicate in [False,True]:
            with self.subTest(duplicate=duplicate):
                if duplicate:
                    import shutil
                    shutil.rmtree(self.root/'artifacts/proposal_fixture')
                    shutil.rmtree(self.root/'reports/proposal_fixture')
                FakePredictor.altered=not duplicate
                candidate,data=self.runtime()
                with self.assertRaisesRegex(RuntimeError,'already has an upload attempt' if duplicate else 'reload inference differs'):
                    self.execute(candidate,data,duplicate=duplicate)
                self.assert_failed()

    def test_tolerance_never_allows_changed_threshold_and_nested_attempts_found(self):
        with self.assertRaisesRegex(RuntimeError,'threshold'):
            preparation.check_reload(np.array([.5-1e-7]),np.array([.5+1e-7]))
        with self.assertRaisesRegex(RuntimeError,'Incomplete'):
            preparation.probability_vector([.2],4277)
        record=self.root/'reports/batch/nested/online_attempts/one.json'
        preparation.write(record,dict(file_sha256='same'))
        self.assertEqual(preparation.duplicate_attempts('same'),[str(record.relative_to(self.root))])
        self.assertEqual(preparation.duplicate_attempts('different'),[])


if __name__=='__main__':
    unittest.main()
