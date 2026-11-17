"""Mock resource coverage/serialization and CPU-only supervisor guards."""
import copy
import gc
import hashlib
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock,patch
import weakref

import numpy as np
import pandas as pd
import torch

import mitra_nominal_onehot_probe as p


class Engine(torch.nn.Module):
    def forward(self,support,labels,query):return torch.zeros((1,query.shape[1],10))


class Predictor(p.base.MitraPredictor):
    previous=None
    changed_reload=False
    bad_coverage=False
    def engine(self):
        if self._engine is None:self._engine=Engine()
        return self._engine
    def release(self):self._engine=None
    def predict_proba(self,x):
        old=type(self).previous() if type(self).previous else None
        if old is not None and old is not self:raise AssertionError('Original predictor retained')
        chunk=1000 if self.bad_coverage else self.query_chunk
        support=torch.as_tensor(self.support).unsqueeze(0);labels=torch.as_tensor(self.labels).unsqueeze(0)
        for start in range(0,len(x),chunk):self.engine()(support,labels,torch.zeros((1,min(chunk,len(x)-start),19)))
        value=.4 if old is None and self.changed_reload else .6
        return np.tile([1-value,value],(len(x),1))


def completed():
    schema=dict(actual_retained_features=19,original11_exact=True,mirror_rule=p.core.MIRROR_RULE)
    result=dict(status='worker_complete',release_complete=True,settings=p.core.SETTINGS,budget_seconds=1200,
                no_accuracy_computed=True,real_targets_read=False,cases=[])
    result['tiny']=dict(status='complete',support_rows=128,query_rows=33,actual_retained_features=19,cpu_mps_input_exact=True,
        cpu_mps_max_difference=.001,threshold_equality_required=False,optimizer_steps=0,
        cpu_forward_records=p.coverage(128,33),mps_forward_records=p.coverage(128,33))
    for n,q in [(4636,2318),(8693,4277)]:
        result['cases'].append(dict(status='complete',support_rows=n,query_rows=q,actual_support_rows=n,
            actual_retained_features=19,optimizer_steps=0,schema=schema,full_support=True,cpu_mps_input_exact=True,
            subset_passed=True,reload_passed=True,reload_threshold_equal=True,portable_cpu_state=True,
            original_predictor_released=True,schema_reload_exact=True,subset_max_difference=0.,reload_max_difference=0.,
            first_forward_records=p.coverage(n,q),reload_forward_records=p.coverage(n,q),subset_forward_records=p.coverage(n,128)))
    return result


class FakeProcess:
    def __init__(self,out,report,hang=False):self.out,self.report,self.live=out,report,hang;self.pid=123;self.exitcode=0;self.killed=False;self.closed=False
    def start(self):
        for path in [self.out,self.report]:
            p.write(path/'partial.json',{'preserved':True})
            if not self.live:p.write(path/'worker_result.json',completed())
    def is_alive(self):return self.live
    def join(self,timeout=None):pass
    def kill(self):self.live=False;self.killed=True;self.exitcode=-9
    def close(self):self.closed=True


class ResourceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4);Predictor.previous=None;Predictor.changed_reload=False;Predictor.bad_coverage=False
    def tearDown(self):
        Predictor.previous=None;Predictor.changed_reload=False;Predictor.bad_coverage=False

    def frames(self,n,q):
        x=p.base.synthetic_frame(n,42);x.loc[::17,'Destination']=None
        return x,p.base.synthetic_frame(q,43)

    def fit(self,x,y,ids,seed,device):
        prep=p.core.NominalOneHotPreprocessor().fit(x,y,seed)
        model=Predictor(prep,{'fake':True},{'w':torch.ones(1)},prep.transform(x),y,device,{},2318)
        Predictor.previous=weakref.ref(model)
        return model

    def test_two_full_mock_cases_reload_schema_and_actual_forward_coverage(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(p.core,'fit_nominal_onehot',self.fit), \
             patch.object(p,'input_transfer_exact') as transfer,patch.object(p,'snapshot',return_value={}), \
             patch.object(p.base,'release_memory',side_effect=gc.collect):
            for n,q in [(4636,2318),(8693,4277)]:
                x,query=self.frames(n,q);row=p.case(x,query,Path(folder))
                self.assertEqual(row['first_forward_records'],p.coverage(n,q))
                self.assertEqual(row['reload_forward_records'],p.coverage(n,q))
                self.assertEqual(row['reload_max_difference'],0.)
                self.assertTrue(row['schema_reload_exact']);self.assertTrue(row['original_predictor_released'])
                self.assertIsNone(Predictor.previous())
                self.assertFalse((Path(folder)/f'portable_{n}.joblib').exists())
            self.assertEqual(transfer.call_count,4)

    def test_bad_reload_or_partial_context_does_not_pass(self):
        x,q=self.frames(4636,2318)
        with tempfile.TemporaryDirectory() as folder,patch.object(p.core,'fit_nominal_onehot',self.fit), \
             patch.object(p,'input_transfer_exact'),patch.object(p,'snapshot',return_value={}), \
             patch.object(p.base,'release_memory',side_effect=gc.collect):
            Predictor.changed_reload=True
            with self.assertRaises(RuntimeError):p.case(x,q,Path(folder))
            Predictor.changed_reload=False;Predictor.bad_coverage=True
            with self.assertRaises(RuntimeError):p.case(x,q,Path(folder))
            self.assertFalse(list(Path(folder).glob('*.joblib')))

    def test_cross_backend_threshold_flip_allowed_but_reload_flip_forbidden(self):
        result=completed();result['tiny']['cpu_mps_max_difference']=.019
        self.assertTrue(p.qualify(result))
        for field,value in [('cpu_mps_max_difference',.021),('threshold_equality_required',True)]:
            bad=copy.deepcopy(result);bad['tiny'][field]=value
            with self.assertRaises(RuntimeError):p.qualify(bad)
        first=np.array([[.5000001,.4999999]]);second=np.array([[.4999999,.5000001]])
        with self.assertRaises(RuntimeError):p.same_predictions(first,second)

    def test_qualify_refuses_missing_case_partial_reload_and_not19(self):
        for mutation in ['partial','reload','features','support','release']:
            result=completed()
            if mutation=='partial':result['cases'].pop()
            if mutation=='reload':result['cases'][1]['reload_forward_records']=p.coverage(8693,128)
            if mutation=='features':result['cases'][0]['actual_retained_features']=18
            if mutation=='support':result['cases'][0]['actual_support_rows']=1000
            if mutation=='release':result['release_complete']=False
            with self.assertRaises(RuntimeError):p.qualify(result)

    def test_attributes_reader_explicitly_excludes_targets(self):
        rows=[]
        def reader(path,**kwargs):
            self.assertEqual(kwargs['usecols'],p.ATTRIBUTES);self.assertNotIn('Transported',kwargs['usecols'])
            n=8693 if str(path).endswith('train.csv') else 4277
            frame=pd.DataFrame({'PassengerId':np.arange(n)});rows.append(n);return frame
        features=pd.DataFrame({'fake':np.zeros(12970)})
        with patch.object(p.pd,'read_csv',side_effect=reader),patch.object(p.data,'compact_features',return_value=features), \
             patch.object(p.data,'feature_hash',return_value='proof'),patch.object(p,'read',return_value={
                 'status':'exact_compact13_match','holdout_targets_read':False,'compact_values_sha256':'proof'}):
            a,b=p.attributes_only()
        self.assertEqual(rows,[8693,4277]);self.assertEqual((len(a),len(b)),(8693,4277))

    def test_input_transfer_detects_signed_zero_bit_change(self):
        value=np.array([[0.,-0.]],np.float32)
        with patch.object(p.base.torch,'as_tensor',return_value=torch.tensor([[0.,0.]])):
            with self.assertRaises(RuntimeError):p.input_transfer_exact(value)

    def test_freeze_rejects_source_or_env_drift_without_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);file=root/'source.py';file.write_text('before')
            frozen=dict(sources={'source.py':p.base.sha256(file)},evidence_files={},
                environment_sha256=hashlib.sha256(b'pin').hexdigest(),settings=p.core.SETTINGS,backend=p.base.SETTINGS,checkpoint_sha256=p.base.WEIGHT_SHA)
            with patch.object(p,'ROOT',root),patch.object(p.base,'environment_lock',return_value='pin'),patch.object(p.base,'verify_checkpoint') as weights:
                p.check_frozen(frozen);weights.assert_called_once()
                file.write_text('after')
                with self.assertRaises(RuntimeError):p.check_frozen(frozen)
                weights.assert_called_once()

    def fake_run(self,out,report,process):
        context=Mock();context.Process.return_value=process
        with patch.object(p,'OUTPUT',out),patch.object(p,'REPORT',report),patch.object(p,'check_quota'), \
             patch.object(p,'contract',return_value=({'sources':{}},'fixture')),patch.object(p,'check_frozen'), \
             patch.object(p.multiprocessing,'get_context',return_value=context):
            return p.run('PRIVATE_SESSION')

    def test_parent_success_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            out,report=Path(folder)/'artifacts',Path(folder)/'reports';process=FakeProcess(out,report)
            result=self.fake_run(out,report,process)
            self.assertTrue(result['resource_qualification']);self.assertTrue(process.closed)
            for name in ['freeze.json','result.json','worker_result.json','requirements.lock']:
                self.assertEqual((out/name).read_bytes(),(report/name).read_bytes())
            with self.assertRaises(FileExistsError):self.fake_run(out,report,process)
            self.assertNotIn('PRIVATE_SESSION',(out/'result.json').read_text())

    def test_timeout_preserves_partial_and_never_reports_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            out,report=Path(folder)/'artifacts',Path(folder)/'reports';process=FakeProcess(out,report,hang=True)
            with patch.object(p,'supervise',side_effect=TimeoutError('external budget')):
                with self.assertRaises(TimeoutError):self.fake_run(out,report,process)
            self.assertTrue(process.killed);self.assertTrue(process.closed)
            self.assertEqual(p.read(out/'partial.json'),{'preserved':True})
            self.assertFalse(p.read(out/'result.json')['resource_qualification'])
            self.assertTrue(p.read(out/'external_timeout.json')['child_reaped'])

    def test_deadline_supervisor_kills_only_its_worker(self):
        process=Mock();process.is_alive.side_effect=[True,True,False]
        with self.assertRaises(TimeoutError):p.supervise(process,time.monotonic()-.1)
        process.kill.assert_called_once_with();process.join.assert_called_once_with(timeout=5)

    def test_quota_or_uncommitted_guard_never_spawns(self):
        with tempfile.TemporaryDirectory() as folder:
            out,report=Path(folder)/'artifacts',Path(folder)/'reports'
            with patch.object(p,'OUTPUT',out),patch.object(p,'REPORT',report),patch.object(p,'check_quota',side_effect=RuntimeError('quota')),patch.object(p.multiprocessing,'get_context') as spawn:
                with self.assertRaises(RuntimeError):p.run('PRIVATE_SESSION')
                spawn.assert_not_called();self.assertFalse(out.exists())
            with patch.object(p,'OUTPUT',out),patch.object(p,'REPORT',report),patch.object(p,'check_quota'),patch.object(p,'contract',side_effect=RuntimeError('uncommitted')),patch.object(p.multiprocessing,'get_context') as spawn:
                with self.assertRaises(RuntimeError):p.run('PRIVATE_SESSION')
                spawn.assert_not_called();self.assertFalse(p.read(out/'result.json')['resource_qualification'])


if __name__=='__main__':unittest.main()
