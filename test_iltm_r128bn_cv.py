"""Lightweight synthetic and mocked checks; never construct native iLTM or load weights."""
import ast
import copy
from contextlib import ExitStack,nullcontext
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import Mock,patch

import numpy as np
import pandas as pd

import iltm_r128bn_cv as c
from test_iltm_r128bn_full8_probe import FakeClassifier,complete_native_record


def fixture():
    n=6954;x=pd.DataFrame()
    for i,name in enumerate(c.resource.COLUMNS):
        x[name]=np.array(['Earth','Mars','__MISSING__']*(n//3),dtype=object) if i in c.resource.CATEGORICAL else np.arange(n,dtype=float)
    return dict(dev=np.arange(n),y=np.arange(n,dtype=np.int64)%2,features=x,
                groups=np.arange(n).astype(str),passengers=np.arange(n).astype(str),evidence={})


def splits():
    return [(np.r_[np.arange(0,k*2318),np.arange((k+1)*2318,6954)],np.arange(k*2318,(k+1)*2318)) for k in range(3)]


def native_record(rows=4636):
    value=complete_native_record();value['training_rows']=rows
    value['inference_retrieval_rows']=[rows]*8
    for sample in value['native_observation']['sampling']:sample['input_rows']=rows
    for member in value['native_observation']['finetuning']:
        member.update(train_rows=4172,validation_rows=464,training_retrieval_rows=4172)
    return value


class FakeNative(FakeClassifier):
    _model_cache={};calls=[]
    def fit(self,x,y,eval_set=None,fit_max_time=None,return_partial_on_timeout=False):
        self.__class__.calls.append((x.copy(),y.copy(),eval_set,fit_max_time,return_partial_on_timeout))
        self._model=None;self.__class__._model_cache['one']=object()
        self.predictors_=[dict(timed_out=False,y_ctxt_superset=np.zeros(len(x))) for _ in range(8)]
        return self
    def predict_proba(self,x):return np.tile([.7,.3],(len(x),1))
    def _release_training_model(self):self._model=None


class Observer:
    def __init__(self,*args):self.records=native_record()['native_observation'];self.error=None
    def __call__(self,*args):pass


class CVTests(unittest.TestCase):
    def test_no_native_import_at_module_scope(self):
        for node in ast.parse(Path(c.__file__).read_text()).body:
            if isinstance(node,ast.Import):self.assertFalse({'torch','iltm','joblib'}&{x.name.split('.')[0] for x in node.names})
            if isinstance(node,ast.ImportFrom):self.assertNotIn(node.module.split('.')[0],['torch','iltm','joblib'])

    def test_exact_split_coverage_and_group_disjoint(self):
        inputs=fixture();folds=splits();c.validate_splits(inputs,folds,'random');c.validate_splits(inputs,folds,'group')
        bad=copy.deepcopy(folds);bad[1]=(bad[0][0],bad[0][1])
        with self.assertRaises(RuntimeError):c.validate_splits(inputs,bad,'random')
        inputs['groups'][:]='shared'
        with self.assertRaises(RuntimeError):c.validate_splits(inputs,folds,'group')

    def test_packet_roundtrip_only_outer_train_labels_and_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);inputs=fixture();a,b=splits()[0]
            meta=c.save_packet(out,inputs,a,b,'random',0);x,y,q,loaded=c.load_packet(out)
            self.assertEqual(meta,loaded);pd.testing.assert_frame_equal(x,inputs['features'].iloc[a].reset_index(drop=True))
            pd.testing.assert_frame_equal(q,inputs['features'].iloc[b].reset_index(drop=True));np.testing.assert_array_equal(y,inputs['y'][a])
            with np.load(out/'input.npz') as cached:
                arrays={k:cached[k] for k in cached.files};self.assertNotIn('query_labels',cached.files)
            arrays['query_labels']=inputs['y'][b];np.savez(out/'input.npz',**arrays)
            meta['packet_sha256']=c.sha(out/'input.npz');c.write(out/'input.json',meta)
            with self.assertRaisesRegex(RuntimeError,'only train labels'):c.load_packet(out)

    def test_packet_overlap_and_hash_change_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);inputs=fixture();a,b=splits()[0];meta=c.save_packet(out,inputs,a,b,'random',0)
            with np.load(out/'input.npz') as cached:arrays={k:cached[k].copy() for k in cached.files}
            arrays['query_ids'][0]=arrays['train_ids'][0];np.savez(out/'input.npz',**arrays)
            meta['packet_sha256']=c.sha(out/'input.npz');c.write(out/'input.json',meta)
            with self.assertRaisesRegex(RuntimeError,'boundary'):c.load_packet(out)

    def test_resource_requires_all_flags_and_recomputed_cost(self):
        worker=dict(training_rows=8693,query_rows=4277,monotonic_seconds=3000.,status='worker_complete')
        row={**worker,'sampled_tree_rss_highwater_bytes':1000}
        result=dict(status='complete',resource_execution_complete=True,cost_gate_passed=True,resource_qualification=True,
            probe_members=8,candidate_members=8,budget_seconds=7200,rss_limit_bytes=c.resource.LIMIT,cases=[row],
            projection=c.resource.projection([row]))
        monitor=dict(limit_bytes=c.resource.LIMIT,samples=4,sampled_tree_rss_highwater_bytes=1000)
        c.validate_resource(result,worker,monitor)
        for key in ['resource_execution_complete','cost_gate_passed','resource_qualification']:
            with self.assertRaises(RuntimeError):c.validate_resource({**result,key:False},worker,monitor)
        with self.assertRaises(RuntimeError):c.validate_resource({**result,'projection':{}},worker,monitor)

    def test_resource_gate_before_target_reader_and_no_overwrite(self):
        import exaone_validation_data as data
        with tempfile.TemporaryDirectory() as folder,patch.object(c,'OUTPUT',Path(folder)/'art'),patch.object(c,'REPORT',Path(folder)/'rep'), \
             patch.object(c,'resource_gate',side_effect=RuntimeError('resource pending')),patch.object(data,'load_data') as load:
            with self.assertRaisesRegex(RuntimeError,'resource pending'):c.run('random')
            load.assert_not_called();(c.OUTPUT/'random').mkdir(parents=True)
            with self.assertRaises(FileExistsError):c.run('random')

    def test_fixed_A_band_and_group_floor_are_not_stable_qualification(self):
        self.assertTrue(c.assess('random',{'accuracy':.811110152430256},{})['passed'])
        self.assertFalse(c.assess('random',{'accuracy':.811},{})['passed'])
        refs={'v71_Mitra2_zeroshot':{'accuracy':.8169398907103825},'v44_seed_ensemble':{'accuracy':.814495254529767}}
        result=c.assess('group',{'accuracy':.8069398907103825},refs)
        self.assertTrue(result['passed']);self.assertFalse(result['S_supported']);self.assertFalse(result['automatic_upload'])
        self.assertFalse(c.assess('group',{'accuracy':.806},refs)['passed'])
        refs['v71_Mitra2_zeroshot']['accuracy']+=.01
        with self.assertRaises(RuntimeError):c.assess('group',{'accuracy':.9},refs)

    def test_paired_counts_and_undefined_correlation(self):
        y=np.array([0,1,0,1]);p=np.array([.1,.8,.8,.1]);ref=np.array([.8,.8,.1,.1]);folds=[([],np.arange(4))]
        result=c.paired(p,ref,y,folds)['overall']
        self.assertEqual((result['corrected'],result['regressed'],result['both_correct'],result['both_wrong']),(1,1,1,1))
        result=c.paired(y.astype(float),ref,y,folds)['overall'];self.assertIsNone(result['error_correlation'])

    def test_group_requires_committed_complete_passed_random(self):
        import exaone_validation_data as data
        with tempfile.TemporaryDirectory() as folder,patch.object(c,'OUTPUT',Path(folder)/'art'), \
             patch.object(c,'REPORT',Path(folder)/'rep'),patch.object(c,'ROOT',Path(folder)),patch.object(c.resource,'require_committed'), \
             patch.object(data,'random_references') as refs:
            for directory in [c.OUTPUT/'random',c.REPORT/'random']:
                c.write(directory/'protocol.json',{'phase':'random'})
                c.write(directory/'summary.json',{'status':'complete','candidate':c.NAME,'completed_folds':3,'continue_to_group':False})
                c.write(directory/'assessment.json',{'passed':False})
            with self.assertRaisesRegex(RuntimeError,'passed random'):c.random_gate(fixture(),{'evidence':{}})
            refs.assert_not_called()

    def test_mock_worker_native8_train_only_eval_none_release_and_epoch_diagnostic(self):
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            out=Path(folder)/'fold0';out.mkdir();inputs=fixture();a,b=splits()[0]
            meta=c.save_packet(out,inputs,a,b,'random',0);c.write(out.parent/'protocol.json',{'frozen':{'sources':{}}})
            stack.enter_context(patch.object(c,'assert_frozen'));stack.enter_context(patch.object(c.resource,'require_committed'))
            fakepath=Path(folder)/'torch/__init__.py'
            stack.enter_context(patch.object(c.importlib.util,'find_spec',return_value=types.SimpleNamespace(origin=str(fakepath))))
            stack.enter_context(patch.dict(c.os.environ,{'DYLD_FALLBACK_LIBRARY_PATH':str(fakepath.parent/'lib')}))
            interface=types.ModuleType('iltm.inference_interface');interface.__file__=str(c.resource.VENDOR/'iltm/inference_interface.py')
            native=types.ModuleType('iltm');native.iLTMClassifier=FakeNative;native.inference_interface=interface
            torch=types.ModuleType('torch');torch.set_num_threads=Mock();torch.set_num_interop_threads=Mock()
            pools=types.ModuleType('threadpoolctl');pools.threadpool_info=lambda:[];pools.threadpool_limits=lambda **kwargs:nullcontext()
            stack.enter_context(patch.dict('sys.modules',{'iltm':native,'iltm.inference_interface':interface,'torch':torch,'threadpoolctl':pools}))
            stack.enter_context(patch.object(c.resource,'NativeObservation',Observer))
            FakeNative.calls=[];FakeNative._model_cache.clear()
            result=c.worker(out,time.monotonic()+60)
            self.assertEqual(result.get('status'),'worker_complete',result)
            self.assertEqual(result['hit_epoch_cap'],[False]*8);self.assertEqual(FakeNative._model_cache,{})
            self.assertEqual(len(FakeNative.calls),1);x,y,e,t,partial=FakeNative.calls[0]
            np.testing.assert_array_equal(y,inputs['y'][a]);self.assertEqual(len(x),4636)
            self.assertIsNone(e);self.assertIsNone(t);self.assertFalse(partial)
            p=c.validate_worker(result,meta,out);self.assertEqual(p.shape,(2318,))
            changed=copy.deepcopy(result);changed['native_observation']['finetuning'][-1]['timed_out']=True
            with self.assertRaises(RuntimeError):c.validate_worker(changed,meta,out)

    def test_worker_identity_failure_preserves_first_error_without_native_import(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(c,'assert_frozen',side_effect=RuntimeError('changed source')), \
             patch.object(c.importlib.util,'find_spec') as native:
            out=Path(folder)/'fold0';out.mkdir();c.write(out.parent/'protocol.json',{'frozen':{}})
            result=c.worker(out,time.monotonic()+60);native.assert_not_called()
            self.assertEqual(result['status'],'failed_inconclusive');self.assertEqual(result['error'],'changed source')

    def test_parent_native_failure_does_not_publish_winner(self):
        import exaone_validation_data as data
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            stack.enter_context(patch.object(c,'OUTPUT',Path(folder)/'art'));stack.enter_context(patch.object(c,'REPORT',Path(folder)/'rep'))
            stack.enter_context(patch.object(c,'resource_gate',return_value={'evidence':{}}));stack.enter_context(patch.object(c,'assert_frozen'))
            inputs=fixture();stack.enter_context(patch.object(data,'load_data',return_value=inputs))
            stack.enter_context(patch.object(data,'random_references',return_value=(splits(),{},{})))
            stack.enter_context(patch.object(c.resource,'child_environment',return_value={}))
            process=Mock();process.poll.return_value=0
            def launch(*args,**kwargs):
                out=c.OUTPUT/'random/fold0';c.write(out/'worker_result.json',dict(status='failed_inconclusive',error_type='MemoryError',error='native failure'))
                c.write(out/'partial.json',{'stage':'earlier'});return process
            stack.enter_context(patch.object(c.subprocess,'Popen',side_effect=launch))
            stack.enter_context(patch.object(c.resource,'supervise',return_value=100))
            result=c.run('random');self.assertEqual(result['status'],'failed_inconclusive')
            self.assertEqual(result['worker_result']['error'],'native failure');self.assertEqual(result['completed_folds'],0)
            self.assertFalse((c.OUTPUT/'random/oof.npz').exists());self.assertFalse((c.REPORT/'random/summary.json').exists())


if __name__=='__main__':unittest.main()
