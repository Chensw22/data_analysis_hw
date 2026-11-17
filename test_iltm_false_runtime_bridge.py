"""Synthetic/mocked bridge checks; no native iLTM imports, weight loads, or real data."""
import ast
from contextlib import ExitStack,nullcontext
import copy
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock,patch

import numpy as np
import pandas as pd
import joblib

import iltm_false_runtime_bridge as b


class FakeEight:
    _model_cache={}
    raise_predict=False
    def __init__(self):
        self._model=None;self.classes_=np.array([0,1])
        self.predictors_=[{'y_ctxt_superset':np.zeros(8192)} for _ in range(8)]
    def predict_proba(self,q):
        if type(self).raise_predict:raise RuntimeError('first prediction failure')
        return np.tile([.7,.3],(len(q),1))
    def _release_training_model(self):self._model=None


class BridgeTests(unittest.TestCase):
    def test_plan_fixed_scope_and_no_native_imports(self):
        plan=b.read(b.ROOT/b.PLAN);b.validate_plan(plan)
        changed=copy.deepcopy(plan);changed['second_stage']['fit']=True
        with self.assertRaises(RuntimeError):b.validate_plan(changed)
        for node in ast.parse(Path(b.__file__).read_text()).body:
            if isinstance(node,ast.Import):self.assertFalse({'torch','iltm','numpy','joblib'}&{x.name.split('.')[0] for x in node.names})
        self.assertFalse(plan['qualification']['eight_FALSE_training_repeated'])

    def test_entry_snapshot_before_import_and_worker_flag(self):
        env={k:v for k,v in os.environ.items() if k!=b.FLAG}
        with patch.dict(os.environ,env,clear=True),patch.object(b.full,'child_environment',return_value={'OMP_NUM_THREADS':'4'}):
            snapshot=b.entry_environment();self.assertEqual(snapshot[b.FLAG],'FALSE');self.assertNotIn(b.FLAG,os.environ)
            os.environ[b.FLAG]='True';self.assertEqual(snapshot[b.FLAG],'FALSE')
            with self.assertRaises(RuntimeError):b.entry_environment()
        for value in ['FALSE','True','0','']:
            with patch.dict(os.environ,{b.FLAG:value}):
                with self.assertRaises(RuntimeError):b.entry_environment()
                if value=='FALSE':b.require_false()
                else:
                    with self.assertRaises(RuntimeError):b.require_false()

    def test_scalar_trace_observes_progress_with_existing_profile(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{b.FLAG:'FALSE'}):
            audit=b.FlagAudit(folder,time.monotonic()+60)
            scope={};exec(compile('def validate_and_update():\n    return True\n',str(b.full.VENDOR/'iltm/utils.py'),'exec'),scope)
            profile_events=[]
            def profile(frame,event,arg):
                if event=='return' and frame.f_code.co_name=='validate_and_update':profile_events.append(True)
            previous_profile=sys.getprofile();previous_trace=sys.gettrace()
            try:
                sys.setprofile(profile);sys.settrace(audit)
                self.assertTrue(scope['validate_and_update']())
            finally:sys.settrace(previous_trace);sys.setprofile(previous_profile)
            self.assertEqual(audit.record['training_progress_checks'],1);self.assertEqual(profile_events,[True])
            self.assertEqual(set(audit.__dict__),{'out','deadline','record'})
            with patch.dict(os.environ,{b.FLAG:'True'}):
                with self.assertRaises(RuntimeError):audit.check('bad')

    def test_original_single_called_exactly_and_trace_restored(self):
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder);out=root/'single';old=root/'old/8693_4277';old.mkdir(parents=True)
            baseline=np.tile([.7,.3],(4277,1));np.savez(old/'probabilities.npz',first=baseline)
            stack.enter_context(patch.object(b.single,'OUTPUT',root/'old'));stack.enter_context(patch.object(b.single,'qualify'))
            stack.enter_context(patch.dict(os.environ,{b.FLAG:'FALSE'}));audit=b.FlagAudit(root,time.monotonic()+60)
            original=dict(configuration={'members':1},training_input_sha256='train',query_input_sha256='query',synthetic_labels_sha256='labels')
            frozen=dict(resource_freezes={'single':{}},original_records={'single':original})
            def run_case(path,rows,queries,freeze,deadline):
                self.assertEqual((rows,queries),(8693,4277));path.mkdir();np.savez(path/'probabilities.npz',first=baseline,reloaded=baseline)
                audit.record.update(training_progress_checks=1,original_single_stage_checks=1)
                return {**original,'status':'worker_complete'}
            previous=sys.gettrace()
            with patch.object(b.single,'run_case',side_effect=run_case) as run:
                value=b.run_single(out,frozen,time.monotonic()+60,audit);self.assertEqual(value['status'],'worker_complete');run.assert_called_once()
            self.assertIs(sys.gettrace(),previous);self.assertEqual(b.read(out/'bridge_comparison.json')['baseline_max_difference'],0.)

    def test_failed_single_preserves_native_message_and_trace(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{b.FLAG:'FALSE'}), \
             patch.object(b.single,'run_case',return_value={'status':'failed_inconclusive','error_type':'MemoryError','error':'native memory'}):
            audit=b.FlagAudit(folder,time.monotonic()+60);previous=sys.gettrace()
            with self.assertRaisesRegex(RuntimeError,'native memory'):
                b.run_single(Path(folder)/'single',{'resource_freezes':{'single':{}}},time.monotonic()+60,audit)
            self.assertIs(sys.gettrace(),previous)

    def eight_context(self,folder,stack):
        root=Path(folder);old=root/'old/8693_4277';old.mkdir(parents=True)
        FakeEight.raise_predict=False;FakeEight._model_cache.clear();joblib.dump(FakeEight(),old/'model.joblib')
        q=pd.DataFrame({'value':np.arange(4277,dtype=float)})
        np.savez(old/'probabilities.npz',first=np.tile([.7,.3],(4277,1)))
        original=dict(model_sha256=b.sha(old/'model.joblib'),query_input_sha256=b.single.frame_hash(q),
                      retained_predictor_tensors={'unique_tensor_storage_bytes':1,'dtypes':['float16']})
        stack.enter_context(patch.object(b.full,'OUTPUT',root/'old'))
        stack.enter_context(patch.object(b.full,'expected_parameters',return_value={'n_ensemble':8}))
        stack.enter_context(patch.object(b.full,'check_model',new=lambda *args:None))
        stack.enter_context(patch.object(b.full,'tensor_inventory',new=lambda *args:{'unique_tensor_storage_bytes':2,'dtypes':['float16']}))
        stack.enter_context(patch.object(b.single,'synthetic',return_value=(None,None,q)))
        native=types.ModuleType('iltm');native.iLTMClassifier=FakeEight
        pools=types.ModuleType('threadpoolctl');pools.threadpool_limits=lambda **kwargs:nullcontext()
        stack.enter_context(patch.dict(sys.modules,{'iltm':native,'threadpoolctl':pools}))
        stack.enter_context(patch.dict(os.environ,{b.FLAG:'FALSE'}))
        return dict(original_records={'full8':original}),b.FlagAudit(root,time.monotonic()+60)

    def test_eight_historical_load_complete_queries_and_portable_reload_only(self):
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            frozen,audit=self.eight_context(folder,stack)
            result=b.run_eight(Path(folder)/'eight',frozen,time.monotonic()+60,audit)
            self.assertEqual(result['status'],'complete');self.assertFalse(result['training_repeated'])
            self.assertEqual(result['partition_query_rows'],[1024,1024,1024,1024,181])
            self.assertEqual(result['reload_max_difference'],0.);self.assertEqual(FakeEight._model_cache,{})
            self.assertTrue(all(v==0 for v in result['baseline_differences'].values()))

    def test_first_eight_failure_survives_cleanup_failure(self):
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            frozen,audit=self.eight_context(folder,stack);FakeEight.raise_predict=True
            original_check=audit.check
            def check(stage,*args,**kwargs):
                if stage=='after_eight_final_release':raise RuntimeError('secondary cleanup failure')
                return original_check(stage,*args,**kwargs)
            stack.enter_context(patch.object(audit,'check',side_effect=check))
            with self.assertRaisesRegex(RuntimeError,'first prediction failure'):
                b.run_eight(Path(folder)/'eight',frozen,time.monotonic()+60,audit)
            record=b.read(Path(folder)/'eight/result.json')
            self.assertEqual(record['error'],'first prediction failure');self.assertIn('secondary',record['cleanup_error'])
            FakeEight.raise_predict=False

    def test_parent_snapshots_before_qualification_and_preserves_failed_child(self):
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            out=Path(folder)/'artifact';report=Path(folder)/'report';order=[]
            stack.enter_context(patch.object(b,'OUT',out));stack.enter_context(patch.object(b,'REPORT',report))
            stack.enter_context(patch.object(b,'child_environment',side_effect=lambda:order.append('snapshot') or {b.FLAG:'FALSE'}))
            stack.enter_context(patch.object(b,'contract',side_effect=lambda:order.append('qualification') or {}))
            process=Mock();process.poll.return_value=0
            def launch(*args,**kwargs):
                self.assertEqual(kwargs['env'][b.FLAG],'FALSE')
                b.write(out/'worker_result.json',{'status':'failed_inconclusive','error':'first native error'});return process
            stack.enter_context(patch.object(b.subprocess,'Popen',side_effect=launch))
            stack.enter_context(patch.object(b.full,'supervise',return_value=10))
            result=b.run();self.assertEqual(order,['snapshot','qualification']);self.assertFalse(result['bridge_qualified'])
            self.assertEqual(result['worker_result']['error'],'first native error')
            with self.assertRaises(FileExistsError):b.run()

    def test_qualification_rejects_partial_or_false_eight_training_claim(self):
        for record in [{'status':'failed_inconclusive'},{'status':'complete','eight_FALSE_training_repeated':True}]:
            with self.assertRaises(RuntimeError):b.qualify(record,Path('/unused'),{})
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{b.FLAG:'FALSE'}):
            with self.assertRaises(TimeoutError):b.FlagAudit(folder,time.monotonic()-1).check('expired')


if __name__=='__main__':unittest.main()
