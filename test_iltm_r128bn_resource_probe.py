"""CPU/synthetic mocks only: no native iLTM imports, weight loads or model creation."""
import ast
import copy
from contextlib import ExitStack
import gc
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import tempfile
import time
import types
import unittest
from unittest.mock import Mock,patch

import numpy as np

import iltm_r128bn_resource_probe as p


class FakeClassifier:
    def __init__(self,*,device='cpu',n_ensemble=8,seed=3,cat_features=None,checkpoint='r128bn',
                 adaptive_memory=True,gpu_oom_retries=2,auto_stop_on_low_cpu_memory=True,
                 auto_amp_on_low_mem=False,auto_disable_retrieval_on_low_mem=False,
                 bottleneck_size=0,tree_model='XGBoost',preprocessing='realmlp_td_s_v0',tree_embedding=True,
                 concat_tree_with_orig_features=True,clip_predictions=True,normalize_predictions=False):
        self.params=locals().copy();self.params.pop('self');self.params.update(**p.OVERRIDES,clip_predictions=False)
        self.device=device;self.inference_chunk_rows=10000;self.retrieval_context_max_rows=8192
        self.predictors_=[{'timed_out':False}];self.preprocessors_=[{}];self.classes_=np.array([0,1]);self._fit_succeeded=True
    def get_params(self,deep=False):return copy.deepcopy(self.params)
    def predict_proba(self,x):
        value=np.asarray(x,dtype=float)[:,0];prob=np.where(value>=0,.7,.3)
        return np.column_stack([1-prob,prob])


def fake_fitted():
    model=FakeClassifier(device='cpu',n_ensemble=1,seed=p.SEED,cat_features=p.CATEGORICAL,checkpoint=str(p.WEIGHT),**p.FAIL_CLOSED)
    return model,p.expected_parameters(FakeClassifier)


class ProbeTests(unittest.TestCase):
    def test_fixed_new_plan_and_no_quota_or_checkpoint_rescue(self):
        original=json.loads((p.ROOT/'reports/iltm_resource_design.json').read_text())
        plan={**original,'checkpoint_filename':'r128bn.pth','checkpoint_revision':p.REVISION,
              'checkpoint_bytes':p.CHECKPOINT_BYTES,'checkpoint_sha256':p.CHECKPOINT_SHA,
              'checkpoint_overrides':p.OVERRIDES,
              'setup_evidence':{**original['setup_evidence'],p.SETUP:'0'*64}}
        p.validate_plan(plan)
        for key,value in [('checkpoint_filename','xgbrconcat.pth'),('checkpoint_revision','main'),
                ('resource_monotonic_seconds',2400),('probe_member_count',8),('automatic_cv',True),
                ('checkpoint_overrides',{**p.OVERRIDES,'bottleneck_size':64})]:
            with self.assertRaises(RuntimeError):p.validate_plan({**plan,key:value})
        self.assertNotIn('session_file',inspect.signature(p.run).parameters)
        self.assertNotIn('quota_guard',Path(p.__file__).read_text())
        params=p.expected_parameters(FakeClassifier)
        for key,value in p.OVERRIDES.items():self.assertEqual(params[key],value)
        self.assertEqual(params['tree_model'],'XGBoost')

    def test_setup_checkpoint_and_original_environment_reference(self):
        plan=json.loads((p.ROOT/'reports/iltm_resource_design.json').read_text())
        setup=json.loads((p.ROOT/p.SETUP).read_text())
        p.validate_setup(plan,setup)
        for group,key,value in [('checkpoint','filename','xgbrconcat.pth'),('checkpoint','sha256','0'*64),
                               ('environment','reference_field','unapproved_environment')]:
            changed=copy.deepcopy(setup);changed[group][key]=value
            with self.assertRaises(RuntimeError):p.validate_setup(plan,changed)

    def test_unchanged_resource_helpers_are_ast_identical_to_reviewed_probe(self):
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        old=functions(p.ROOT/'iltm_resource_probe.py');new=functions(Path(p.__file__))
        for name in ['NativeObservation','child_environment','synthetic','check_model','probabilities',
                     'compare','tensor_inventory','process_tree','terminate','supervise','qualify','projection']:
            self.assertEqual(old[name],new[name],name)

    def test_no_native_imports_at_module_scope(self):
        tree=ast.parse(Path(p.__file__).read_text())
        forbidden={'torch','iltm','joblib','pandas','xgboost','catboost','mitra_candidate','tabdpt'}
        for node in tree.body:
            if isinstance(node,ast.Import):self.assertFalse(forbidden & {x.name.split('.')[0] for x in node.names})
            if isinstance(node,ast.ImportFrom):self.assertNotIn(node.module.split('.')[0],forbidden)

    def test_synthetic_shape_schema_determinism_no_query_labels(self):
        a,y,q=p.synthetic(4636,2318);b,z,r=p.synthetic(4636,2318)
        self.assertEqual(a.shape,(4636,13));self.assertEqual(q.shape,(2318,13));self.assertEqual(list(a.columns),p.COLUMNS)
        self.assertEqual(p.frame_hash(a),p.frame_hash(b));self.assertEqual(p.frame_hash(q),p.frame_hash(r))
        np.testing.assert_array_equal(y,np.arange(4636)%2);np.testing.assert_array_equal(y,z)
        self.assertNotIn('Transported',a);self.assertTrue(a.isna().any().all())

    def test_configuration_change_and_native_partial_fail_closed(self):
        model,expected=fake_fitted();p.check_model(model,expected,True)
        model.params['adaptive_memory']=True
        with self.assertRaises(RuntimeError):p.check_model(model,expected,True)
        model.params=copy.deepcopy(expected)
        for predictors in [[],[{'timed_out':True}],[{'timed_out':False}]*2]:
            model.predictors_=predictors
            with self.assertRaises(RuntimeError):p.check_model(model,expected,True)
        model.predictors_=[{'timed_out':False}];model.classes_=np.array([1,0])
        with self.assertRaises(RuntimeError):p.check_model(model,expected,True)

    def test_probability_persistence_partition_permutation_and_threshold(self):
        model,_=fake_fitted();query=np.arange(-1100,1218).reshape(-1,1)
        first=p.probabilities(model,query)
        self.assertEqual(p.compare(first,p.probabilities(model,query[::-1])[::-1],2e-5),0.)
        split=np.concatenate([p.probabilities(model,query[i:i+1024]) for i in range(0,len(query),1024)])
        self.assertEqual(p.compare(first,split,2e-5),0.)
        restored=pickle.loads(pickle.dumps(model));self.assertEqual(p.compare(first,p.probabilities(restored,query),2e-6),0.)
        with self.assertRaises(RuntimeError):p.compare(np.array([[.5000001,.4999999]]),np.array([[.4999999,.5000001]]),2e-6)
        for bad in [np.full((2318,2),np.nan),np.full((2318,2),.8),np.ones((2318,3)),np.tile([-1.,2.],(2318,1))]:
            with patch.object(model,'predict_proba',return_value=bad):
                with self.assertRaises(RuntimeError):p.probabilities(model,query)

    def test_return_observer_copies_only_scalars_and_does_not_mutate(self):
        root=Path('/synthetic/native');observer=p.NativeObservation(root);changed=[]
        loc=dict(max_epochs=2048,epoch=7,global_batch_idx=16,stop_training=True,timeout_triggered=False,
                 es={'checks_done':9,'best_val_metric':.9},X_train=np.zeros((20,3)),y_val=np.zeros(3),y_ctxt_superset=np.zeros(20))
        frame=types.SimpleNamespace(f_code=types.SimpleNamespace(co_filename=str(root/'utils.py'),co_name='fine_tune_main_network'),f_locals=loc)
        result={'timed_out':False};snapshot=pickle.dumps(loc)
        observer(frame,'call',None);observer(frame,'return',result)
        self.assertEqual(snapshot,pickle.dumps(loc));self.assertEqual(result,{'timed_out':False})
        record=observer.records['finetuning'][0]
        self.assertEqual((record['epochs_entered'],record['optimizer_updates']),(8,16));self.assertTrue(record['selected_weights_restored'])
        self.assertNotIn('best_val_metric',json.dumps(observer.records));json.dumps(observer.records)
        self.assertEqual(observer.starts,{})
        parent=types.SimpleNamespace(f_locals={'epoch':6,'global_batch_idx':14})
        validate=types.SimpleNamespace(f_code=types.SimpleNamespace(co_filename=str(root/'utils.py'),co_name='validate_and_update'),
                                      f_locals={'where':'epoch7'},f_back=parent)
        observer(validate,'call',None);observer(validate,'return',True)
        self.assertEqual(observer.records['best_weight_positions'],[{'where':'epoch7','epoch_index':6,'updates':14}])

    def test_ps_tree_only_numeric_descendants_and_rss_units(self):
        text='100 1 1024\n101 100 2048\n102 101 4096\n200 1 8192\nbad line\n'
        with patch.object(p.subprocess,'check_output',return_value=text):value=p.process_tree(100)
        self.assertEqual(value,{'pids':[100,101,102],'rss_bytes':7168*1024})

    def test_supervisor_rss_and_deadline_kill_own_group_preserve_monitor(self):
        with tempfile.TemporaryDirectory() as folder:
            for exceeded in ['rss','deadline']:
                process=Mock();process.pid=200;process.poll.return_value=None
                with patch.object(p,'process_tree',return_value={'pids':[200,201],'rss_bytes':p.LIMIT+1 if exceeded=='rss' else 10}), \
                     patch.object(p,'terminate') as terminate:
                    with self.assertRaises(MemoryError if exceeded=='rss' else TimeoutError):
                        p.supervise(process,time.monotonic()+10 if exceeded=='rss' else time.monotonic()-1,folder)
                    terminate.assert_called_once_with(process,[200,201])
                    self.assertTrue((Path(folder)/'external_monitor.json').exists())

    def test_projection_counts_eight_retained_predictors_without_eight_checkpoints(self):
        records=[]
        for n,q in p.CASES:
            records.append(dict(training_rows=n,native_observation={'loads':[{'seconds':20}]},fit_seconds=70,
                full_prediction_seconds=2,save_seconds=1,reload_seconds=3,portable_bytes=100,
                retained_predictor_tensors={'unique_tensor_storage_bytes':200},sampled_tree_rss_highwater_bytes=1000))
        result=p.projection(records);self.assertTrue(result['passed'])
        self.assertEqual(result['cases'][0]['seconds'],1.25*(20+8*56))
        self.assertEqual(result['cases'][0]['memory_bytes'],1.25*(1000+7*200))
        records[1]['fit_seconds']=10000;self.assertFalse(p.projection(records)['passed'])
        records[1]['fit_seconds']=70;records[0]['portable_bytes']=p.LIMIT;self.assertFalse(p.projection(records)['passed'])

    def test_qualification_recomputes_saved_arrays_and_rejects_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);first=np.tile([.7,.3],(2318,1));idx=np.arange(0,2318,7)
            np.savez(out/'probabilities.npz',first=first,reloaded=first,permuted=first,partition=first,subset=first[idx],subset_ids=idx)
            (out/'model.joblib').write_bytes(b'synthetic pickle placeholder')
            record=dict(training_rows=4636,query_rows=2318,status='worker_complete',release_complete=True,probe_members=1,
                candidate_members=8,full_prediction_rows=2318,reload_rows=2318,original_released_before_reload=True,
                portable_cpu=True,no_real_data=True,no_query_labels=True,no_accuracy_reported=True,
                model_sha256=p.sha(out/'model.joblib'),probabilities_sha256=p.sha(out/'probabilities.npz'))
            self.assertTrue(p.qualify(record,out))
            for change in [dict(status='partial'),dict(probe_members=8),dict(reload_rows=2317)]:
                with self.assertRaises(RuntimeError):p.qualify({**record,**change},out)
            (out/'probabilities.npz').write_bytes(b'changed')
            with self.assertRaises(RuntimeError):p.qualify(record,out)

    def test_parent_preserves_worker_first_error_separately_from_partial(self):
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            out=Path(folder)/'artifact';report=Path(folder)/'report'
            for key,value in [('OUTPUT',out),('REPORT',report)]:stack.enter_context(patch.object(p,key,value))
            stack.enter_context(patch.object(p,'contract',return_value={'sources':{}}))
            stack.enter_context(patch.object(p,'assert_frozen'));stack.enter_context(patch.object(p,'child_environment',return_value={}))
            process=Mock();process.poll.return_value=0
            def popen(*args,**kwargs):
                case=out/'4636_2318'
                p.write(case/'worker_result.json',dict(status='failed_inconclusive',error_type='MemoryError',error='first native failure',current_stage='native_fit'))
                p.write(case/'partial.json',dict(current_stage='earlier_partial'))
                return process
            stack.enter_context(patch.object(p.subprocess,'Popen',side_effect=popen))
            stack.enter_context(patch.object(p,'supervise',return_value=100))
            result=p.run()
            self.assertEqual(result['failed_child_record']['error'],'first native failure')
            self.assertEqual(result['failed_child_partial']['current_stage'],'earlier_partial')
            self.assertFalse(result['resource_qualification']);self.assertNotIn('PRIVATE_SESSION',json.dumps(result))
            self.assertFalse((out/'8693_4277').exists())

    def test_no_overwrite_and_frozen_source_before_native_import(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'exists';out.mkdir()
            with patch.object(p,'OUTPUT',out),patch.object(p,'contract') as contract:
                with self.assertRaises(FileExistsError):p.run()
                contract.assert_not_called()
            target=Path(folder)/'case'
            with patch.object(p,'assert_frozen',side_effect=RuntimeError('identity changed')),patch.object(p.importlib.util,'find_spec') as spec:
                result=p.run_case(target,4636,2318,{},time.monotonic()+60)
                spec.assert_not_called()
            self.assertEqual(result['error'],'identity changed');self.assertEqual(result['current_stage'],'before_native_import')
            self.assertEqual(result['status'],'failed_inconclusive')


if __name__=='__main__':unittest.main()
