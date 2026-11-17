"""Synthetic CPU/mocked-device sequence tests; no real attributes, weights or MPS."""
import copy
from contextlib import ExitStack
import gc
import tempfile
from pathlib import Path
import time
import unittest
from unittest.mock import Mock,patch

import numpy as np
import pandas as pd
import torch

import mitra_support3072_gc_probe as p


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.ModuleList([torch.nn.Linear(13,4),torch.nn.Linear(4,10)])
    def forward(self,support,labels,query,*args):
        return self.layers[1](torch.tanh(self.layers[0](query)))


def proof_fixture(step=1):
    state=dict(model_object_id=1,optimizer_object_id=2,model_training=True,module_training={'':True},
        parameters={'synthetic':'fixture'},buffers={},all_gradients_none=True,all_input_gradients_none=True,
        optimizer_steps={'synthetic':float(step)},optimizer_groups=[],optimizer_state={'synthetic':'fixture'},
        inputs={'data':{},'targets':{},'query':{}},rng={name:{} for name in ['python_complete_state_sha256',
            'numpy_global_complete_state_sha256','sampler','counterfactual_sampler','torch_cpu','torch_mps']})
    return dict(status='complete',state_preserved=True,barrier=p.guard.BARRIER,device='mps',gc_collected=2,
        before=state,after=copy.deepcopy(state),observations=[{'stage':stage} for stage in [
        'before_pre_hash','after_pre_hash','barrier_after_synchronize1','barrier_after_gc_collect',
        'barrier_after_empty_cache','barrier_after_synchronize2','before_post_hash','after_post_hash']])


def stage_fixture(out,kind):
    n,q=(3708,928) if kind=='small' else (6954,1739);steps=[1] if kind=='small' else [1,2]
    stages=['V0','V1_before_barrier','V1_after_barrier'] if kind=='small' else ['V0','V1','V2']
    counters=[[],[1],[1]] if kind=='small' else [[],[1],[2]]
    result=dict(status='worker_complete',stage=kind,release_complete=True,real_targets_read=False,no_accuracy_computed=True,
        exact_replay_of_real_label_split=False,support_rows=n,query_rows=q,optimizer_identity_persistent=True,
        resources_after_release={'mps_current_bytes':0},updates=[],validations=[],barriers=[],query_ids_sha256=p.base.ids_hash(np.arange(q)))
    for step in steps:
        result['updates'].append(dict(step=step,optimizer_state_steps=[step],support_rows=3072,query_rows=256,
            parameters_with_finite_grad=75670026,optimizer_state_tensor_bytes=p.update_probe.OPTIMIZER_BYTES,
            disjoint_ids=True,query_equal_v81=True,support_prefix_equal_v81=True,rng_state_equal_v81=True,
            updated_blocks={f'layers.{k}':True for k in range(12)}))
        path=out/f'barrier_{step}.json';p.write(path,proof_fixture(step))
        result['barriers'].append(dict(step=step,file=path.name,sha256=p.base.sha256(path),gc_collected=2))
    result['validations']=[dict(stage=stage,status='complete',forward_records=[dict(support_rows=n,label_rows=n,query_rows=q)],
        optimizer_steps=counter,finite_normalized_probabilities=True) for stage,counter in zip(stages,counters)]
    if kind=='small':
        probability=np.tile([.6,.4],(928,1));path=out/'probability_comparison.npz'
        np.savez(path,before=probability,after=probability.copy(),query_ids=np.arange(q))
        result.update(probability_max_difference=0.,threshold_equal=True,probability_comparison_sha256=p.base.sha256(path))
    p.write(out/'worker_result.json',result)
    return result


class Process:
    order=[]
    def __init__(self,target,args):
        self.kind,self.out,self.report,self.frozen,self.deadline=args;self.exitcode=0;self.pid=999;self.closed=False
    def start(self):
        self.order.append((self.kind,self.deadline));out=Path(self.out);report=Path(self.report)
        result=stage_fixture(out,self.kind)
        p.write(report/'worker_result.json',result)
    def is_alive(self):return False
    def join(self,timeout=None):pass
    def close(self):self.closed=True
    def kill(self):raise AssertionError('Completed worker unexpectedly killed')


class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4);self.deadline=p.base.DEADLINE;Process.order=[]
    def tearDown(self):p.base.DEADLINE=self.deadline

    def test_attribute_columns_exclude_real_targets_and_synthetic_split_is_disjoint(self):
        x=p.base.synthetic_frame(8693,7)
        def read_csv(path,**kw):
            self.assertEqual(kw['usecols'],p.ATTRIBUTES);self.assertNotIn('Transported',kw['usecols'])
            return pd.DataFrame({'PassengerId':np.arange(8693)})
        with patch.object(p.pd,'read_csv',side_effect=read_csv),patch.object(p.attributes,'compact_features',return_value=x):
            features,y,a,b=p.load_synthetic_target_split()
        np.testing.assert_array_equal(y,np.arange(8693)%2)
        self.assertEqual((len(a),len(b)),(6954,1739));self.assertEqual(len(np.intersect1d(a,b)),0)

    def test_cpu_actual_small_and_full_sequences_with_real_adam_and_guard(self):
        x=p.base.synthetic_frame(8693,1);y=np.arange(8693,dtype=np.int64)%2
        a,b=p.train_test_split(np.arange(8693),test_size=.2,stratify=y,random_state=p.SEED)
        real_tensor=torch.as_tensor;real_barrier=p.guard.lifecycle_barrier;real_capture=p.guard.capture_state
        def cpu_tensor(value,*args,**kwargs):kwargs['device']='cpu';return real_tensor(value,*args,**kwargs)
        def cpu_barrier(model,opt,data,targets,query,rng,ref,device,log_path):
            return real_barrier(model,opt,data,targets,query,rng,ref,'cpu',log_path)
        def cpu_capture(model,opt,data,targets,query,rng,ref,device='cpu'):
            return real_capture(model,opt,data,targets,query,rng,ref,'cpu')
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            stack.enter_context(patch.object(p,'check_frozen'))
            stack.enter_context(patch.object(p,'load_synthetic_target_split',return_value=(x,y,a,b)))
            stack.enter_context(patch.object(p.base.torch.backends.mps,'is_available',return_value=True))
            stack.enter_context(patch.object(p.base.torch.mps,'set_per_process_memory_fraction'))
            stack.enter_context(patch.object(p.base.torch.mps,'recommended_max_memory',return_value=10000))
            stack.enter_context(patch.object(p.base.torch.mps,'manual_seed'))
            stack.enter_context(patch.object(p.base.torch,'as_tensor',side_effect=cpu_tensor))
            stack.enter_context(patch.object(p.original,'fresh_model',side_effect=lambda device:(Tiny(),{})))
            stack.enter_context(patch.object(p.base,'release_memory',side_effect=gc.collect))
            stack.enter_context(patch.object(p.update_probe,'synchronized_snapshot',return_value={'mps_current_bytes':0}))
            stack.enter_context(patch.object(p.guard,'lifecycle_barrier',side_effect=cpu_barrier))
            stack.enter_context(patch.object(p.guard,'capture_state',side_effect=cpu_capture))
            for kind in ['small','full']:
                out,report=Path(folder)/kind,Path(folder)/(kind+'_report');out.mkdir();report.mkdir()
                p.run_stage(kind,out,report,{},time.monotonic()+120)
                result=p.read(out/'worker_result.json')
                self.assertEqual(result['status'],'worker_complete',result)
                self.assertTrue(result['release_complete'])
                self.assertEqual([r['optimizer_steps'] for r in result['validations']],[[],[1],[1]] if kind=='small' else [[],[1],[2]])
                self.assertEqual(len(result['barriers']),1 if kind=='small' else 2)
                for row in result['barriers']:
                    proof=p.read(out/row['file']);self.assertEqual(proof['before'],proof['after'])
                if kind=='small':self.assertEqual(result['probability_max_difference'],0.)

    def test_complete_proof_and_saved_probability_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);record=stage_fixture(out,'small');self.assertTrue(p.qualify_stage(record,out,'small'))
            bad=copy.deepcopy(record);bad['release_complete']=False
            with self.assertRaises(RuntimeError):p.qualify_stage(bad,out,'small')
            proof=p.read(out/'barrier_1.json');proof['after']={'changed':True};p.write(out/'barrier_1.json',proof)
            record['barriers'][0]['sha256']=p.base.sha256(out/'barrier_1.json')
            with self.assertRaises(RuntimeError):p.qualify_stage(record,out,'small')
            record=stage_fixture(out,'small')
            with np.load(out/'probability_comparison.npz') as saved:before=saved['before'].copy();after=saved['after'].copy()
            after[0]=[.4,.6];np.savez(out/'probability_comparison.npz',before=before,after=after,query_ids=np.arange(928))
            record['probability_comparison_sha256']=p.base.sha256(out/'probability_comparison.npz')
            with self.assertRaises(RuntimeError):p.qualify_stage(record,out,'small')

    def test_full_second_update_or_validation_missing_is_not_qualified(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);record=stage_fixture(out,'full');self.assertTrue(p.qualify_stage(record,out,'full'))
            for key in ['updates','validations','barriers']:
                bad=copy.deepcopy(record);bad[key].pop()
                with self.assertRaises(RuntimeError):p.qualify_stage(bad,out,'full')

    def mock_run(self,out,report):
        context=Mock();context.Process.side_effect=lambda target,args:Process(target,args)
        with patch.object(p,'OUTPUT',out),patch.object(p,'REPORT',report),patch.object(p.update_probe,'quota_guard'), \
             patch.object(p,'contract',return_value=({'sources':{}},'fixture')),patch.object(p,'check_frozen'), \
             patch.object(p.multiprocessing,'get_context',return_value=context):
            return p.run('PRIVATE_SESSION')

    def test_two_fresh_children_share_one_deadline_and_nooverwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            out,report=Path(folder)/'artifact',Path(folder)/'report'
            result=self.mock_run(out,report)
            self.assertTrue(result['diagnostic_qualification']);self.assertTrue(result['no_automatic_fullfit'])
            self.assertEqual([v[0] for v in Process.order],['small','full'])
            self.assertEqual(Process.order[0][1],Process.order[1][1])
            with self.assertRaises(FileExistsError):self.mock_run(out,report)

    def test_timeout_preserves_partial_and_prevents_full_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            out,report=Path(folder)/'artifact',Path(folder)/'report'
            with patch.object(p,'supervise',side_effect=TimeoutError('budget')):
                with self.assertRaises(TimeoutError):self.mock_run(out,report)
            self.assertEqual([v[0] for v in Process.order],['small'])
            self.assertTrue((out/'small/worker_result.json').exists())
            self.assertFalse(p.read(out/'result.json')['diagnostic_qualification'])
            self.assertTrue(p.read(out/'external_timeout.json')['partial_evidence_preserved'])

    def test_deadline_kills_only_owned_child(self):
        process=Mock();process.is_alive.side_effect=[True,True,False]
        with self.assertRaises(TimeoutError):p.supervise(process,time.monotonic()-.1)
        process.kill.assert_called_once_with();process.join.assert_called_once_with(timeout=5)


if __name__=='__main__':unittest.main()
