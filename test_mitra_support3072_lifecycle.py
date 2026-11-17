"""Restricted AST diff and CPU numerical/runtime evidence; no real labels or GPU."""
import ast
import contextlib
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import mitra_support3072 as original
import mitra_support3072_lifecycle as repaired
from test_mitra_support3072 import Tiny,Prep

ROOT=Path(__file__).resolve().parent
ORIGINAL_SHA='3d7759b2e359db474ebe8f6b933da5c668e278fcd2c80c013123f3ce70a38501'


def dump(node):return ast.dump(node,include_attributes=False)


def restricted_ast_report():
    old_source=(ROOT/'mitra_support3072.py').read_text();new_source=(ROOT/'mitra_support3072_lifecycle.py').read_text()
    if hashlib.sha256(old_source.encode()).hexdigest()!=ORIGINAL_SHA:raise AssertionError('Frozen original source changed.')
    old,new=ast.parse(old_source),ast.parse(new_source)
    old.body.pop(0);new.body.pop(0)  # Module documentation only.
    added_sources=['mitra_support3072_lifecycle.py','mitra_lifecycle_guard.py',
        'reports/support3072_lifecycle_repair_design.json','reports/support3072_gc_diagnostic_plan.json']
    old_assign=next(n for n in old.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in n.targets))
    new_assign=next(n for n in new.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in n.targets))
    if ast.literal_eval(new_assign.value)!=ast.literal_eval(old_assign.value)+added_sources:raise AssertionError('Unapproved source-list difference.')
    new_assign.value=copy.deepcopy(old_assign.value)
    approved_import=ast.parse('import mitra_lifecycle_guard as lifecycle_guard').body[0]
    matches=[n for n in new.body if dump(n)==dump(approved_import)]
    if len(matches)!=1:raise AssertionError('Exactly the lifecycle guard import permitted.')
    new.body.remove(matches[0])
    provenance=[n for n in new.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RUNTIME_PROVENANCE' for t in n.targets)]
    if len(provenance)!=1:raise AssertionError('One provenance constant required.')
    if not isinstance(provenance[0].value,ast.Call) or dump(provenance[0].value.func)!=dump(ast.Name(id='dict',ctx=ast.Load())):
        raise AssertionError('Provenance must be literal metadata.')
    for keyword in provenance[0].value.keywords:ast.literal_eval(keyword.value)
    new.body.remove(provenance[0])
    helpers=[n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='lifecycle_barrier_record']
    if len(helpers)!=1:raise AssertionError('Only one logging helper permitted.')
    allowed_calls={'Path','Path(log_path).with_name','Path(log_path).stem','proof_path.exists','RuntimeError',
        'lifecycle_guard.lifecycle_barrier','proof.get','dict','result.update','base.sha256'}
    for node in ast.walk(helpers[0]):
        if isinstance(node,ast.Call) and ast.unparse(node.func) not in allowed_calls:
            raise AssertionError('Logging helper contains an unapproved call: '+ast.unparse(node.func))
    if sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='lifecycle_guard.lifecycle_barrier' for n in ast.walk(helpers[0]))!=1:
        raise AssertionError('One actual guard call per helper invocation required.')
    new.body.remove(helpers[0])
    fit=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='fit_support3072')
    metadata=next(n for n in fit.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='metadata' for t in n.targets))
    added=[k for k in metadata.value.keywords if k.arg=='runtime_provenance']
    expected=ast.parse('copy.deepcopy(RUNTIME_PROVENANCE)',mode='eval').body
    if len(added)!=1 or dump(added[0].value)!=dump(expected):raise AssertionError('Only runtime provenance logging may be added to metadata.')
    metadata.value.keywords.remove(added[0])
    insertion=ast.parse("if step:\n    row['lifecycle_barrier'] = lifecycle_barrier_record(model, optimizer, data, targets, valid_data, rng, reference_rng, device, log_path, step)").body[0]
    loops=[n for n in ast.walk(fit) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and ast.unparse(n.iter)=='range(51)']
    if len(loops)!=1:raise AssertionError('Original51-step inner loop required.')
    loop=loops[0];locations=[i for i,n in enumerate(loop.body) if dump(n)==dump(insertion)]
    if len(locations)!=1:raise AssertionError('Exactly one approved inner step>0 barrier insertion required.')
    index=locations[0]
    if ast.unparse(loop.body[index-1])!="row['stage'] = 'inner'" or 'before_validation' not in ast.unparse(loop.body[index+1]):
        raise AssertionError('Barrier must sit immediately after update/stage assignment and before validation checkpoint.')
    loop.body.pop(index)
    if dump(old)!=dump(new):raise AssertionError('Numerical program AST changed outside the approved barrier/provenance/logging changes.')
    return dict(status='passed',frozen_original_sha256=ORIGINAL_SHA,numerical_program_ast_identical_after_whitelist=True,
        allowed_changes=['module_docstring','guard_import','appended_source_provenance','literal_runtime_provenance',
            'one_logging_helper','one_metadata_provenance_keyword','one_inner_step_positive_barrier_assignment'],
        step0_or_refit_barrier_added=False,new_runtime_full_cv_empirically_reproduced=False)


def compare_nested(test,left,right):
    if torch.is_tensor(left):torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
    elif isinstance(left,dict):
        test.assertEqual(set(left),set(right))
        for key in left:compare_nested(test,left[key],right[key])
    elif isinstance(left,(list,tuple)):
        test.assertEqual(len(left),len(right))
        for a,b in zip(left,right):compare_nested(test,a,b)
    else:test.assertEqual(left,right)


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.previous_threads)
    def setUp(self):
        self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(repaired.base,'DEADLINE',None))
        self.stack.enter_context(patch.object(torch.mps,'synchronize',side_effect=AssertionError('CPU tests must not use MPS')))
        self.stack.enter_context(patch.object(torch.mps,'empty_cache',side_effect=AssertionError('CPU tests must not use MPS')))

    def test_restricted_ast_difference(self):
        report=restricted_ast_report();print(json.dumps(report),flush=True)
        self.assertEqual(repaired.SETTINGS,original.SETTINGS)
        self.assertFalse(repaired.RUNTIME_PROVENANCE['complete_three_fold_runtime_reproduction'])

    def trajectory(self,use_barrier):
        base=repaired.base;base.seed_everything(37,'cpu')
        model=base.make_model(dict(dim=8,dim_output=10,n_layers=2,n_heads=2,task='CLASSIFICATION'),device='cpu')
        self.assertEqual(type(model).__name__,'Tab2D')
        # Exercise original long-SDPA chunk checkpointing at a deliberately tiny boundary.
        base.install_attention_backend(model,sequence_threshold=8)
        optimizer=base.optimizer_for(model);data=torch.randn(48,3);labels=torch.arange(48)%2;query=torch.randn(11,3)
        rng,reference_rng=np.random.default_rng(19),np.random.default_rng(19);records=[]
        for step in [1,2]:
            p=rng.permutation(48);np.testing.assert_array_equal(p,reference_rng.permutation(48))
            support,targets=p[:32],p[32:40]
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=base.literal_learning_rate(step)
            model.train();logits=base.model_forward(model,data[support],labels[support],data[targets])
            self.assertEqual(tuple(logits.shape),(8,10));loss=F.cross_entropy(logits,labels[targets]);value=float(loss.detach())
            loss.backward();optimizer.step();del logits,loss;optimizer.zero_grad(set_to_none=True)
            if use_barrier:
                proof=repaired.lifecycle_barrier_record(model,optimizer,data,labels,query,rng,reference_rng,'cpu',None,step)
                self.assertTrue(proof['state_preserved'])
            probabilities=base.predict_arrays(model,data,labels,query,chunk=2318)
            records.append(dict(loss=value,state=copy.deepcopy(model.state_dict()),adam=copy.deepcopy(optimizer.state_dict()),
                rng=copy.deepcopy(repaired.lifecycle_guard.rng_state(rng,reference_rng,'cpu')),probabilities=probabilities.copy()))
        return records

    def test_small_real_tab2d_two_update_cpu_ab_exact(self):
        first=self.trajectory(False);second=self.trajectory(True);compare_nested(self,first,second)
        print(json.dumps(dict(cpu_tab2d_updates=2,native_classes=10,support_rows=32,query_rows=8,
            exact_loss_model_adam_rng_probabilities=True,real_labels_used=False,full_MPS_equivalence_not_claimed=True)),flush=True)

    def fit_mock(self,module,selected):
        Prep.fitted=[];x=pd.DataFrame(np.arange(12600).reshape(4200,3)/12600,index=np.arange(4200)*7+101)
        ids=x.index.to_numpy();y=np.arange(4200)%2;created=[];guard_steps=[];calls=[]
        def fresh(device):
            model=Tiny();created.append({k:v.detach().clone() for k,v in model.state_dict().items()});return model,{'tiny':True}
        actual_episode=module.episode_step
        def episode(*args,**kwargs):
            row=actual_episode(*args,**kwargs);calls.append((row['step'],row['query_ids_sha256'],row['support_prefix_ids_sha256']));return row
        original_guard=repaired.lifecycle_barrier_record
        def record(*args):
            guard_steps.append(args[-1]);return original_guard(*args)
        losses=[1.]*51 if selected==0 else [1.,.9]+[.8]*49
        with patch.object(module.base,'MitraPreprocessor',Prep),patch.object(module,'fresh_model',fresh),\
                patch.object(module.base,'model_forward',lambda model,support,labels,query:model(query)),\
                patch.object(module.base,'predict_arrays',side_effect=lambda model,data,labels,query,chunk:np.full((len(query),2),.5)),\
                patch.object(module,'log_loss',side_effect=losses),patch.object(module.base,'release_memory',side_effect=gc.collect),\
                patch.object(module,'episode_step',side_effect=episode),patch.object(repaired,'lifecycle_barrier_record',side_effect=record),\
                contextlib.redirect_stdout(io.StringIO()):
            artifact=module.fit_support3072(x,y,ids,9,device='cpu')
        self.assertEqual(len(created),2)
        compare_nested(self,created[0],created[1])
        return artifact,calls,guard_steps

    def test_mock51_selection_zero_nonzero_exact50_inner_barriers(self):
        for selected in [0,2]:
            with self.subTest(selected=selected):
                a,old_calls,_=self.fit_mock(original,selected);b,new_calls,barriers=self.fit_mock(repaired,selected)
                self.assertEqual(barriers,list(range(1,51)));self.assertEqual(old_calls,new_calls)
                self.assertEqual(b.metadata['selected_steps'],selected);self.assertEqual(b.metadata['refit_optimizer_steps'],selected)
                self.assertEqual(len(b.metadata['inner_history']),51);self.assertEqual(len(b.metadata['refit_history']),selected)
                self.assertNotIn('lifecycle_barrier',b.metadata['inner_history'][0])
                self.assertTrue(all('lifecycle_barrier' not in row for row in b.metadata['refit_history']))
                compare_nested(self,a.state,b.state);np.testing.assert_array_equal(a.support,b.support);np.testing.assert_array_equal(a.labels,b.labels)
                for left,right in zip(a.metadata['inner_history'],b.metadata['inner_history']):
                    for key in ['validation_log_loss','training_full_head_ce','support_ids_sha256','query_ids_sha256']:
                        self.assertEqual(left.get(key),right.get(key))

    def test_proof_file_basename_hash_and_failed_guard_stops(self):
        proof=dict(status='complete',state_preserved=True,barrier=['fixture'],gc_collected=0,observations=[],observation_caveat='fixture')
        with tempfile.TemporaryDirectory() as directory:
            training=Path(directory)/'training_log.json'
            def fake(*args,**kwargs):repaired.lifecycle_guard.write(kwargs['log_path'],proof);return proof
            with patch.object(repaired.lifecycle_guard,'lifecycle_barrier',side_effect=fake):
                record=repaired.lifecycle_barrier_record(None,None,None,None,None,None,None,'cpu',training,1)
                self.assertEqual(record['proof_file'],'training_log.lifecycle_step01.json')
                self.assertEqual(record['proof_sha256'],repaired.base.sha256(Path(directory)/record['proof_file']))
                self.assertNotIn('proof',record)
                with self.assertRaises(RuntimeError):repaired.lifecycle_barrier_record(None,None,None,None,None,None,None,'cpu',training,1)
        with patch.object(repaired.lifecycle_guard,'lifecycle_barrier',return_value=dict(proof,status='incomplete',state_preserved=False)):
            with self.assertRaises(RuntimeError):repaired.lifecycle_barrier_record(None,None,None,None,None,None,None,'cpu',None,1)

    def test_failed_barrier_prevents_next_validation_and_refit(self):
        x=pd.DataFrame(np.arange(12600).reshape(4200,3)/12600);y=np.arange(4200)%2;ids=np.arange(4200)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'training_log.json'
            with patch.object(repaired.base,'MitraPreprocessor',Prep),patch.object(repaired,'fresh_model',side_effect=lambda device:(Tiny(),{})) as fresh,\
                    patch.object(repaired.base,'model_forward',lambda model,support,labels,query:model(query)),\
                    patch.object(repaired.base,'predict_arrays',side_effect=lambda model,data,labels,query,chunk:np.full((len(query),2),.5)) as predict,\
                    patch.object(repaired.base,'release_memory',side_effect=gc.collect),\
                    patch.object(repaired,'lifecycle_barrier_record',side_effect=RuntimeError('state changed')),\
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'state changed'):repaired.fit_support3072(x,y,ids,9,device='cpu',log_path=path)
                self.assertEqual(predict.call_count,1);self.assertEqual(fresh.call_count,1)
            failed=json.loads(path.read_text());self.assertEqual(failed['status'],'failed_inconclusive')
            self.assertEqual([r['step'] for r in failed['inner_history']],[0]);self.assertEqual(failed['refit_history'],[])


if __name__=='__main__':unittest.main()
