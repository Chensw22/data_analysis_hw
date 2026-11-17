"""Synthetic lifecycle qualification and portable export tests; no GPU/real data."""
import copy
from contextlib import ExitStack
import gc
import hashlib
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
import pandas as pd
import torch

import prepare_support3072_lifecycle_submission as e
from test_prepare_support3072_submission import Predictor, FakePrep, InlineProcess
from test_mitra_support3072_gc_probe import stage_fixture


def tensor(shape, identity, dtype='torch.float32', device='mps:0', grad=False):
    return dict(object_id=identity,storage_id=identity+100,storage_data_ptr=identity+200,
        storage_bytes=int(np.prod(shape))*4,storage_offset=0,shape=shape,stride=[1]*len(shape),
        dtype=dtype,device=device,requires_grad=grad,bytes_sha256='a'*64)


def proof(step):
    state=dict(model_object_id=1,optimizer_object_id=2,optimizer_type='AdamW',model_training=True,
        module_training={'':True},parameters={'weight':tensor([1],3,grad=True)},buffers={},
        all_gradients_none=True,all_input_gradients_none=True,optimizer_steps={'weight':float(step)},
        optimizer_state_object_id=4,optimizer_groups_object_id=5,
        optimizer_groups=[dict(object_id=6,parameter_ids=[3],options=dict(lr=3e-6,weight_decay=.3))],
        optimizer_state={'weight':dict(parameter_object_id=3,state_object_id=7,values={
            'step':{'tensor':tensor([],8,device='cpu')},'exp_avg':{'tensor':tensor([1],9)},
            'exp_avg_sq':{'tensor':tensor([1],10)}})},
        inputs=dict(data=tensor([6954,13],11),targets=tensor([6954],12,dtype='torch.int64'),query=tensor([1739,13],13)),
        rng=dict(python_complete_state_sha256='b'*64,numpy_global_complete_state_sha256='c'*64,
            sampler={'state':step},counterfactual_sampler={'state':step},
            torch_cpu=dict(bytes_sha256='d'*64),torch_mps=dict(bytes_sha256='e'*64)),grad_enabled=True)
    return dict(status='complete',state_preserved=True,device='mps',barrier=e.core.lifecycle_guard.BARRIER,
        gc_collected=0,before=state,after=copy.deepcopy(state),observation_caveat='Synthetic fixture',
        observations=[dict(stage=s) for s in ['before_pre_hash','after_pre_hash','barrier_after_synchronize1',
        'barrier_after_gc_collect','barrier_after_empty_cache','barrier_after_synchronize2','before_post_hash','after_post_hash']])


class LifecycleExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.stack=ExitStack()
        self.deadline=e.base.DEADLINE
        self.stack.enter_context(patch.object(e,'ROOT',self.root))
        self.stack.enter_context(patch.object(e.diagnostic,'OUTPUT',self.root/'artifacts/probes/gc'))
        self.stack.enter_context(patch.object(e.diagnostic,'REPORT',self.root/'reports/gc'))
        self.stack.enter_context(patch.object(e.probe,'require_committed'))
        self.stack.enter_context(patch.object(e.diagnostic,'check_frozen'))
        self.stack.enter_context(patch.object(e.base,'verify_checkpoint'))
        self.stack.enter_context(patch.object(e.base,'environment_lock',return_value='fixture==1\n'))
        self.stack.enter_context(patch.object(e.base,'release_memory',side_effect=gc.collect))
        self.stack.enter_context(patch.object(e.probe,'synchronized_snapshot',return_value={}))
        self.stack.enter_context(patch.object(e.probe,'quota_guard'))
        self.stack.enter_context(patch.object(e.subprocess,'check_output',return_value='fixturecommit\n'))
        self.stack.enter_context(patch.object(e.screen,'check_log'))
        self.env=hashlib.sha256(b'fixture==1\n').hexdigest()
        runtime_sources=['mitra_lifecycle_guard.py','test_mitra_lifecycle_guard.py','mitra_support3072_gc_probe.py',
                         'test_mitra_support3072_gc_probe.py','mitra_support3072_lifecycle.py','test_mitra_support3072_lifecycle.py']
        for name in set(e.OWN+runtime_sources):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# fixture\n')
        e.write(self.root/e.DESIGN,dict(new_fullfit_budget_seconds=7200,maximum_new_fullfit_runs=1,
                no_overwrite_resume_or_checkpoint_reuse=True,no_automatic_fullfit_or_upload=True,evidence={}))
        e.write(self.root/e.FAILURE_REVIEW,dict(evidence_sha256={}))
        e.write(self.root/e.STATIC,dict(status='passed_cpu_and_static_after_device_dispatch_fix',tests_run=17,
            cpu_two_update_tab2d_exact=True,mps_index_zero_dispatch_regression_mock=True,full_runtime_CV_reproduced=False,
            ast=dict(numerical_program_ast_identical_after_whitelist=True,step0_or_refit_barrier_added=False),
            source_sha256={name:e.sha(self.root/name) for name in runtime_sources}))
        e.write(self.root/'artifacts/proposal_nov16_support3072/failure.json',dict(status='failed_incomplete_not_upload_eligible'))
        e.write(self.root/'reports/proposal_nov16_support3072/failure.json',dict(status='failed_incomplete_not_upload_eligible'))
        self.frozen=dict(sources={name:e.sha(self.root/name) for name in runtime_sources},
            evidence_files={},environment_sha256=self.env,source_commit='historical')
        self.stack.enter_context(patch.object(e.diagnostic,'contract',side_effect=lambda:(copy.deepcopy(self.frozen),'fixture==1\n')))
        self.resource_fixture()
        self.previous_core=e.previous.core;self.previous_sources=list(e.previous.core.SOURCES)
        self.previous_facts=dict(evidence_files={},sources={},screen_accuracy=.8211101524302559,
            screen_log_loss=.371853,source_protocol_sha256='f'*64,environment_sha256=self.env)
        self.old_collect=self.stack.enter_context(patch.object(e.previous,'collect_evidence',side_effect=lambda:copy.deepcopy(self.previous_facts)))
        e.write(self.root/e.AUTH,dict(batch_id=e.BATCH,competition='spaceship-titanic',maximum_upload_attempts=10,
            weekly_reserve_percent=30,no_duplicate_or_blind_retry_upload=True))
        self.decision_fixture()
        e.write(self.root/'artifacts'/e.OLD_FULL/'fit_metadata.json',dict(selected_steps=15))
        self.passengers=pd.Series([f'{i:05d}_01' for i in range(4277)])
        path=self.root/'data/raw/test.csv';path.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'PassengerId':self.passengers}).to_csv(path,index=False)
        self.x=pd.DataFrame(np.zeros((8693,13)),columns=e.base.COMPACT)
        self.query=pd.DataFrame(np.zeros((4277,13)),columns=e.base.COMPACT);self.y=np.arange(8693,dtype=np.int64)%2
        Predictor.previous=None;Predictor.changed_reload=False;Predictor.bad_coverage=False

    def tearDown(self):
        e.base.DEADLINE=self.deadline;Predictor.previous=None;Predictor.changed_reload=False;Predictor.bad_coverage=False
        self.stack.close();self.tmp.cleanup()

    def resource_fixture(self):
        out,report=e.diagnostic.OUTPUT,e.diagnostic.REPORT
        rows=[];audits=[]
        for kind in ['small','full']:
            directory=out/kind;directory.mkdir(parents=True)
            record=stage_fixture(directory,kind)
            # Full resource template has the same explicit tensor schema used by exporter tests.
            if kind=='full':
                for step in [1,2]:
                    e.write(directory/f'barrier_{step}.json',proof(step))
                    record['barriers'][step-1]['sha256']=e.sha(directory/f'barrier_{step}.json')
                    record['barriers'][step-1]['gc_collected']=0
                e.write(directory/'worker_result.json',record)
            shutil.copytree(directory,report/kind)
            rows.append(dict(stage=kind,worker_result_sha256=e.sha(directory/'worker_result.json'),release_complete=True))
            for step in ([1] if kind=='small' else [1,2]):
                audits.append(dict(stage=kind,step=step,proof_sha256=e.sha(directory/f'barrier_{step}.json')))
        result=dict(status='complete',diagnostic_qualification=True,budget_seconds=1200,externally_supervised=True,
            no_real_targets=True,no_accuracy_computed=True,no_automatic_fullfit=True,seconds=176,stages=rows)
        for folder in [out,report]:
            e.write(folder/'freeze.json',self.frozen);e.write(folder/'result.json',result)
            (folder/'requirements.lock').write_text('fixture==1\n')
        e.write(report/'root_audit.json',dict(status='root_verified_complete_bounded_diagnostic',
            small_saved_probabilities_bitwise_equal=True,no_real_accuracy=True,full_optimizer_steps=[0,1,2],state_proofs=audits))
        for name in self.frozen['sources']:
            path=out/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(self.root/name,path)

    def decision_fixture(self):
        facts=e.collect_evidence()
        decision=dict(status='approved_conditional_exploratory_export',batch_id=e.BATCH,batch_authorization_sha256=e.sha(self.root/e.AUTH),
            version=e.NAME,module='mitra_support3072_lifecycle',run=e.RUN,proposal=e.PROPOSAL,validation_kind='random_development',
            exploratory=True,prior_A_eligible=None,prior_S_supported=False,threshold=.5,maximum_attempts=1,
            export_budget_seconds=7200,fit_seed=e.SEED,max_steps=50,maximum_new_fullfit_runs=1,
            rationale='Synthetic test only',**e.INHERITANCE,**facts)
        e.write(self.root/e.DECISION,decision)

    def training_fixture(self,out):
        sources={name:e.sha(self.root/name) for name in e.core.SOURCES}
        metadata=dict(status='fit_complete',selected_steps=2,source_hashes=sources,environment_sha256=self.env,
            runtime_provenance=e.core.RUNTIME_PROVENANCE,inner_history=[dict(step=0,stage='inner')],
            refit_history=[dict(step=i,stage='refit') for i in [1,2]])
        for step in range(1,51):
            value=proof(step);path=out/f'training_log.lifecycle_step{step:02d}.json';e.write(path,value)
            item={k:value[k] for k in ['status','state_preserved','barrier','gc_collected','observations','observation_caveat']}
            item.update(step=step,proof_persisted=True,proof_file=path.name,proof_sha256=e.sha(path))
            metadata['inner_history'].append(dict(step=step,stage='inner',learning_rate=3e-6,lifecycle_barrier=item))
        return metadata

    def fake_fit(self,x,y,ids,seed,device,log_path,max_steps):
        self.assertEqual((len(x),len(y),seed,device,max_steps),(8693,8693,e.SEED,'mps',50))
        np.testing.assert_array_equal(ids,np.arange(8693))
        metadata=self.training_fixture(log_path.parent);e.write(log_path,metadata)
        model=Predictor(FakePrep(),{'fake':True},{'weight':torch.ones(1)},np.zeros((8693,13),np.float32),y,'mps',metadata,2318)
        Predictor.previous=weakref.ref(model);return model

    def preparation_patches(self):
        stack=ExitStack();stack.enter_context(patch.object(e,'configure_runtime'))
        stack.enter_context(patch.object(e,'load_full_data',return_value=(self.x,self.y,self.query,self.passengers)))
        stack.enter_context(patch.object(e.core,'fit_support3072',self.fake_fit))
        context=Mock();context.Process.side_effect=lambda target,args:InlineProcess(target,args)
        stack.enter_context(patch.object(e.multiprocessing,'get_context',return_value=context));return stack

    def test_inherited_original_cv_module_unchanged_and_no_approval_created(self):
        facts=e.collect_evidence()
        self.assertIs(e.previous.core,self.previous_core);self.assertEqual(e.previous.core.SOURCES,self.previous_sources)
        self.assertNotIn('status',facts);self.assertNotIn('approved',facts)
        self.assertFalse(facts['runtime_evidence']['runtime_three_fold_reproduced'])
        self.assertTrue(any(k.endswith('probability_comparison.npz') for k in facts['evidence_files']))
        self.assertFalse(any(k.startswith('reports/') and k.endswith('.npz') for k in facts['evidence_files']))
        with patch.object(e.previous,'collect_evidence',side_effect=RuntimeError('incomplete original CV')):
            with self.assertRaises(RuntimeError):e.collect_evidence()

    def test_failed_or_changed_diagnostic_static_and_npz_refused(self):
        out=e.diagnostic.OUTPUT;original=e.read(out/'result.json')
        for patch_value in [dict(status='failed_inconclusive'),dict(seconds=1201),dict(diagnostic_qualification=False)]:
            e.write(out/'result.json',{**original,**patch_value})
            e.write(e.diagnostic.REPORT/'result.json',{**original,**patch_value})
            with self.assertRaises(RuntimeError):e.collect_evidence()
        for folder in [out,e.diagnostic.REPORT]:e.write(folder/'result.json',original)
        path=out/'small/probability_comparison.npz';path.write_bytes(b'changed')
        with self.assertRaises(RuntimeError):e.collect_evidence()

    def test_decision_must_be_committed_matching_before_full_labels(self):
        (self.root/e.DECISION).unlink()
        with patch.object(e,'load_full_data') as labels,patch.object(e.multiprocessing,'get_context') as spawn:
            with self.assertRaises(FileNotFoundError):e.prepare('PRIVATE_SESSION')
            labels.assert_not_called();spawn.assert_not_called()
        self.decision_fixture();original=e.read(self.root/e.DECISION)
        for key,value in [('runtime_three_fold_reproduced',True),('module','mitra_support3072'),('export_budget_seconds',8000),
                          ('maximum_new_fullfit_runs',2),('status','draft'),('sources',{})]:
            e.write(self.root/e.DECISION,{**original,key:value})
            with self.assertRaises(RuntimeError):e.decision_evidence()
        e.write(self.root/e.DECISION,original)
        with patch.object(e.probe,'require_committed',side_effect=RuntimeError('uncommitted')),patch.object(e,'load_full_data') as labels:
            with self.assertRaises(RuntimeError):e.prepare('PRIVATE_SESSION')
            labels.assert_not_called()

    def test_all50_proofs_schema_state_hash_and_phase_boundaries(self):
        out=self.root/'synthetic_training';out.mkdir();log=self.training_fixture(out);evidence=e.decision_evidence()
        self.assertEqual(e.lifecycle_proofs(log,out,evidence)['verified_inner_barriers'],50)
        bad=copy.deepcopy(log);bad['inner_history'][0]['lifecycle_barrier']={}
        with self.assertRaises(RuntimeError):e.lifecycle_proofs(bad,out,evidence)
        bad=copy.deepcopy(log);bad['refit_history'][0]['lifecycle_barrier']={}
        with self.assertRaises(RuntimeError):e.lifecycle_proofs(bad,out,evidence)
        path=out/'training_log.lifecycle_step50.json';original=e.read(path)
        for mutate in [lambda p:p['after'].update(model_object_id=999),
                       lambda p:p['before'].pop('rng'),
                       lambda p:p['before']['optimizer_steps'].update(weight=49.),
                       lambda p:p['before']['optimizer_state']['weight']['values'].pop('exp_avg_sq')]:
            value=copy.deepcopy(original);mutate(value)
            if value['after']==original['after']:value['after']=copy.deepcopy(value['before'])
            e.write(path,value);bad=copy.deepcopy(log);bad['inner_history'][50]['lifecycle_barrier']['proof_sha256']=e.sha(path)
            with self.assertRaises((RuntimeError,KeyError)):e.lifecycle_proofs(bad,out,evidence)
        path.unlink()
        with self.assertRaises(RuntimeError):e.lifecycle_proofs(log,out,evidence)

    def test_complete_mock_export_and_submit_verify(self):
        with self.preparation_patches():result=e.prepare('PRIVATE_SESSION')
        out=self.root/'artifacts'/e.PROPOSAL
        self.assertEqual(result['reload_rows'],4277);self.assertEqual(result['reload_max_absolute_difference'],0.)
        self.assertEqual(result['lifecycle_proofs']['verified_inner_barriers'],50)
        self.assertFalse(result['runtime_three_fold_reproduced']);self.assertIsNone(Predictor.previous())
        record=e.read(out/'worker_result.json')
        self.assertEqual([r['query_rows'] for r in record['reload_forward_records']],[2318,1959])
        self.assertEqual(len(list(out.glob('training_log.lifecycle_step*.json'))),50)
        self.assertFalse(list((self.root/'reports'/e.PROPOSAL).glob('training_log.lifecycle_step*.json')))
        self.assertNotIn('PRIVATE_SESSION',(out/'preparation_protocol.json').read_text())
        with patch.object(e.submission,'ROOT',self.root),patch.object(e.submission,'AUTH',self.root/e.AUTH), \
             patch.object(e.submission.subprocess,'check_output',return_value=''):
            bound,_=e.submission.verify(e.PROPOSAL)
        self.assertEqual(bound['version'],e.NAME)
        with self.assertRaises(FileExistsError):e.prepare('PRIVATE_SESSION')

    def test_reload_failure_preserves_proofs_first_error_and_no_upload_files(self):
        Predictor.changed_reload=True
        with self.preparation_patches():
            with self.assertRaises(RuntimeError):e.prepare('PRIVATE_SESSION')
        out=self.root/'artifacts'/e.PROPOSAL
        self.assertIn('reload probability',e.read(out/'failure.json')['worker_error'])
        self.assertEqual(len(list(out.glob('training_log.lifecycle_step*.json'))),50)
        for name in ['submission.csv','authorization.json','manifest.json']:self.assertFalse((out/name).exists())

    def test_bad_forward_or_duplicate_refuses_publication(self):
        with self.preparation_patches(),patch.object(e,'duplicate_attempts',return_value=['old attempt']):
            with self.assertRaises(RuntimeError):e.prepare('PRIVATE_SESSION')
        out=self.root/'artifacts'/e.PROPOSAL
        self.assertFalse((out/'pending').exists());self.assertFalse((out/'submission.csv').exists())
        self.assertTrue((out/'model.joblib').exists())

    def test_external_timeout_retains_partial_no_manifest(self):
        with self.preparation_patches(),patch.object(e,'supervise',side_effect=TimeoutError('external timeout')):
            with self.assertRaises(TimeoutError):e.prepare('PRIVATE_SESSION')
        out=self.root/'artifacts'/e.PROPOSAL
        self.assertTrue((out/'external_timeout.json').exists());self.assertTrue((out/'training_log.json').exists())
        self.assertFalse((out/'manifest.json').exists())
        process=Mock();process.is_alive.side_effect=[True,True,False]
        with self.assertRaises(TimeoutError):e.supervise(process,time.monotonic()-.1)
        process.kill.assert_called_once_with()

    def test_quota_batch_and_unauthorized_worker_do_not_fit(self):
        for target, method, kw in [(e.probe,'quota_guard',dict(side_effect=RuntimeError('reserve'))),
                                  (e.submission,'count_attempts',dict(return_value=10))]:
            with patch.object(target,method,**kw),patch.object(e.multiprocessing,'get_context') as spawn:
                with self.assertRaises(RuntimeError):e.prepare('PRIVATE_SESSION')
                spawn.assert_not_called()
        (self.root/e.DECISION).unlink();out=self.root/'unauthorized';out.mkdir()
        with patch.object(e,'load_full_data') as labels,patch.object(e,'configure_runtime') as runtime:
            e.worker(out,{},time.monotonic()+60,'PRIVATE_SESSION')
        labels.assert_not_called();runtime.assert_not_called()


if __name__=='__main__':unittest.main()
