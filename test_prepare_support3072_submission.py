"""Synthetic completed-screen and portable-export mocks; never real targets/MPS."""
import copy
from contextlib import ExitStack
import gc
import hashlib
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
import pandas as pd
import torch

import prepare_support3072_submission as e


class FakePrep:
    columns = e.base.COMPACT
    n_features = 13
    def transform(self, x):return np.zeros((len(x),13),np.float32)


class FakeEngine(torch.nn.Module):
    def forward(self, support, labels, query):return torch.zeros((1,query.shape[1],10))


class Predictor(e.base.MitraPredictor):
    previous = None
    changed_reload = False
    bad_coverage = False
    def engine(self):
        if self._engine is None:self._engine=FakeEngine()
        return self._engine
    def release(self):self._engine=None
    def predict_proba(self, x):
        previous=type(self).previous() if type(self).previous else None
        if previous is not None and previous is not self:raise AssertionError('Original predictor retained on reload')
        engine=self.engine();chunk=1500 if type(self).bad_coverage else 2318
        support=torch.as_tensor(self.support).unsqueeze(0);labels=torch.as_tensor(self.labels).unsqueeze(0)
        for start in range(0,len(x),chunk):engine(support,labels,torch.zeros((1,min(chunk,len(x)-start),13)))
        p=np.full(len(x),.6 if previous is self or not type(self).changed_reload else .4)
        return np.column_stack([1-p,p])


class InlineProcess:
    def __init__(self,target,args):self.target,self.args=target,args;self.exitcode=0;self.pid=12345;self.closed=False
    def start(self):self.target(*self.args)
    def is_alive(self):return False
    def join(self,timeout=None):pass
    def kill(self):raise AssertionError('Completed inline worker should not be killed')
    def close(self):self.closed=True


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.stack=ExitStack()
        self.old_deadline=e.base.DEADLINE
        for obj in [e,e.data]:self.stack.enter_context(patch.object(obj,'ROOT',self.root))
        self.stack.enter_context(patch.object(e.probe,'require_committed'))
        self.stack.enter_context(patch.object(e.data.common,'require_committed'))
        self.stack.enter_context(patch.object(e.screen,'assert_frozen'))
        self.stack.enter_context(patch.object(e.screen,'check_log'))
        self.stack.enter_context(patch.object(e,'old_full_evidence',return_value={'selected_steps':15}))
        self.stack.enter_context(patch.object(e.base,'verify_checkpoint'))
        self.stack.enter_context(patch.object(e.base,'environment_lock',return_value='fixture==1\n'))
        self.stack.enter_context(patch.object(e.base,'release_memory',side_effect=gc.collect))
        self.stack.enter_context(patch.object(e.probe,'synchronized_snapshot',return_value={}))
        self.stack.enter_context(patch.object(e.subprocess,'check_output',return_value='fixturecommit\n'))
        self.env=hashlib.sha256(b'fixture==1\n').hexdigest()
        for name in set(e.OWN+e.screen.OWN+e.core.SOURCES):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# fixture\n')
        e.write(self.root/e.AUTH,dict(batch_id=e.BATCH,competition='spaceship-titanic',maximum_upload_attempts=10,
                weekly_reserve_percent=30,no_duplicate_or_blind_retry_upload=True))
        old=self.root/'artifacts'/e.OLD_FULL
        e.write(old/'fit_metadata.json',dict(status='fit_complete',selected_steps=15))
        self.source={name:e.sha(self.root/name) for name in e.core.SOURCES}
        self.frozen=dict(resource=dict(sources=self.source,environment_sha256=self.env,source_commit='historical'),
                         sources={name:e.sha(self.root/name) for name in e.screen.OWN},evidence={})
        self.stack.enter_context(patch.object(e.screen,'resource_contract',side_effect=lambda:(
            dict(resource=copy.deepcopy(self.frozen['resource']),sources={},evidence={}), 'fixture==1\n')))
        self.screen_fixture()
        self.stack.enter_context(patch.object(e.screen,'reference_inputs',return_value=(self.inputs,self.splits,self.ref,self.old_logs)))
        self.decision_fixture()
        self.passengers=pd.Series([f'{i:05d}_01' for i in range(4277)])
        path=self.root/'data/raw/test.csv';path.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'PassengerId':self.passengers}).to_csv(path,index=False)
        self.x=pd.DataFrame(np.zeros((8693,13)),columns=e.base.COMPACT)
        self.query=pd.DataFrame(np.zeros((4277,13)),columns=e.base.COMPACT)
        self.y=np.arange(8693,dtype=np.int64)%2
        Predictor.previous=None;Predictor.changed_reload=False;Predictor.bad_coverage=False

    def tearDown(self):
        e.base.DEADLINE=self.old_deadline;Predictor.previous=None;Predictor.changed_reload=False;Predictor.bad_coverage=False
        self.stack.close();self.tmp.cleanup()

    def screen_fixture(self):
        out=self.root/'artifacts'/e.RUN;report=self.root/'reports'/e.RUN
        out.mkdir(parents=True);report.mkdir(parents=True)
        dev=np.arange(6954,dtype=np.int64);y=dev%2;folds=np.repeat(np.arange(3),2318)
        self.splits=[(np.flatnonzero(folds!=k),np.flatnonzero(folds==k)) for k in range(3)]
        self.inputs=dict(dev=dev,y=y);self.ref=np.where(y,.7,.3);p=np.where(y,.8,.2)
        self.old_logs=[{'selected_steps':15} for _ in range(3)]
        score=e.data.metrics(p,y,self.splits);ref=e.data.metrics(self.ref,y,self.splits)
        protocol=dict(candidate=e.NAME,seed=e.SEED,folds=3,development_rows=6954,threshold=.5,settings=e.screen.SETTINGS,
            budget_seconds=18000,externally_supervised=True,plan_sha256=e.screen.PLAN_SHA,old_holdout_labels_read=False,
            automatic_export=False,automatic_upload=False,frozen=self.frozen,
            development_ids_sha256=e.base.ids_hash(dev),development_labels_sha256=e.base.ids_hash(y))
        records=[]
        for fold,(a,b) in enumerate(self.splits):
            name=f'{e.NAME}_seed{e.SEED+fold}_{e.base.ids_hash(dev[a])[:12]}.json'
            log=dict(selected_steps=2,source_hashes=self.source,environment_sha256=self.env)
            for folder in [out,report]:e.write(folder/name,log)
            single=[(np.array([],dtype=int),np.arange(len(b)))]
            record=dict(status='complete',fold=fold,seed=e.SEED+fold,training_log_sha256=e.sha(out/name),
                train_ids_sha256=e.base.ids_hash(dev[a]),validation_ids_sha256=e.base.ids_hash(dev[b]),selected_steps=2,
                paired_inner_steps=50,paired_common_refit_steps=2,metrics=e.data.metrics(p[b],y[b],single),
                reference=e.data.metrics(self.ref[b],y[b],single),paired=e.screen.paired(p[b],self.ref[b],y[b],single))
            records.append(record)
            for folder in [out,report]:e.write(folder/f'fold{fold}.json',record)
            np.savez(out/f'fold{fold}_rows.npz',train_ids=dev[a],validation_ids=dev[b])
        child=dict(status='worker_complete',completed_folds=3,release_complete=True,metrics=score,reference=ref,folds=records)
        e.write(out/'worker_result.json',child)
        filename=f'{e.NAME}_seed{e.SEED}.npz'
        np.savez(out/filename,prob=p,fold_accuracy=score['fold_accuracy'],dev_ids=dev,fold_id=folds)
        np.savez(out/'partial_oof.npz',prob=p,dev_ids=dev,fold_id=folds)
        assessment=dict(status='complete',candidate=e.NAME,completed_folds=3,metrics=score,reference=ref,
            continuation=e.screen.continuation(score,ref),paired=e.screen.paired(p,self.ref,y,self.splits),
            budget_seconds=18000,externally_supervised=True,old_holdout_labels_read=False,S_supported=False,
            automatic_export=False,automatic_upload=False)
        selection=dict(version=e.NAME,continuation_passed=True,screen_accuracy=score['accuracy'],automatic_eligibility=False)
        for folder in [out,report]:
            e.write(folder/'protocol.json',protocol);e.write(folder/'assessment.json',assessment);e.write(folder/'selection.json',selection)
            (folder/'requirements.lock').write_text('fixture==1\n')
            pd.DataFrame([dict(version=e.NAME,accuracy=score['accuracy'],log_loss=score['log_loss'])]).to_csv(folder/'screen_results.csv',index=False)
        summary={**assessment,'result_files_sha256':{name:e.sha(out/name) for name in [filename,'screen_results.csv','assessment.json','selection.json','worker_result.json']}}
        for folder in [out,report]:e.write(folder/'summary.json',summary);e.write(folder/'progress.json',summary)

    def decision_fixture(self):
        facts=e.collect_evidence()
        decision=dict(status='approved_conditional_exploratory_export',batch_id=e.BATCH,batch_authorization_sha256=e.sha(self.root/e.AUTH),
            version=e.NAME,module='mitra_support3072',run=e.RUN,proposal=e.PROPOSAL,validation_kind='random_development',
            exploratory=True,prior_A_eligible=None,prior_S_supported=False,threshold=.5,maximum_attempts=1,
            export_budget_seconds=7200,fit_seed=e.SEED,max_steps=50,rationale='Synthetic test authorization only',
            **{k:facts[k] for k in ['evidence_files','screen_accuracy','screen_log_loss','source_protocol_sha256']})
        e.write(self.root/e.DECISION,decision)

    def fake_fit(self,x,y,ids,seed,device,log_path,max_steps):
        self.assertEqual((len(x),len(y),seed,device,max_steps),(8693,8693,e.SEED,'mps',50))
        np.testing.assert_array_equal(ids,np.arange(8693))
        metadata=dict(status='fit_complete',selected_steps=2,source_hashes=self.source,environment_sha256=self.env)
        model=Predictor(FakePrep(),{'fake':True},{'weight':torch.ones(1)},np.zeros((8693,13),np.float32),y,'mps',metadata,2318)
        Predictor.previous=weakref.ref(model)
        e.write(log_path,metadata)
        return model

    def preparation_patches(self):
        stack=ExitStack()
        stack.enter_context(patch.object(e.probe,'quota_guard'))
        stack.enter_context(patch.object(e,'configure_runtime'))
        stack.enter_context(patch.object(e,'load_full_data',return_value=(self.x,self.y,self.query,self.passengers)))
        stack.enter_context(patch.object(e.core,'fit_support3072',self.fake_fit))
        context=Mock();context.Process.side_effect=lambda target,args:InlineProcess(target,args)
        stack.enter_context(patch.object(e.multiprocessing,'get_context',return_value=context))
        return stack

    def test_collect_complete_facts_never_create_approval(self):
        facts=e.collect_evidence()
        self.assertNotIn('status',facts);self.assertNotIn('approved',facts)
        self.assertTrue(any('fold2_rows.npz' in k for k in facts['evidence_files']))
        self.assertTrue(any('v93_Mitra' in k for k in facts['evidence_files']))
        self.assertEqual(facts['screen_accuracy'],1.)
        self.assertEqual(e.decision_evidence()['decision']['proposal'],e.PROPOSAL)

    def test_incomplete_failed_or_partial_fold_evidence_refuses_before_targets(self):
        out=self.root/'artifacts'/e.RUN
        original=e.read(out/'summary.json')
        for value in [{'status':'incomplete'},{'completed_folds':2},{'continuation':{'passed':False}}]:
            e.write(out/'summary.json',{**original,**value})
            with patch.object(e.screen,'reference_inputs') as labels:
                with self.assertRaises(RuntimeError):e.collect_evidence()
                labels.assert_not_called()
        e.write(out/'summary.json',original)
        np.savez(out/'fold2_rows.npz',train_ids=np.array([1]),validation_ids=np.array([2]))
        with self.assertRaises(RuntimeError):e.collect_evidence()

    def test_changed_oof_rejected(self):
        out=self.root/'artifacts'/e.RUN
        (out/f'{e.NAME}_seed{e.SEED}.npz').write_bytes(b'changed')
        with self.assertRaises(RuntimeError):e.collect_evidence()

    def test_missing_or_uncommitted_decision_precedes_full_labels_and_spawn(self):
        (self.root/e.DECISION).unlink()
        with patch.object(e,'load_full_data') as full,patch.object(e.multiprocessing,'get_context') as spawn:
            with self.assertRaises(FileNotFoundError):e.prepare('PRIVATE_SESSION')
            full.assert_not_called();spawn.assert_not_called()
        self.decision_fixture()
        with patch.object(e.probe,'require_committed',side_effect=RuntimeError('uncommitted')),patch.object(e,'load_full_data') as full:
            with self.assertRaises(RuntimeError):e.prepare('PRIVATE_SESSION')
            full.assert_not_called()

    def test_decision_scope_and_evidence_change_rejected(self):
        original=e.read(self.root/e.DECISION)
        for field,value in [('export_budget_seconds',8000),('fit_seed',e.SEED+1),('max_steps',100),('screen_accuracy',.9),('status','draft')]:
            e.write(self.root/e.DECISION,{**original,field:value})
            with self.assertRaises(RuntimeError):e.decision_evidence()
        e.write(self.root/e.DECISION,original)

    def test_full_mock_export_reload_and_submit_verify_compatibility(self):
        with self.preparation_patches():result=e.prepare('PRIVATE_SESSION')
        self.assertEqual(result['reload_rows'],4277);self.assertEqual(result['reload_max_absolute_difference'],0.)
        self.assertEqual(result['paired_common_refit_steps'],2)
        self.assertIsNone(Predictor.previous())
        out=self.root/'artifacts'/e.PROPOSAL
        record=e.read(out/'worker_result.json')
        self.assertEqual([r['query_rows'] for r in record['first_forward_records']],[2318,1959])
        self.assertEqual(record['first_forward_records'],record['reload_forward_records'])
        with patch.object(e.submission,'ROOT',self.root),patch.object(e.submission,'AUTH',self.root/e.AUTH), \
             patch.object(e.submission.subprocess,'check_output',return_value=''):
            bound,used=e.submission.verify(e.PROPOSAL)
        self.assertEqual(bound['version'],e.NAME);self.assertEqual(used,0)
        self.assertNotIn('PRIVATE_SESSION',(out/'preparation_protocol.json').read_text())
        with self.assertRaises(FileExistsError):e.prepare('PRIVATE_SESSION')

    def test_reload_mismatch_or_actual_partial_coverage_leaves_no_upload_files(self):
        Predictor.changed_reload=True
        with self.preparation_patches():
            with self.assertRaises(RuntimeError):e.prepare('PRIVATE_SESSION')
        out=self.root/'artifacts'/e.PROPOSAL
        self.assertIn('reload probability',e.read(out/'failure.json')['worker_error'])
        self.assertFalse((out/'submission.csv').exists());self.assertFalse((out/'authorization.json').exists())
        self.assertTrue((out/'training_log.json').exists())

    def test_actual_forward_coverage_guard(self):
        Predictor.bad_coverage=True
        with self.preparation_patches():
            with self.assertRaises(RuntimeError):e.prepare('PRIVATE_SESSION')
        out=self.root/'artifacts'/e.PROPOSAL
        self.assertIn('forward/chunk coverage',e.read(out/'failure.json')['worker_error'])
        self.assertFalse((out/'manifest.json').exists())

    def test_duplicate_csv_rejected_and_pending_removed(self):
        with self.preparation_patches(),patch.object(e,'duplicate_attempts',return_value=['existing attempt']):
            with self.assertRaises(RuntimeError):e.prepare('PRIVATE_SESSION')
        out=self.root/'artifacts'/e.PROPOSAL
        self.assertTrue((out/'model.joblib').exists());self.assertFalse((out/'pending').exists())
        for name in ['submission.csv','authorization.json','manifest.json','selection.json']:
            self.assertFalse((out/name).exists());self.assertFalse((self.root/'reports'/e.PROPOSAL/name).exists())

    def test_timeout_preserves_worker_records_and_withdraws_artifacts(self):
        with self.preparation_patches(),patch.object(e,'supervise',side_effect=TimeoutError('external timeout')):
            with self.assertRaises(TimeoutError):e.prepare('PRIVATE_SESSION')
        out=self.root/'artifacts'/e.PROPOSAL
        self.assertTrue((out/'worker_result.json').exists());self.assertTrue((out/'training_log.json').exists())
        self.assertTrue(e.read(out/'external_timeout.json')['partial_evidence_preserved'])
        self.assertFalse((out/'manifest.json').exists());self.assertFalse((out/'submission.csv').exists())

    def test_exhausted_batch_and_quota_refuse_before_worker(self):
        with patch.object(e.submission,'count_attempts',return_value=10),patch.object(e.probe,'quota_guard'),patch.object(e.multiprocessing,'get_context') as spawn:
            with self.assertRaises(RuntimeError):e.prepare('PRIVATE_SESSION')
            spawn.assert_not_called()
        with patch.object(e.probe,'quota_guard',side_effect=RuntimeError('quota stopped')),patch.object(e.multiprocessing,'get_context') as spawn:
            with self.assertRaises(RuntimeError):e.prepare('PRIVATE_SESSION')
            spawn.assert_not_called()

    def test_supervisor_kills_only_its_expired_worker(self):
        process=Mock()
        process.is_alive.side_effect=[True,True,False]
        with self.assertRaises(TimeoutError):e.supervise(process,time.monotonic()-.1)
        process.kill.assert_called_once_with()
        process.join.assert_called_once_with(timeout=5)

    def test_unauthorized_child_never_opens_targets_or_touches_mps(self):
        (self.root/e.DECISION).unlink()
        out=self.root/'artifacts/unauthorized';out.mkdir()
        with patch.object(e,'load_full_data') as labels,patch.object(e.base,'release_memory') as release, \
             patch.object(e.probe,'synchronized_snapshot') as snapshot,patch.object(e,'configure_runtime') as runtime:
            e.worker(out,{},time.monotonic()+60,'PRIVATE_SESSION')
        labels.assert_not_called();release.assert_not_called();snapshot.assert_not_called();runtime.assert_not_called()
        self.assertEqual(e.read(out/'worker_result.json')['status'],'failed_inconclusive')


if __name__=='__main__':unittest.main()
