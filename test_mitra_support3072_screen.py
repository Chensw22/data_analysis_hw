"""CPU synthetic/cache contract tests; no model loading or candidate score."""
import copy
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import mitra_support3072_screen as m


def rows(ids,seed,count,new):
    rng=np.random.default_rng(seed);result=[]
    for step in range(1,count+1):
        p=rng.permutation(len(ids));s=np.r_[p[:1024],p[1280:3328]];q=p[1024:1280]
        row=dict(step=step,support_rows=3072 if new else 1024,query_rows=256,finite_gradients=True,disjoint_ids=True,
            cumulative_optimizer_steps=step,learning_rate=m.base.literal_learning_rate(step),
            support_ids_sha256=m.base.ids_hash(ids[s] if new else ids[p[:1024]]),query_ids_sha256=m.base.ids_hash(ids[q]))
        if new:row.update(support_prefix_ids_sha256=m.base.ids_hash(ids[p[:1024]]),local_support_ids_sha256=m.base.ids_hash(s),
            local_query_ids_sha256=m.base.ids_hash(q),query_equal_v81=True,support_prefix_equal_v81=True,rng_state_equal_v81=True)
        result.append(row)
    return result


def log_fixture(new):
    ids=np.arange(4636,dtype=np.int64)+1000;query=np.arange(7,dtype=np.int64);y=np.arange(len(ids))%2;seed=20261112
    inner,valid=m.train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
    history=[dict(step=0),*rows(ids[inner],seed,50,new)]
    for row in history:row.update(stage='inner',validation_log_loss=.3+abs(row['step']-2)*.001,
        validation_support_rows=len(inner),validation_query_rows=len(valid),validation_full_support=True,
        validation_optimizer_steps=[] if row['step']==0 else [row['step']],
        validation_optimizer_state_tensor_bytes=0 if row['step']==0 else m.probe.OPTIMIZER_BYTES)
    refit=rows(ids,seed,2,new)
    for row in refit:row['stage']='refit'
    record=dict(status='fit_complete',seed=seed,training_rows=len(ids),prediction_rows=len(query),
        training_ids_sha256=m.base.ids_hash(ids),prediction_ids_sha256=m.base.ids_hash(query),
        full_prediction_support=True,fresh_refit_from_original_checkpoint=True,
        inner_train_ids_sha256=m.base.ids_hash(ids[inner]),inner_validation_ids_sha256=m.base.ids_hash(ids[valid]),
        inner_training_rows=len(inner),inner_validation_rows=len(valid),inner_history=history,refit_history=refit,
        selected_steps=2,inner_optimizer_steps=50,refit_optimizer_steps=2,recipe=m.SETTINGS if new else m.old_core.SETTINGS,
        checkpoint_sha256=m.base.WEIGHT_SHA,query_chunk=2318,parameters=75670026)
    return record,ids,query,y,seed


class ScreenTests(unittest.TestCase):
    def test_all_inner_and_common_refit_pairing(self):
        old,*_=log_fixture(False);new,ids,query,y,seed=log_fixture(True)
        m.check_log(new,old,ids,query,y,seed,True)
        for field in ['query_ids_sha256','support_prefix_ids_sha256','local_support_ids_sha256']:
            bad=copy.deepcopy(new);bad['inner_history'][50][field]='changed'
            with self.subTest(field=field),self.assertRaises(RuntimeError):m.check_log(bad,old,ids,query,y,seed,True)
        bad=copy.deepcopy(new);bad['refit_history'][1]['query_ids_sha256']='changed'
        with self.assertRaises(RuntimeError):m.check_log(bad,old,ids,query,y,seed,True)
        bad=copy.deepcopy(new);bad['inner_history'].pop()
        with self.assertRaises(RuntimeError):m.check_log(bad,old,ids,query,y,seed,True)

    def test_gate_needs_strict_improvement_and_two_folds(self):
        ref=dict(accuracy=.8,log_loss=.4,fold_accuracy=[.8]*3)
        self.assertFalse(m.continuation(ref,ref)['passed'])
        self.assertTrue(m.continuation(dict(ref,log_loss=.39),ref)['passed'])
        self.assertFalse(m.continuation(dict(ref,accuracy=.79,log_loss=.39),ref)['passed'])
        self.assertFalse(m.continuation(dict(ref,log_loss=.39,fold_accuracy=[.81,.79,.79]),ref)['passed'])

    def test_unqualified_resource_before_targets(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(m,'ROOT',Path(directory)),\
                patch.object(m.probe,'quota_guard'),patch.object(m,'resource_contract',side_effect=RuntimeError('not qualified')),\
                patch.object(m,'reference_inputs') as targets:
            with self.assertRaisesRegex(RuntimeError,'not qualified'):m.run(Path('/not/read'))
            targets.assert_not_called()

    def test_external_supervisor_kills_own_worker_at_deadline(self):
        class Process:
            def __init__(self):self.alive=True;self.killed=False;self.exitcode=None
            def is_alive(self):return self.alive
            def kill(self):self.killed=True;self.alive=False;self.exitcode=-9
            def join(self,timeout):pass
        process=Process()
        with self.assertRaises(TimeoutError):m.supervise(process,time.monotonic()-1)
        self.assertTrue(process.killed);self.assertFalse(process.is_alive())
        process.exitcode=1
        with self.assertRaises(RuntimeError):m.supervise(process,time.monotonic()+1)

    def exercise_folds(self,fail=False):
        dev=np.arange(6);y=dev%2;splits=[(dev[dev//2!=k],dev[dev//2==k]) for k in range(3)]
        inputs=dict(dev=dev,y=y,features=pd.DataFrame({'feature':dev}));reference=.2+.6*y;made=[];seeds=[]
        class Fake:
            def __init__(self,x,labels,ids,seed,**kwargs):
                if made and not made[-1].released:raise AssertionError('Previous engine resident.')
                self.released=False;self.support=np.zeros((len(ids),13));self.labels=labels;self.query_chunk=2318
                self.metadata={'selected_steps':2,'source_hashes':{},'environment_sha256':'fixture'};made.append(self);seeds.append(seed)
                if kwargs['max_steps']!=50 or kwargs['device']!='mps':raise AssertionError('Changed recipe.')
            def predict_proba(self,q):
                if fail and len(made)==2:raise RuntimeError('injected prediction failure')
                p=.1+.8*(q.feature.to_numpy()%2);return np.column_stack([1-p,p])
            def release(self):self.released=True
        core=types.SimpleNamespace(SETTINGS=m.SETTINGS,fit_support3072=Fake,SOURCES=[])
        original_import=m.importlib.import_module
        frozen={'resource':{'sources':{},'environment_sha256':'fixture'},'sources':{}}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'artifact';report=Path(directory)/'report';out.mkdir();report.mkdir()
            with patch.object(m.importlib,'import_module',side_effect=lambda name,*a,**k:core if name=='mitra_support3072' else original_import(name,*a,**k)),patch.object(m,'assert_frozen'),\
                    patch.object(m,'check_log') as checked,patch.object(m.probe,'quota_guard') as quota,\
                    patch.object(m.base,'release_memory'),patch.object(m.probe,'synchronized_snapshot',return_value={}):
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'injected'):m.train_folds(out,report,inputs,splits,reference,[{'selected_steps':3}]*3,frozen,time.monotonic()+60,Path('/private'))
                    self.assertFalse((out/'screen_results.csv').exists())
                    with np.load(out/'partial_oof.npz') as cache:self.assertEqual(np.isfinite(cache['prob']).sum(),2)
                else:
                    result=m.train_folds(out,report,inputs,splits,reference,[{'selected_steps':3}]*3,frozen,time.monotonic()+60,Path('/private'))
                    self.assertEqual(result['status'],'worker_complete');self.assertEqual(checked.call_count,3)
                    self.assertEqual(seeds,[20261112,20261113,20261114]);self.assertEqual(quota.call_count,3)
                    self.assertFalse((out/'screen_results.csv').exists())
                self.assertTrue(all(item.released for item in made))

    def test_three_folds_seeds_quota_and_sequential_release(self):self.exercise_folds()
    def test_partial_failure_keeps_partial_without_csv(self):self.exercise_folds(True)

    def test_publish_refuses_incomplete_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with self.assertRaises(RuntimeError):m.publish(out,out,{},[],np.array([]),{'status':'worker_complete','completed_folds':2},{},time.monotonic()+5)
            self.assertFalse((out/'screen_results.csv').exists())

    def parent_fixture(self,failure=None):
        dev=np.arange(6);y=dev%2;splits=[(dev[dev//2!=k],dev[dev//2==k]) for k in range(3)]
        inputs=dict(dev=dev,y=y);reference=.2+.6*y;prob=.1+.8*y
        frozen=dict(resource={'sources':{}},sources={},evidence={})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'artifacts'/m.RUN
            class Process:
                exitcode=0
                def start(self):
                    record=dict(status='worker_complete',completed_folds=3,release_complete=True,
                        metrics=m.data.metrics(prob,y,splits),reference=m.data.metrics(reference,y,splits))
                    if failure=='worker':record.update(status='failed_inconclusive',error_type='FloatingPointError',error='injected worker numeric error',current_stage='inner:step7')
                    m.write(out/'worker_result.json',record)
                    np.savez(out/'partial_oof.npz',prob=prob,dev_ids=dev,fold_id=dev//2)
                def is_alive(self):return False
                def close(self):pass
            context=types.SimpleNamespace(Process=lambda **_:Process())
            original=m.publish
            original_import=m.importlib.import_module
            def publishing(*args):
                if failure=='publish':
                    (out/'screen_results.csv').write_text('partial final publication')
                    raise RuntimeError('injected final write failure')
                return original(*args)
            with patch.object(m,'ROOT',root),patch.object(m,'OWN',[]),patch.object(m.probe,'quota_guard'),\
                    patch.object(m.probe,'require_committed'),patch.object(m,'resource_contract',return_value=(frozen,'fixture==1\n')),\
                    patch.object(m,'reference_inputs',return_value=(inputs,splits,reference,[])),patch.object(m,'assert_frozen'),\
                    patch.object(m.importlib,'import_module',side_effect=lambda name,*a,**k:types.SimpleNamespace(SETTINGS=m.SETTINGS) if name=='mitra_support3072' else original_import(name,*a,**k)),\
                    patch.object(m.subprocess,'check_output',return_value='committed_fixture\n'),\
                    patch.object(m.multiprocessing,'get_context',return_value=context),patch.object(m,'supervise'),\
                    patch.object(m,'publish',side_effect=publishing):
                if failure:
                    with self.assertRaises(RuntimeError):m.run(Path('/private/not_persisted'))
                    for kind in ['artifacts','reports']:
                        folder=root/kind/m.RUN
                        self.assertFalse((folder/'screen_results.csv').exists())
                        failed=json.loads((folder/'summary.json').read_text());self.assertEqual(failed['status'],'failed_inconclusive')
                        if failure=='worker':
                            self.assertEqual(failed['worker_error_type'],'FloatingPointError')
                            self.assertEqual(failed['worker_current_stage'],'inner:step7')
                else:
                    result=m.run(Path('/private/not_persisted'));self.assertEqual(result['completed_folds'],3)
                    self.assertTrue(result['continuation']['passed'])
                protocol=json.loads((out/'protocol.json').read_text())
                self.assertEqual(protocol['source_commit'],'committed_fixture')
                self.assertNotIn('/private/not_persisted',(out/'protocol.json').read_text())

    def test_parent_publishes_only_complete_supervised_worker(self):self.parent_fixture()
    def test_parent_preserves_original_worker_error(self):self.parent_fixture('worker')
    def test_parent_finalization_failure_removes_final_csv(self):self.parent_fixture('publish')


if __name__=='__main__':unittest.main()
