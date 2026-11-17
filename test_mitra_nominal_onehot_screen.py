"""Synthetic CPU/mock screen tests, never invoke candidate model inference."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import mitra_nominal_onehot_screen as m


def fixture():
    dev=np.arange(6);y=dev%2;splits=[(dev[dev//2!=k],dev[dev//2==k]) for k in range(3)]
    inputs=dict(dev=dev,y=y,features=pd.DataFrame({'feature':dev}),passengers=np.asarray(['a1','a2','b1','b2','c1','c2'],dtype=object),groups=np.array(['a','a','b','b','c','c']))
    return inputs,splits,{m.REFERENCE:.2+.6*y}


def metadata(ids,labels,seed,settings):
    return dict(status='fit_complete',candidate=m.NAME,settings=settings,seed=seed,training_rows=len(ids),
        training_ids_sha256=m.base.ids_hash(ids),labels_sha256=m.base.ids_hash(labels),checkpoint_sha256=m.base.WEIGHT_SHA,
        optimizer_steps=0,full_prediction_support=True,schema=dict(columns=['x'],active_columns=['x'],vocabularies={'HomePlanet':['A']},
            parent_by_column={'x':'HomePlanet'},derived_mirror_mask=[1],original11_exact=True,actual_retained_features=1))


class NominalScreenTests(unittest.TestCase):
    def test_fixed_gain_loss_and_two_fold_gate(self):
        raw=dict(accuracy=.8,log_loss=.4,fold_accuracy=[.8]*3)
        self.assertFalse(m.continuation(raw,raw)['passed'])
        candidate=dict(raw,accuracy=.801)
        self.assertTrue(m.continuation(candidate,raw)['passed'])
        for changed in [dict(accuracy=.8009),dict(log_loss=.401),dict(fold_accuracy=[.81,.79,.79])]:
            self.assertFalse(m.continuation(dict(candidate,**changed),raw)['passed'])

    def test_identity_zero_optimizer_and_schema_required(self):
        ids=np.arange(4);y=ids%2;core=types.SimpleNamespace(SETTINGS={'optimizer_steps':0});record=metadata(ids,y,17,core.SETTINGS)
        m.check_fit_metadata(record,ids,y,17,core)
        for field,value in [('optimizer_steps',1),('seed',18),('full_prediction_support',False),('training_ids_sha256','wrong')]:
            with self.subTest(field=field),self.assertRaises(RuntimeError):m.check_fit_metadata(dict(record,**{field:value}),ids,y,17,core)
        bad=copy.deepcopy(record);bad['schema']['original11_exact']=False
        with self.assertRaises(RuntimeError):m.check_fit_metadata(bad,ids,y,17,core)

    def test_resource_refusal_precedes_development_targets(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(m,'ROOT',Path(directory)),patch.object(m,'quota_guard'),\
                patch.object(m,'resource_contract',side_effect=RuntimeError('unqualified')),patch.object(m,'reference_inputs') as inputs:
            with self.assertRaisesRegex(RuntimeError,'unqualified'):m.run(Path('/private/not_read'))
            inputs.assert_not_called()

    def test_resource_contract_complete_worker_and_identity(self):
        plan=(m.ROOT/m.PLAN).read_bytes();lock='fixture==1\n';digest=hashlib.sha256(lock.encode()).hexdigest()
        current=dict(sources={},evidence_files={},environment_sha256=digest,checkpoint_sha256=m.base.WEIGHT_SHA,source_commit='new')
        saved=dict(current,source_commit='previous');worker={'cases':[{'fixture':True}],'tiny':{'fixture':True}}
        probe=types.SimpleNamespace(contract=lambda:(current,lock),qualify=lambda record:True,check_frozen=lambda record:None)
        original_import=m.importlib.import_module
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/m.PLAN).parent.mkdir(parents=True);(root/m.PLAN).write_bytes(plan)
            public=root/m.RESOURCE;local=root/m.ARTIFACT_RESOURCE;public.mkdir(parents=True);local.mkdir(parents=True)
            for folder in [public,local]:
                m.write(folder/'freeze.json',saved);m.write(folder/'worker_result.json',worker);(folder/'requirements.lock').write_text(lock)
            result=dict(status='complete',resource_qualification=True,cases=worker['cases'],tiny=worker['tiny'],worker_result_sha256=m.base.sha256(public/'worker_result.json'))
            for folder in [public,local]:m.write(folder/'result.json',result)
            with patch.object(m,'ROOT',root),patch.object(m.data,'ROOT',root),patch.object(m,'assert_frozen'),patch.object(m.data.common,'require_committed'),\
                    patch.object(m.data,'report_pair'),patch.object(m.data.common,'verify_protocol',return_value=({'mitra_environment_sha256':digest},{})),\
                    patch.object(m.importlib,'import_module',side_effect=lambda name,*a,**k:probe if name=='mitra_nominal_onehot_probe' else original_import(name,*a,**k)):
                frozen,actual_lock=m.resource_contract();self.assertEqual(actual_lock,lock);self.assertEqual(len(frozen['evidence']),8)
                for folder in [public,local]:m.write(folder/'result.json',dict(result,resource_qualification=False))
                with self.assertRaises(RuntimeError):m.resource_contract()

    def test_external900s_deadline_kills_only_spawned_worker(self):
        class Process:
            exitcode=None;alive=True;killed=False
            def is_alive(self):return self.alive
            def join(self,timeout):pass
            def kill(self):self.killed=True;self.alive=False;self.exitcode=-9
        process=Process()
        with self.assertRaisesRegex(TimeoutError,'900s'):m.supervise(process,time.monotonic()-1)
        self.assertTrue(process.killed)

    def exercise_folds(self,fail=False):
        inputs,splits,references=fixture();made=[];seeds=[];settings={'optimizer_steps':0}
        class Fake:
            def __init__(self,x,labels,ids,seed,device):
                if made and not made[-1].released:raise AssertionError('Previous engine resident.')
                self.released=False;self.support=np.zeros((len(ids),1),dtype=np.float32);self.labels=labels;self.query_chunk=2318
                self.metadata=metadata(ids,labels,seed,settings)
                self.metadata['transformed_support_sha256']=hashlib.sha256(self.support.tobytes()).hexdigest()
                made.append(self);seeds.append(seed)
            def predict_proba(self,q):
                if fail and len(made)==2:raise RuntimeError('injected failure')
                p=.1+.8*(q.feature.to_numpy()%2);return np.column_stack([1-p,p])
            def release(self):self.released=True
        core=types.SimpleNamespace(SETTINGS=settings,fit_nominal_onehot=Fake);original_import=m.importlib.import_module
        probe=types.SimpleNamespace(predict=lambda model,q:(model.predict_proba(q),[dict(support_rows=len(model.labels),label_rows=len(model.labels),query_rows=len(q))]))
        with tempfile.TemporaryDirectory() as directory:
            out,report=Path(directory)/'artifact',Path(directory)/'report';out.mkdir();report.mkdir()
            with patch.object(m.importlib,'import_module',side_effect=lambda name,*a,**k:core if name=='mitra_nominal_onehot' else probe if name=='mitra_nominal_onehot_probe' else original_import(name,*a,**k)),\
                    patch.object(m,'assert_frozen'),patch.object(m,'quota_guard') as quota,patch.object(m.base,'release_memory'),patch.object(m,'snapshot',return_value={}):
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'injected'):m.train_folds(out,report,inputs,splits,references,{},time.monotonic()+20,Path('/private'))
                    with np.load(out/'partial_oof.npz') as cache:self.assertEqual(np.isfinite(cache['prob']).sum(),2)
                else:
                    result=m.train_folds(out,report,inputs,splits,references,{},time.monotonic()+20,Path('/private'))
                    self.assertEqual(result['completed_folds'],3);self.assertEqual(seeds,[20261112,20261113,20261114]);self.assertEqual(quota.call_count,3)
                    self.assertEqual((out/'fold0.json').read_bytes(),(report/'fold0.json').read_bytes())
                self.assertTrue(all(x.released for x in made));self.assertFalse((out/'screen_results.csv').exists())

    def test_complete_folds_seed_quota_schema_and_release(self):self.exercise_folds()
    def test_partial_failure_keeps_partial_without_csv(self):self.exercise_folds(True)

    def parent_fixture(self,failure=None):
        inputs,splits,references=fixture();prob=.1+.8*inputs['y'];frozen={'sources':{},'evidence':{}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'artifacts'/m.RUN
            class Process:
                exitcode=0
                def start(self):
                    child=dict(status='worker_complete',completed_folds=3,release_complete=True,metrics=m.data.metrics(prob,inputs['y'],splits),
                        references={k:m.data.metrics(p,inputs['y'],splits) for k,p in references.items()})
                    if failure=='worker':child.update(status='failed_inconclusive',error_type='FloatingPointError',error='injected bad probabilities')
                    m.write(out/'worker_result.json',child)
                    np.savez(out/'partial_oof.npz',prob=prob,dev_ids=inputs['dev'],fold_id=inputs['dev']//2)
                    for fold,(a,b) in enumerate(splits):
                        m.write(out/f'fold{fold}.json',{'fold':fold});np.savez(out/f'fold{fold}_rows.npz',train_ids=a,validation_ids=b)
                def is_alive(self):return False
                def close(self):pass
            context=types.SimpleNamespace(Process=lambda **_:Process());original_publish=m.publish
            def publishing(*args):
                if failure=='publish':
                    (out/'screen_results.csv').write_text('partial publication');raise RuntimeError('injected final failure')
                return original_publish(*args)
            with patch.object(m,'ROOT',root),patch.object(m,'OWN',[]),patch.object(m,'quota_guard'),\
                    patch.object(m.data.common,'require_committed'),patch.object(m,'resource_contract',return_value=(frozen,'fixture==1\n')),\
                    patch.object(m,'reference_inputs',return_value=(inputs,splits,references)),patch.object(m,'assert_frozen'),\
                    patch.object(m.subprocess,'check_output',return_value='fixture_commit\n'),\
                    patch.object(m.multiprocessing,'get_context',return_value=context),patch.object(m,'supervise'),patch.object(m,'publish',side_effect=publishing):
                if failure:
                    with self.assertRaises(RuntimeError):m.run(Path('/private/not_persisted'))
                    for kind in ['artifacts','reports']:
                        folder=root/kind/m.RUN;self.assertFalse((folder/'screen_results.csv').exists());self.assertFalse((folder/'assessment.json').exists())
                        failed=json.loads((folder/'summary.json').read_text());self.assertEqual(failed['status'],'failed_inconclusive')
                        if failure=='worker':self.assertEqual(failed['worker_error_type'],'FloatingPointError')
                else:
                    result=m.run(Path('/private/not_persisted'));self.assertEqual(result['completed_folds'],3);self.assertFalse(result['S_supported'])
                    with np.load(out/f'{m.NAME}_seed20261112.npz',allow_pickle=False) as saved:self.assertEqual(saved['passenger_ids'].dtype.kind,'U')
                    with self.assertRaisesRegex(RuntimeError,'overwrite'):m.run(Path('/private/not_persisted'))
                self.assertNotIn('/private/not_persisted',(out/'protocol.json').read_text());self.assertFalse((out/'selection.json').exists())

    def test_parent_complete_publication(self):self.parent_fixture()
    def test_parent_failure_retains_worker_error(self):self.parent_fixture('worker')
    def test_parent_finalization_failure_removes_final_outputs(self):self.parent_fixture('publish')

    def test_cleanup_error_keeps_original_worker_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            out,report=Path(directory)/'artifact',Path(directory)/'report';out.mkdir();report.mkdir()
            with patch.object(m,'assert_frozen',side_effect=ValueError('original fit failure')),\
                    patch.object(m.base,'DEADLINE',None),\
                    patch.object(m.base,'release_memory',side_effect=RuntimeError('cleanup failure')):
                m.worker(out,report,{},[],{}, {},time.monotonic()+30,Path('/private'))
            record=m.read(out/'worker_result.json');self.assertEqual(record['error_type'],'ValueError')
            self.assertEqual(record['error'],'original fit failure');self.assertEqual(record['cleanup_error_type'],'RuntimeError')
            self.assertFalse(record['release_complete'])


if __name__=='__main__':unittest.main()
