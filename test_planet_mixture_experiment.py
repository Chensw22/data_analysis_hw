"""Synthetic optimizer and orchestration tests; no native models/real targets."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock,patch

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import planet_mixture_experiment as e


class GateTests(unittest.TestCase):
    def test_objective_gradient_and_regularization_definition(self):
        raw=np.array([0.,.2,.7,1.,.4]);kumo=np.array([.3,.7,.2,.8,.4]);y=np.array([0,1,0,1,1])
        index=np.array([0,1,2,3,-1]);alpha=np.array([.2,.3,.6,.7])
        value,grad=e.gate_objective(alpha,raw,kumo,y,index,.4)
        r=np.clip(raw,1e-7,1-1e-7);k=np.clip(kumo,1e-7,1-1e-7)
        weights=np.r_[alpha,.4];p=(1-weights)*r+weights*k
        expected=-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))+.01*np.mean((alpha-.5)**2)
        self.assertAlmostEqual(value,expected,places=14)
        for j in range(4):
            a=alpha.copy();b=alpha.copy();a[j]+=1e-6;b[j]-=1e-6
            numerical=(e.gate_objective(a,raw,kumo,y,index,.4)[0]-e.gate_objective(b,raw,kumo,y,index,.4)[0])/2e-6
            self.assertAlmostEqual(grad[j],numerical,places=7)

    def test_missing_unknown_constant_fallback_and_unclipped_final(self):
        values=e.normalize_planets([None,np.nan,'','  ','__MISSING__','Earth','Pluto'])
        self.assertEqual(values.tolist(),['__MISSING__']*5+['Earth','Pluto'])
        gate=dict(kind='planet',alpha=[1.,0.,.2,.8],constant_fallback=.3)
        actual,unknown=e.apply_gate(gate,np.array([0.,1.,0.]),np.array([1.,0.,1.]),['Earth',None,'Pluto'])
        np.testing.assert_allclose(actual,[1.,.2,.3]);self.assertEqual(unknown,1)
        y=np.arange(80)%2;raw=np.where(y,.75,.25);kumo=np.where(y,.8,.2)
        constant=e.fit_gate(raw,kumo,y,['Earth']*80,'constant')
        planet=e.fit_gate(raw,kumo,y,['Earth']*80,'planet',constant['alpha'][0])
        self.assertTrue(constant['success']);self.assertTrue(planet['success'])
        np.testing.assert_array_equal(planet['alpha'][1:],[.5,.5,.5])
        with self.assertRaises(ValueError):e.fit_gate(raw,kumo,y+.1,['Earth']*80,'constant')

    def test_optimizer_failure_or_nonfinite_gradient_rejected(self):
        raw=np.array([.2,.8]);y=np.array([0,1])
        for result in [Mock(success=False,message='not converged'),
                       Mock(success=True,fun=.5,jac=np.array([np.nan]),x=np.array([.5]),message='bad')]:
            with patch.object(e,'minimize',return_value=result):
                with self.assertRaises(RuntimeError):e.fit_gate(raw,raw,y,['Earth']*2,'constant')

    def test_inner_partition_outer_disjoint_and_label_counts(self):
        e.validate_rows(np.array([1,3]),np.array([5]),np.array([1,3,5]),np.array([7,9]),np.array([0,1]))
        for a,b,outer,held,y in [([1,3],[3],[1,3],[7],[0,1]),([1],[5],[1,5],[5],[0]),
                                ([1],[5],[1,3,5],[7],[0]),([1],[5],[1,5],[7],[0,1])]:
            with self.assertRaises(ValueError):e.validate_rows(np.array(a),np.array(b),np.array(outer),np.array(held),np.array(y))

    def test_continuation_no_constant_fallback_and_strict_improvement(self):
        base=dict(accuracy=.8,log_loss=.4,fold_accuracy=[.8,.8,.8])
        scores={key:copy.deepcopy(base) for key in ['raw','kumo','constant','planet']}
        self.assertFalse(e.continuation(scores)['passed'])
        scores['planet']['log_loss']=.39;self.assertTrue(e.continuation(scores)['passed'])
        scores['kumo']['accuracy']=.81;self.assertFalse(e.continuation(scores)['passed'])
        scores['planet']['accuracy']=.82;scores['planet']['fold_accuracy']=[.79,.79,.9]
        self.assertFalse(e.continuation(scores)['passed'])


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.stack=ExitStack()
        for key,value in [('ROOT',self.root),('OUTPUT',self.root/'artifacts'/e.DIRECTORY),('REPORT',self.root/'reports'/e.DIRECTORY)]:
            self.stack.enter_context(patch.object(e,key,value))
        e.write(self.root/e.PLAN,dict(recipe=e.RECIPE,continuation=e.CONTINUATION))
        self.stack.enter_context(patch.object(e,'protocol_contract',return_value={'sources':{},'evidence':{}}))
        self.stack.enter_context(patch.object(e,'assert_frozen'))
        dev=np.arange(6954,dtype=np.int64);y=dev%2
        splits=list(StratifiedKFold(3,shuffle=True,random_state=20261112).split(dev,y))
        self.inputs=dict(dev=dev,y=y,splits=splits,roster=pd.DataFrame({'HomePlanet':['Earth']*12970}),
            features_sha256='synthetic',planets=np.asarray(e.CATEGORIES*1739)[:6954],
            probabilities={'raw':np.where(y,.65,.35),'kumo':np.where(y,.7,.3)})
        self.stack.enter_context(patch.object(e,'load_inputs',return_value=self.inputs))
        self.deadlines=[];self.seen=[]

    def tearDown(self):self.stack.close();self.tmp.cleanup()

    def fake_job(self,kind,out,deadline):
        self.deadlines.append(deadline);job=e.read(out/'job.json')
        with np.load(out/'rows.npz',allow_pickle=False) as packet:
            self.assertEqual(set(packet.files),{'train_ids','query_ids','outer_train_ids','outer_validation_ids','train_labels'})
            a=packet['train_ids'];b=packet['query_ids'];y=packet['train_labels']
            e.validate_rows(a,b,packet['outer_train_ids'],packet['outer_validation_ids'],y)
            np.testing.assert_array_equal(y,self.inputs['y'][a])
            self.assertEqual(job['seed'],20261117+100*job['outer_fold']+job['inner_fold'])
            # Intentionally different from cached outer predictions, so fit provenance is testable.
            p=np.where(b%2,.8 if kind=='raw' else .85,.2 if kind=='raw' else .15)
            np.savez(out/'prediction.npz',query_ids=b,prob=p)
            e.write(out/'metadata.json',{'synthetic':True})
            record=dict(status='worker_complete',release_complete=True,kind=kind,seed=job['seed'],
                outer_fold=job['outer_fold'],inner_fold=job['inner_fold'],train_ids_sha256=e.ids_hash(a),
                query_ids_sha256=e.ids_hash(b),train_labels_sha256=e.ids_hash(y),prediction_rows=len(b),
                query_labels_read=False,outer_validation_labels_read=False,prediction_sha256=e.sha(out/'prediction.npz'),
                metadata_sha256=e.sha(out/'metadata.json'))
        e.write(out/'worker_result.json',record);self.seen.append((kind,job['outer_fold'],job['inner_fold']))
        return record

    def test_all18_serial_jobs_gate_receives_only_complete_inner_oof_and_cache_alignment(self):
        original=e.fit_gate;calls=[]
        def gate(raw,kumo,y,planet,kind,constant_alpha=None):
            self.assertEqual(len(raw),4636);self.assertEqual(len(y),4636)
            np.testing.assert_allclose(raw,np.where(y,.8,.2));np.testing.assert_allclose(kumo,np.where(y,.85,.15))
            calls.append(kind);return original(raw,kumo,y,planet,kind,constant_alpha)
        with patch.object(e,'execute_job',side_effect=self.fake_job),patch.object(e,'fit_gate',side_effect=gate):result=e.run()
        self.assertEqual(result['completed_fits'],18);self.assertEqual(result['completed_outer_folds'],3)
        self.assertEqual(calls,['constant','planet']*3);self.assertEqual(len(set(self.deadlines)),1)
        self.assertEqual(self.seen,[(kind,o,i) for o in range(3) for i in range(3) for kind in ['raw','kumo']])
        for k,(a,b) in enumerate(self.inputs['splits']):
            with np.load(e.OUTPUT/f'outer{k}_inner_oof.npz',allow_pickle=False) as cache:
                np.testing.assert_array_equal(cache['outer_train_ids'],a)
                self.assertEqual(set(cache.files),{'outer_train_ids','raw','kumo','inner_fold_id'})
                self.assertEqual(set(cache['inner_fold_id']),{0,1,2})
            with np.load(e.OUTPUT/f'outer{k}_predictions.npz',allow_pickle=False) as cache:
                np.testing.assert_array_equal(cache['outer_validation_ids'],b)
                np.testing.assert_array_equal(cache['raw'],self.inputs['probabilities']['raw'][b])
        self.assertEqual(e.read(e.REPORT/'progress.json')['status'],'complete')
        with self.assertRaises(FileExistsError):e.run()

    def test_failure_preserves_first_worker_error_and_never_fits_gate(self):
        def failed(kind,out,deadline):
            e.write(out/'worker_result.json',dict(status='failed_inconclusive',error_type='RuntimeError',error='original native failure'))
            raise RuntimeError('original native failure')
        with patch.object(e,'execute_job',side_effect=failed),patch.object(e,'fit_gate') as fit:
            with self.assertRaisesRegex(RuntimeError,'original native failure'):e.run()
            fit.assert_not_called()
        record=e.read(e.REPORT/'failure.json')
        self.assertEqual(record['worker_failure']['error'],'original native failure')
        self.assertFalse((e.REPORT/'summary.json').exists());self.assertFalse((e.OUTPUT/'results.csv').exists())

    def test_supervisor_kills_only_owned_worker_at_shared_deadline(self):
        process=Mock();process.poll.return_value=None
        with self.assertRaises(TimeoutError):e.supervise(process,time.monotonic()-1)
        process.kill.assert_called_once_with();process.wait.assert_called_once_with(timeout=5)

    def test_worker_rejects_any_query_label_packet_before_native_setup(self):
        import hashlib
        out=e.OUTPUT/'jobs'/'extra_labels';out.mkdir(parents=True)
        frozen=dict(sources={},evidence={},models={'raw':dict(environment_sha256=hashlib.sha256(b'fixture').hexdigest(),weights={})})
        e.write(e.OUTPUT/'freeze.json',frozen)
        np.savez(out/'rows.npz',train_ids=np.array([0,1]),query_ids=np.array([2]),train_labels=np.array([0,1]),
                 outer_train_ids=np.array([0,1,2]),outer_validation_ids=np.array([3]),query_labels=np.array([1]))
        e.write(out/'job.json',dict(kind='raw',outer_fold=0,inner_fold=0,seed=20261117,rows_sha256=e.sha(out/'rows.npz')))
        with patch.object(e.history,'require_committed'),patch.object(e.sys,'prefix',str(self.root/'.venv-mitra')), \
             patch.object(e,'environment_lock',return_value='fixture'),patch.object(e.audit,'initialize_reference') as native:
            result=e.worker('raw',out,time.monotonic()+60)
        native.assert_not_called();self.assertEqual(result['status'],'failed_inconclusive')
        self.assertIn('unexpected labels',result['error'])

    def test_failure_after_cache_creation_relabels_oof_as_partial(self):
        def fail_after_write(*args):
            np.savez(e.OUTPUT/'oof.npz',synthetic=np.arange(2))
            raise RuntimeError('synthetic interrupted scoring')
        with patch.object(e,'execute_job',side_effect=self.fake_job),patch.object(e,'metrics',side_effect=fail_after_write):
            with self.assertRaises(RuntimeError):e.run()
        self.assertFalse((e.OUTPUT/'oof.npz').exists());self.assertTrue((e.OUTPUT/'oof.partial.npz').exists())
        self.assertFalse((e.REPORT/'summary.json').exists())

    def test_reordered_prediction_invalid(self):
        out=e.OUTPUT/'jobs'/'synthetic';out.mkdir(parents=True)
        a=np.array([0,2]);b=np.array([4,6]);y=np.array([0,1])
        job=dict(kind='raw',seed=20261117,outer_fold=0,inner_fold=0)
        np.savez(out/'prediction.npz',query_ids=b[::-1],prob=np.array([.2,.3]));e.write(out/'metadata.json',{})
        record=dict(status='worker_complete',release_complete=True,**job,train_ids_sha256=e.ids_hash(a),
            query_ids_sha256=e.ids_hash(b),train_labels_sha256=e.ids_hash(y),prediction_rows=2,
            query_labels_read=False,outer_validation_labels_read=False,prediction_sha256=e.sha(out/'prediction.npz'),
            metadata_sha256=e.sha(out/'metadata.json'))
        with self.assertRaisesRegex(RuntimeError,'reordered'):e.verify_prediction(out,record,job,a,b,y)


if __name__=='__main__':unittest.main()
