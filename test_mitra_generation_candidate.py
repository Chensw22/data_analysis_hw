import copy
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
import subprocess
import sys
from contextlib import ExitStack

import joblib
import numpy as np

import mitra_generation_candidate as candidate


class GenerationTests(unittest.TestCase):
    def setUp(self):
        candidate.PROBABILITY_CACHE.clear()
        self.x=candidate.mitra.synthetic_frame(12)
        self.query=candidate.mitra.synthetic_frame(3,9)
        self.ids=np.arange(12)+100
        self.y=np.arange(12)%2

    def fake(self,failed=None):
        calls=[];live=[];released=[]
        class Model:
            def __init__(self,generation,x):
                self.generation=generation;self.support=np.zeros((len(x),13))
                self.preprocessor=SimpleNamespace(upstream=SimpleNamespace(mirror=np.ones((1,13))))
                self.metadata={'generation':generation}
            def predict_proba(self,x):
                p=.2 if self.generation=='v1' else .8
                return np.tile([1-p,p],(len(x),1))
            def release(self):released.append(self.generation)
        def fit(generation,x,y,ids,seed,device):
            gc.collect()
            self.assertFalse(any(ref() is not None for ref in live))
            calls.append((generation,np.array(ids),np.array(y),seed))
            if generation==failed:raise RuntimeError('component failed')
            model=Model(generation,x);live.append(weakref.ref(model));return model
        stack=ExitStack()
        stack.enter_context(patch.object(candidate,'fit_component',side_effect=fit))
        stack.enter_context(patch.object(candidate,'verify_checkpoint',return_value=candidate.mitra.MODEL_CONFIG))
        stack.enter_context(patch.object(candidate.mitra,'release_memory'))
        stack.enter_context(patch.object(candidate.mitra,'check_deadline'))
        stack.enter_context(patch.object(candidate.mitra,'snapshot',return_value={'cpu_mock':True}))
        return stack,calls,released

    def artifact(self,version=None,ids=None,labels=None):
        return candidate.GenerationPredictor(self.x,self.y if labels is None else labels,
            self.ids if ids is None else ids,7,version or candidate.NAME,device='cpu')

    def test_sequential_exact_mean_cached_components_and_no_cross_fold_reuse(self):
        stack,calls,released=self.fake()
        with stack:
            v1=self.artifact(candidate.V1).predict_proba(self.query,[500,501,502],use_cache=True)
            v2=self.artifact(candidate.CONTROL).predict_proba(self.query,[500,501,502],use_cache=True)
            mean=self.artifact().predict_proba(self.query,[500,501,502],use_cache=True)
            np.testing.assert_array_equal(mean,.5*v1+.5*v2)
            self.assertEqual(len(calls),2)
            self.assertEqual(released,['v1','v2'])
            self.artifact(ids=self.ids+1000).predict_proba(self.query,[500,501,502],use_cache=True)
            self.assertEqual(len(calls),4)
            self.artifact(labels=1-self.y).predict_proba(self.query,[500,501,502],use_cache=True)
            self.assertEqual(len(calls),6)

    def test_failed_component_has_no_partial_mean_or_prediction_file(self):
        artifact=self.artifact();stack,calls,released=self.fake(failed='v2')
        with tempfile.TemporaryDirectory() as folder,stack:
            log=Path(folder)/'log.json';predictions=Path(folder)/'p.npz'
            with self.assertRaisesRegex(RuntimeError,'component failed'):
                artifact.predict_proba(self.query,[500,501,502],log_path=log,prediction_path=predictions)
            self.assertFalse(predictions.exists())
            record=json.loads(log.read_text())
            self.assertEqual(record['status'],'failed_incomplete')
            self.assertEqual(record['completed_components'],1)
            self.assertNotIn('probability_sha256',record)
        self.assertEqual(released,['v1'])

    def test_serialization_rebuilds_both_components_and_rejects_recipe_changes(self):
        artifact=self.artifact();stack,calls,_=self.fake()
        with tempfile.TemporaryDirectory() as folder,stack:
            p=artifact.predict_proba(self.query,[500,501,502],use_cache=True)
            path=Path(folder)/'model.joblib';joblib.dump(artifact,path)
            self.assertLess(path.stat().st_size,100000)
            restored=joblib.load(path)
            self.assertNotIn('_engine',restored.__dict__)
            self.assertNotIn('state',restored.__dict__)
            np.testing.assert_array_equal(restored.predict_proba(self.query,[500,501,502]),p)
            self.assertEqual(len(calls),4,'Reload must not reuse process probability cache.')
            restored.checkpoints['v1']['sha256']='changed'
            with self.assertRaisesRegex(RuntimeError,'recipe changed'):
                restored.predict_proba(self.query,[500,501,502])

    def test_query_label_boundary_and_v2_exact_original_dispatch(self):
        with patch.object(candidate,'fit_component') as fit:
            with self.assertRaisesRegex(ValueError,'disjoint'):
                self.artifact().predict_proba(self.query,[100,501,502])
            fit.assert_not_called()
        with patch.object(candidate,'verify_checkpoint'),patch.object(candidate.mitra,'fit_mitra',return_value='original') as fit:
            result=candidate.fit_component('v2',self.x,self.y,self.ids,7,'mps')
            self.assertEqual(result,'original')
            fit.assert_called_once_with(self.x,self.y,self.ids,7,'mps','zero-shot',
                checkpoint_dir=candidate.ROOT/candidate.CHECKPOINTS['v2']['directory'],query_chunk=2318)
        self.assertEqual(candidate.mitra.WEIGHT_SHA,candidate.CHECKPOINTS['v2']['sha256'])

    def test_v1_verifier_rejects_wrong_sha_without_modifying_v2(self):
        old=candidate.mitra.WEIGHT_SHA
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);directory=root/'v1';directory.mkdir()
            (directory/'model.safetensors').write_bytes(b'abc')
            (directory/'config.json').write_text(json.dumps(candidate.mitra.MODEL_CONFIG))
            specs=copy.deepcopy(candidate.CHECKPOINTS)
            specs['v1'].update(directory='v1',bytes=3)
            with patch.object(candidate,'ROOT',root),patch.object(candidate,'CHECKPOINTS',specs):
                with self.assertRaisesRegex(RuntimeError,'wrong SHA'):
                    candidate.verify_checkpoint('v1')
                specs['v1']['sha256']=candidate.hashlib.sha256(b'abc').hexdigest()
                self.assertEqual(candidate.verify_checkpoint('v1'),candidate.mitra.MODEL_CONFIG)
        self.assertEqual(candidate.mitra.WEIGHT_SHA,old)

    def test_cpu_context_artifact_loads_in_fresh_process_without_model_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'context.joblib';joblib.dump(self.artifact(),path)
            code=('import joblib,sys; m=joblib.load(sys.argv[1]); '
                  'assert m.__class__.__module__=="mitra_generation_candidate"; '
                  'assert m.training.shape==(12,13); assert len(m.labels)==12; '
                  'assert m.generations==["v1","v2"]; assert "_engine" not in m.__dict__; '
                  'print("portable CPU context loaded; no inference")')
            result=subprocess.run([sys.executable,'-c',code,str(path)],cwd=candidate.ROOT,
                capture_output=True,text=True,check=True,timeout=30)
            self.assertIn('no inference',result.stdout)

    def test_all_three_resource_shapes_required_before_any_screen(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);lock='mock==1\n'
            with patch.object(candidate,'ROOT',root),patch.object(candidate,'source_hashes',return_value={'source':'sha'}), \
                 patch.object(candidate.mitra,'environment_lock',return_value=lock):
                for shape,(n,q) in candidate.SHAPES.items():
                    candidate.write_json(candidate.probe_path(shape)/'result.json',dict(status='complete',
                        support_rows=n,query_rows=q,settings=candidate.SETTINGS,checkpoints=candidate.CHECKPOINTS,
                        source_hashes={'source':'sha'},environment_sha256=candidate.hashlib.sha256(lock.encode()).hexdigest(),
                        full_support=True,subset_passed=True,reload_passed=True,completed_generations=['v1','v2']))
                self.assertEqual(set(candidate.require_resources()),set(candidate.SHAPES))
                path=candidate.probe_path('final')/'result.json';complete=json.loads(path.read_text())
                for field,value in [('completed_generations',['v1']),('support_rows',8192),('reload_passed',False),('status','incomplete')]:
                    candidate.write_json(path,{**complete,field:value})
                    with self.subTest(field=field),self.assertRaises(RuntimeError):candidate.require_resources()


if __name__=='__main__':unittest.main()
