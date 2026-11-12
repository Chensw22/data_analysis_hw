import gc
import json
import tempfile
import unittest
import weakref
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np

import mitra_mirror_ensemble as candidate


class MirrorTests(unittest.TestCase):
    def fake_runtime(self, fail_seed=None):
        calls, live = [], []
        released = []
        base = 7
        probabilities = {base:.1,base+100:.3,base+200:.5,base+300:.9}
        class Fake:
            def __init__(self, rows, seed):
                self.seed = seed
                self.support = np.zeros((rows,13),dtype=np.float32)
                self.preprocessor = SimpleNamespace(upstream=SimpleNamespace(
                    mirror=np.random.RandomState(seed).choice([1,-1],size=(1,13))))
                self.metadata = {'seed':seed,'optimizer_steps':0}
            def predict_proba(self, x):
                p = np.full(len(x),probabilities[self.seed])
                return np.column_stack([1-p,p])
            def release(self):
                released.append(self.seed)
        def fit(x,y,ids,seed,device,recipe,**kwargs):
            gc.collect()
            self.assertFalse(any(ref() is not None for ref in live), 'Prior model object still resident.')
            calls.append(dict(seed=seed,rows=len(x),labels=np.array(y),ids=np.array(ids),
                              recipe=recipe,kwargs=kwargs))
            if seed == fail_seed:
                raise RuntimeError('synthetic component failure')
            model = Fake(len(x),seed)
            live.append(weakref.ref(model))
            return model
        stack = ExitStack()
        stack.enter_context(patch.object(candidate.mitra,'fit_mitra',side_effect=fit))
        stack.enter_context(patch.object(candidate.mitra,'release_memory'))
        stack.enter_context(patch.object(candidate.mitra,'snapshot',return_value={'fake_cpu':True}))
        stack.enter_context(patch.object(candidate.mitra,'check_deadline'))
        return stack,calls,released

    def artifact(self, version=None):
        x = candidate.mitra.synthetic_frame(12)
        return candidate.MirrorContextPredictor(x,np.arange(12)%2,np.arange(12)+101,
            7,version or candidate.NAME,device='cpu')

    def test_exact_fixed_mean_complete_views_and_one_model_at_a_time(self):
        artifact = self.artifact()
        query = candidate.mitra.synthetic_frame(3,8)
        runtime,calls,released = self.fake_runtime()
        with runtime:
            result = artifact.predict_proba(query,[900,901,902])
        np.testing.assert_allclose(result,np.tile([.55,.45],(3,1)),rtol=0,atol=1e-15)
        self.assertEqual([call['seed'] for call in calls],[7,107,207,307])
        self.assertEqual(released,[7,107,207,307])
        self.assertTrue(all(call['rows']==12 and call['kwargs']['query_chunk']==2318 for call in calls))
        self.assertEqual(artifact.metadata['completed_views'],4)
        self.assertEqual(artifact.metadata['weights'],[.25]*4)
        self.assertEqual(len(artifact.expected_mirror_hashes),4)
        self.assertTrue(artifact.metadata['components'][0]['offset0_is_original_v71'])

    def test_matched_control_is_exact_offset_zero(self):
        artifact = self.artifact(candidate.CONTROL)
        runtime,calls,_ = self.fake_runtime()
        with runtime:
            result = artifact.predict_proba(candidate.mitra.synthetic_frame(2),[501,502])
        np.testing.assert_array_equal(result,np.tile([.9,.1],(2,1)))
        self.assertEqual([call['seed'] for call in calls],[7])
        self.assertEqual(artifact.metadata['weights'],[1.])

    def test_failed_component_never_returns_or_saves_partial_mean(self):
        artifact = self.artifact()
        runtime,calls,released = self.fake_runtime(fail_seed=207)
        with tempfile.TemporaryDirectory() as directory, runtime:
            log, predictions = Path(directory)/'log.json',Path(directory)/'pred.npz'
            with self.assertRaisesRegex(RuntimeError,'synthetic component failure'):
                artifact.predict_proba(candidate.mitra.synthetic_frame(2),[501,502],log,predictions)
            self.assertFalse(predictions.exists())
            record=json.loads(log.read_text())
            self.assertEqual(record['status'],'failed_incomplete_ensemble')
            self.assertEqual(record['completed_views'],2)
            self.assertNotIn('mean_probabilities_sha256',record)
        self.assertEqual(released,[7,107])
        self.assertEqual([call['seed'] for call in calls],[7,107,207])

    def test_context_recipe_roundtrip_has_no_network_state_cache(self):
        artifact = self.artifact()
        query = candidate.mitra.synthetic_frame(3,8)
        runtime,_,_ = self.fake_runtime()
        with tempfile.TemporaryDirectory() as directory, runtime:
            expected = artifact.predict_proba(query,[900,901,902])
            path=Path(directory)/'model.joblib'
            joblib.dump(artifact,path)
            self.assertLess(path.stat().st_size,100000)
            restored=joblib.load(path)
            np.testing.assert_array_equal(restored.labels,artifact.labels)
            np.testing.assert_array_equal(restored.train_ids,artifact.train_ids)
            self.assertTrue(restored.training.equals(artifact.training))
            np.testing.assert_array_equal(restored.predict_proba(query,[900,901,902]),expected)
            self.assertNotIn('state',restored.__dict__)
            self.assertNotIn('_engine',restored.__dict__)
            restored.expected_mirror_hashes[0]='incorrect'
            with self.assertRaisesRegex(RuntimeError,'mirror differs'):
                restored.predict_proba(query,[900,901,902])

    def test_outer_ids_and_frozen_recipe_fail_before_model_prediction(self):
        artifact = self.artifact()
        with patch.object(candidate.mitra,'fit_mitra') as fit:
            with self.assertRaisesRegex(ValueError,'disjoint'):
                artifact.predict_proba(candidate.mitra.synthetic_frame(2),[101,501])
            fit.assert_not_called()
            artifact.offsets=(0,200)
            with self.assertRaisesRegex(RuntimeError,'recipe changed'):
                artifact.predict_proba(candidate.mitra.synthetic_frame(2),[500,501])
            fit.assert_not_called()

    def test_baseline_capture_and_control_configuration_are_exact(self):
        original = (candidate.iterate.fit,candidate.iterate.evaluate,candidate.iterate.CONFIGS,candidate.iterate.BLEND)
        try:
            candidate.configure()
            self.assertEqual(candidate.iterate.CONFIGS[candidate.CONTROL]['offsets'],[0])
            self.assertEqual(candidate.iterate.CONFIGS[candidate.CONTROL]['probability_weights'],[1.])
            self.assertEqual(candidate.iterate.CONFIGS[candidate.NAME]['offsets'],[0,100,200,300])
            with patch.object(candidate,'BASE_FIT',return_value=('p','model')) as base:
                self.assertEqual(candidate.fit(candidate.iterate.BASELINE,{},[0],[1],[2],7),('p','model'))
                base.assert_called_once_with(candidate.iterate.BASELINE,{},[0],[1],[2],7)
        finally:
            candidate.iterate.fit,candidate.iterate.evaluate,candidate.iterate.CONFIGS,candidate.iterate.BLEND=original

    def test_runtime_uses_prespecified_stage_deadline_without_gpu_work(self):
        original = candidate.RUN_OUT, candidate.mitra.DEADLINE
        try:
            with patch.object(candidate,'configure'), patch.object(candidate.time,'monotonic',return_value=100), \
                 patch.object(candidate.mitra.torch.backends.mps,'is_available',return_value=True), \
                 patch.object(candidate.mitra.torch,'set_num_threads'), \
                 patch.object(candidate.mitra.torch.mps,'set_per_process_memory_fraction') as cap:
                candidate.configure_runtime(Path('/unused/screen'))
                self.assertEqual(candidate.mitra.DEADLINE,1900)
                candidate.configure_runtime(Path('/unused/stability'),stage='stability')
                self.assertEqual(candidate.mitra.DEADLINE,3700)
                self.assertEqual(candidate.RUN_OUT,Path('/unused/stability'))
                cap.assert_called_with(.65)
                with self.assertRaises(ValueError):
                    candidate.configure_runtime(Path('/unused'),stage='export')
        finally:
            candidate.RUN_OUT, candidate.mitra.DEADLINE = original


if __name__=='__main__':
    unittest.main()
