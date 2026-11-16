import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

import mitra_full_episode_candidate as candidate


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.previous = (candidate.RUN_OUT, candidate.ACTIVE_RUNTIME, candidate.mitra.DEADLINE)

    def tearDown(self):
        candidate.RUN_OUT, candidate.ACTIVE_RUNTIME, candidate.mitra.DEADLINE = self.previous

    def test_full_episode_and_control_route_training_labels_only(self):
        x = candidate.mitra.synthetic_frame(30)
        y = np.arange(20)%2
        class Fake:
            def __init__(self): self.metadata = {}; self.released = False
            def predict_proba(self, frame):
                np.testing.assert_array_equal(frame.index, [20,21,22])
                return np.tile([.4,.6], (len(frame),1))
            def release(self): self.released = True
        trained=[]
        def fit(frame, labels, ids, *args, **kwargs):
            np.testing.assert_array_equal(frame.index, np.arange(20))
            np.testing.assert_array_equal(labels,y)
            np.testing.assert_array_equal(ids,np.arange(20))
            artifact=Fake();trained.append((artifact,args,kwargs));return artifact
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            candidate.configure_runtime(Path(folder))
            stack.enter_context(patch.object(candidate.torch.backends.mps,'is_available',return_value=True))
            stack.enter_context(patch.object(candidate.torch.mps,'set_per_process_memory_fraction'))
            stack.enter_context(patch.object(candidate.torch,'set_num_threads'))
            core = stack.enter_context(patch.object(candidate.core,'fit_full_episode',side_effect=fit))
            control = stack.enter_context(patch.object(candidate.mitra,'fit_mitra',side_effect=fit))
            for name in (candidate.NAME,candidate.CONTROL):
                p,model=candidate.fit(name,{False:(x,[])},np.arange(20),y,np.arange(20,23),7)
                np.testing.assert_array_equal(p,[.6]*3)
                self.assertTrue(model.released)
                with self.assertRaisesRegex(RuntimeError,'Existing fit log'):
                    candidate.fit(name,{False:(x,[])},np.arange(20),y,np.arange(20,23),7)
            self.assertEqual(core.call_count,1)
            self.assertEqual(control.call_count,1)
            self.assertEqual(trained[0][2]['max_steps'],50)
            self.assertEqual(trained[0][2]['device'],'mps')
            self.assertEqual(trained[1][1],(7,'mps','zero-shot'))
            self.assertEqual(trained[1][2]['query_chunk'],2318)
            for query in ([19,20],[20,20],[-1,20],[30,31]):
                with self.subTest(query=query), self.assertRaises(ValueError):
                    candidate.fit(candidate.NAME,{False:(x,[])},np.arange(20),y,np.array(query),8)

    def test_mps_requirement_and_exact_runtime_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'not-created'
            with patch.object(candidate.time,'monotonic',return_value=100):
                candidate.configure_runtime(output,stage='stability')
            self.assertEqual(candidate.mitra.DEADLINE,64900)
            self.assertFalse(output.exists())
            with patch.object(candidate.torch.backends.mps,'is_available',return_value=False):
                with self.assertRaisesRegex(RuntimeError,'no CPU fallback'):
                    candidate.start_runtime('screen')

    def test_all_sources_and_resource_records_frozen_before_baseline(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root=Path(folder);out=root/'run';out.mkdir()
            for relative in candidate.OWN_SOURCES:
                p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(relative)
            (out/'protocol.json').write_text(json.dumps(dict(sources={})))
            stack.enter_context(patch.object(candidate,'ROOT',root))
            stack.enter_context(patch.object(candidate,'verify_reference',return_value=({'sources':{}},'lock')))
            stack.enter_context(patch.object(candidate.subprocess,'check_output',return_value='test-commit\n'))
            candidate.record_protocol(out)
            saved=json.loads((out/'protocol.json').read_text())
            self.assertEqual(set(saved['sources']),set(candidate.OWN_SOURCES))
            for relative,sha in saved['sources'].items():
                self.assertEqual(candidate.mitra.sha256(out/'source'/relative),sha)
            self.assertEqual(saved['full_episode_policy']['settings'],candidate.SETTINGS)
            self.assertFalse(saved['full_episode_policy']['automatic_eligibility'])
            self.assertEqual(saved['full_episode_policy']['weekly_reserve_percent'],30)
        calls=[]
        with patch.object(candidate,'record_protocol',side_effect=lambda _:calls.append('freeze')), \
             patch.object(candidate,'BASE_EVALUATE',side_effect=lambda *a:calls.append('baseline')):
            candidate.evaluate(candidate.iterate.BASELINE,None,None,np.arange(6954),candidate.iterate.SEEDS[0],None)
            with self.assertRaises(ValueError):
                candidate.evaluate(candidate.iterate.BASELINE,None,None,np.arange(8693),candidate.iterate.SEEDS[0],None)
        self.assertEqual(calls,['freeze','baseline'])

    def test_mutated_frozen_recipe_and_environment_rejected(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            out=Path(folder)
            lock='original environment'
            env_sha=hashlib.sha256(lock.encode()).hexdigest()
            record={'full_episode_policy':dict(settings=candidate.SETTINGS.copy(),environment_sha256=env_sha)}
            (out/'full_episode_environment.lock.txt').write_text(lock)
            stack.enter_context(patch.object(candidate,'configure'))
            stack.enter_context(patch.object(candidate.iterate,'assert_frozen'))
            stack.enter_context(patch.object(candidate,'verify_reference'))
            stack.enter_context(patch.object(candidate.mitra,'environment_lock',return_value=lock))
            candidate.write_json(out/'protocol.json',record)
            candidate.assert_frozen(out)
            record['full_episode_policy']['settings']['max_steps']=100
            candidate.write_json(out/'protocol.json',record)
            with self.assertRaisesRegex(RuntimeError,'recipe/environment changed'):
                candidate.assert_frozen(out)
            record['full_episode_policy']['settings']=candidate.SETTINGS.copy()
            candidate.write_json(out/'protocol.json',record)
            (out/'full_episode_environment.lock.txt').write_text('altered')
            with self.assertRaisesRegex(RuntimeError,'recipe/environment changed'):
                candidate.assert_frozen(out)

    def test_cli_refuses_existing_run_before_any_model_or_runtime(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root=Path(folder);(root/'artifacts/run_046').mkdir(parents=True)
            stack.enter_context(patch.object(candidate,'ROOT',root))
            stack.enter_context(patch('sys.argv',['candidate','screen']))
            runtime=stack.enter_context(patch.object(candidate,'start_runtime'))
            verify=stack.enter_context(patch.object(candidate,'verify_reference'))
            with self.assertRaisesRegex(RuntimeError,'refuse overwrite'):
                candidate.main()
            runtime.assert_not_called();verify.assert_not_called()


if __name__=='__main__':
    unittest.main()
