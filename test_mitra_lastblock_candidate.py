import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import mitra_lastblock_candidate as candidate


TINY = dict(dim=8, dim_output=10, n_layers=2, n_heads=2, task='CLASSIFICATION')


class CandidateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        candidate.mitra.DEADLINE = None

    def tearDown(self):
        candidate.mitra.DEADLINE = None

    def fixture(self):
        return candidate.mitra.synthetic_frame(30), np.arange(30)%2, np.arange(30)+100

    def tiny_loader(self, calls):
        def load(directory, device):
            torch.manual_seed(912)
            model = candidate.mitra.make_model(TINY, device=device)
            calls.append({n:p.clone() for n,p in model.state_dict().items()})
            return model, TINY.copy()
        return load

    def test_inner_isolation_selected_count_and_fresh_refit(self):
        x,y,ids = self.fixture()
        initial_states, preprocess_rows, episodes = [], [], []
        real_prep = candidate.mitra.MitraPreprocessor.fit
        real_step = candidate.episode_update
        def prep_fit(self, frame, labels, seed):
            preprocess_rows.append(frame.index.to_numpy().copy())
            np.testing.assert_array_equal(labels, y[frame.index])
            return real_prep(self, frame, labels, seed)
        def update(model, optimizer, data, labels, yy, ii, rng, step):
            episodes.append((ii.copy(), step))
            return real_step(model, optimizer, data, labels, yy, ii, rng, step)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(candidate.mitra, 'load_initial_model', side_effect=self.tiny_loader(initial_states)))
            stack.enter_context(patch.object(candidate.mitra.MitraPreprocessor, 'fit', prep_fit))
            stack.enter_context(patch.object(candidate, 'episode_update', side_effect=update))
            stack.enter_context(patch.object(candidate, 'log_loss', side_effect=[.6,.2,.3]))
            result = candidate.fit_lastblock(x,y,ids,7,'cpu',Path(directory)/'fit.json',max_steps=2)
        self.assertEqual(len(initial_states), 2)
        for name, value in initial_states[0].items():
            torch.testing.assert_close(value, initial_states[1][name], atol=0, rtol=0)
        self.assertEqual([len(rows) for rows in preprocess_rows], [24,30])
        self.assertEqual([len(i) for i,_ in episodes], [24,24,30])
        self.assertEqual([s for _,s in episodes], [1,2,1])
        self.assertEqual(result.metadata['selected_steps'], 1)
        self.assertEqual(result.metadata['refit_optimizer_steps'], 1)
        self.assertEqual(len(result.support),30)
        self.assertEqual(result.metadata['validation_prefix_cache_device'], ['cpu','cpu'])
        self.assertTrue(result.metadata['selection_weights_discarded'])
        self.assertTrue(result.metadata['frozen_prefix_unchanged_refit'])
        np.testing.assert_array_equal(episodes[0][0], ids[preprocess_rows[0]])
        np.testing.assert_array_equal(episodes[-1][0], ids)

    def test_step_zero_is_exact_original_zero_shot_and_earliest_tie(self):
        x,y,ids = self.fixture()
        calls=[]
        with patch.object(candidate.mitra,'load_initial_model',side_effect=self.tiny_loader(calls)), \
             patch.object(candidate,'log_loss',side_effect=[.2,.2,.3]):
            adapted = candidate.fit_lastblock(x,y,ids,7,'cpu',max_steps=2)
            control = candidate.mitra.fit_mitra(x,y,ids,7,'cpu','zero-shot',query_chunk=2318)
        self.assertEqual(adapted.metadata['selected_steps'],0)
        self.assertEqual(adapted.metadata['refit_history'],[])
        query=candidate.mitra.synthetic_frame(5,91)
        np.testing.assert_array_equal(adapted.predict_proba(query),control.predict_proba(query))

    def test_nonfinite_inner_metric_aborts_without_refit(self):
        x,y,ids = self.fixture()
        calls=[]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(candidate.mitra,'load_initial_model',side_effect=self.tiny_loader(calls)), \
             patch.object(candidate,'log_loss',return_value=float('nan')):
            log=Path(directory)/'fit.json'
            with self.assertRaises(FloatingPointError):
                candidate.fit_lastblock(x,y,ids,7,'cpu',log,max_steps=2)
            self.assertEqual(json.loads(log.read_text())['status'],'failed_inconclusive')
        self.assertEqual(len(calls),1)

    def test_configure_runtime_exact_destination_and_phase_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'stability'
            with patch.object(candidate.time,'monotonic',return_value=100):
                candidate.configure_runtime(output,stage='stability')
            self.assertEqual(candidate.RUN_OUT,output)
            self.assertFalse(output.exists())
            self.assertEqual(candidate.mitra.DEADLINE,43300)

    def test_wrapper_passes_only_train_labels_and_disjoint_query(self):
        x,y,ids=self.fixture()
        class Fake:
            metadata={}
            def predict_proba(self, frame):
                np.testing.assert_array_equal(frame.index,[20,21,22])
                return np.tile([.4,.6],(len(frame),1))
            def release(self): pass
        calls=[]
        def train(frame,labels,train_ids,*args):
            calls.append((frame.index.to_numpy(),np.array(labels),train_ids.copy()))
            return Fake()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            candidate.configure_runtime(Path(directory))
            stack.enter_context(patch.object(candidate.torch.backends.mps,'is_available',return_value=True))
            stack.enter_context(patch.object(candidate.torch.mps,'set_per_process_memory_fraction'))
            stack.enter_context(patch.object(candidate,'fit_lastblock',side_effect=train))
            p,_=candidate.fit(candidate.NAME,{False:(x,[])},np.arange(20),y[:20],np.arange(20,23),7)
            np.testing.assert_array_equal(p,[.6,.6,.6])
            with self.assertRaises(ValueError):
                candidate.fit(candidate.NAME,{False:(x,[])},np.arange(20),y[:20],np.array([19,20]),7)
        self.assertEqual(len(calls),1)
        np.testing.assert_array_equal(calls[0][0],np.arange(20))
        np.testing.assert_array_equal(calls[0][1],y[:20])
        np.testing.assert_array_equal(calls[0][2],np.arange(20))

    def test_freeze_includes_core_validator_audits_policy_before_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'run';out.mkdir()
            for name in candidate.OWN_SOURCES:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
            (out/'protocol.json').write_text(json.dumps(dict(sources={})))
            with patch.object(candidate,'ROOT',root), \
                 patch.object(candidate,'verify_reference',return_value=({'sources':{}},'synthetic-lock')):
                candidate.record_protocol(out)
            saved=json.loads((out/'protocol.json').read_text())
            self.assertEqual(set(saved['sources']),set(candidate.OWN_SOURCES))
            self.assertIn('lastblock_policy',saved)
        calls=[]
        with patch.object(candidate,'record_protocol',side_effect=lambda out:calls.append('freeze')), \
             patch.object(candidate,'BASE_EVALUATE',side_effect=lambda *args:calls.append('baseline')):
            candidate.evaluate(candidate.iterate.BASELINE,None,None,None,1,Path('unused'))
        self.assertEqual(calls,['freeze','baseline'])


if __name__ == '__main__':
    unittest.main()
