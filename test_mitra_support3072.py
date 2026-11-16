"""CPU-only selection/refit and global-ID counterfactual tests; no real data/weights."""
import contextlib
import gc
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import mitra_support3072 as core


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(3, 4), torch.nn.Linear(4, 10)])

    def forward(self, query):
        return self.layers[1](torch.tanh(self.layers[0](query)))


class Prep:
    fitted = []
    n_features = 3

    def fit(self, x, y, seed):
        type(self).fitted.append(x.index.to_numpy().copy())
        return self

    def transform(self, x):
        return x.to_numpy(np.float32, copy=True)


class Support3072Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        Prep.fitted = []
        self.x = pd.DataFrame(np.arange(12600).reshape(4200, 3)/12600,
                              index=np.arange(4200)*7+101)
        self.ids = self.x.index.to_numpy()
        self.y = np.arange(4200)%2

    def test_settings_only_change_support_and_sampler(self):
        changed = {k for k in core.SETTINGS if core.SETTINGS[k] != core.original.SETTINGS[k]}
        self.assertEqual(changed, {'episode_support', 'sampler'})
        self.assertEqual(core.SETTINGS['episode_support'], 3072)
        self.assertEqual(core.SETTINGS['sampler'], core.probe.RECIPE['sampler'])
        self.assertIs(core.fresh_model, core.original.fresh_model)

    def test_native_ten_class_ce_global_hashes_and_no_extra_rng_advance(self):
        torch.manual_seed(5)
        model = Tiny()
        tensor = torch.randn(3708, 3); labels = torch.arange(3708)%2
        ids = np.arange(3708)*11+70
        rng, reference, expected = [np.random.default_rng(9) for _ in range(3)]
        optimizer = core.base.optimizer_for(model)
        def forward(net, support, y, query):
            self.assertEqual((len(support), len(query)), (3072, 256))
            return net(query)
        with patch.object(core.base, 'model_forward', forward):
            for step in [1, 2]:
                permutation = expected.permutation(len(tensor))
                query = permutation[1024:1280]
                support = np.r_[permutation[:1024], permutation[1280:3328]]
                loss = float(F.cross_entropy(model(tensor[query]), labels[query]).detach())
                row = core.episode_step(model, optimizer, tensor, labels, ids, rng, reference, step, lambda *a:None)
                self.assertAlmostEqual(row['training_full_head_ce'], loss, places=6)
                self.assertEqual(row['query_ids_sha256'], core.base.ids_hash(ids[query]))
                self.assertEqual(row['support_prefix_ids_sha256'], core.base.ids_hash(ids[permutation[:1024]]))
                self.assertEqual(row['support_ids_sha256'], core.base.ids_hash(ids[support]))
                self.assertEqual(row['local_query_ids_sha256'], core.base.ids_hash(query))
                self.assertEqual(rng.bit_generator.state, expected.bit_generator.state)
                self.assertEqual(reference.bit_generator.state, expected.bit_generator.state)
                self.assertTrue(row['finite_gradients'])
                self.assertEqual(row['cumulative_optimizer_steps'], step)
                self.assertAlmostEqual(row['learning_rate'], core.base.literal_learning_rate(step))
                self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_inner_selection_earliest_tie_zero_and_fresh_refit(self):
        for selected in [0, 2]:
            with self.subTest(selected=selected):
                Prep.fitted = []
                model_refs, states, optimizer_refs, validations = [], [], [], []
                factory = core.base.optimizer_for
                def fresh(device):
                    if model_refs:self.assertIsNone(model_refs[0](), 'Inner model retained across reset')
                    model = Tiny(); model_refs.append(weakref.ref(model))
                    states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
                    return model, {'tiny':True}
                def optimizer_for(model):
                    if optimizer_refs:self.assertIsNone(optimizer_refs[0](), 'Inner optimizer retained across reset')
                    optimizer = factory(model); optimizer_refs.append(weakref.ref(optimizer)); return optimizer
                def predict(model, data, targets, valid, chunk):
                    self.assertEqual(chunk, 2318)
                    self.assertTrue(all(p.grad is None for p in model.parameters()))
                    self.assertFalse(set(data[:, 0].tolist()) & set(valid[:, 0].tolist()))
                    validations.append((len(data),len(valid)))
                    return np.full((len(valid),2),.5)
                losses = [1.]*51 if selected == 0 else [1.,.9]+[.8]*49
                with patch.object(core.base,'MitraPreprocessor',Prep), patch.object(core,'fresh_model',fresh), \
                     patch.object(core.base,'optimizer_for',optimizer_for), patch.object(core.base,'model_forward',lambda m,s,y,q:m(q)), \
                     patch.object(core.base,'predict_arrays',predict), patch.object(core,'log_loss',side_effect=losses), \
                     patch.object(core.base,'release_memory',side_effect=gc.collect), contextlib.redirect_stdout(io.StringIO()):
                    artifact = core.fit_support3072(self.x,self.y,self.ids,9,device='cpu')
                meta = artifact.metadata
                self.assertEqual(meta['status'],'fit_complete')
                self.assertEqual(meta['selected_steps'],selected)
                self.assertEqual(meta['refit_optimizer_steps'],selected)
                self.assertEqual(meta['inner_optimizer_steps'],50)
                self.assertEqual(len(model_refs),2)
                self.assertTrue(all(ref() is None for ref in model_refs+optimizer_refs))
                self.assertEqual(len(optimizer_refs),1+(selected>0))
                self.assertEqual(validations,[(3360,840)]*51)
                self.assertEqual(len(Prep.fitted[0]),3360)
                np.testing.assert_array_equal(Prep.fitted[1],self.ids)
                self.assertEqual(meta['inner_train_ids_sha256'],core.base.ids_hash(Prep.fitted[0]))
                self.assertEqual([r['step'] for r in meta['inner_history']],list(range(51)))
                self.assertEqual([r['step'] for r in meta['refit_history']],list(range(1,selected+1)))
                for phase, history, pool in [('inner',meta['inner_history'][1:],Prep.fitted[0]),('refit',meta['refit_history'],self.ids)]:
                    rng = np.random.default_rng(9)
                    for row in history:
                        old_support, old_query = core.probe.original.sample_episode(len(pool),rng)
                        self.assertEqual(row['stage'],phase)
                        self.assertEqual(row['query_ids_sha256'],core.base.ids_hash(pool[old_query]))
                        self.assertEqual(row['support_prefix_ids_sha256'],core.base.ids_hash(pool[old_support]))
                self.assertEqual(meta['inner_history'][0]['validation_optimizer_steps'],[])
                self.assertEqual(meta['inner_history'][50]['validation_optimizer_steps'],[50])
                self.assertEqual(artifact.query_chunk,2318)
                np.testing.assert_array_equal(artifact.labels,self.y)
                self.assertEqual(len(artifact.support),4200)
                for key in states[0]:
                    torch.testing.assert_close(states[0][key],states[1][key],rtol=0,atol=0)
                    if not selected:torch.testing.assert_close(artifact.state[key],states[1][key],rtol=0,atol=0)
                if selected:self.assertTrue(any(not torch.equal(artifact.state[k],states[1][k]) for k in states[1]))

    def test_capacity_duplicate_ids_and_horizon_fail_before_model(self):
        with patch.object(core,'fresh_model',side_effect=AssertionError('must not load')):
            with self.assertRaises(ValueError):core.fit_support3072(self.x,self.y,self.ids,9,max_steps=100)
            with self.assertRaises(ValueError):core.fit_support3072(self.x,self.y,np.zeros(4200),9)
            with self.assertRaises(ValueError):core.fit_support3072(self.x.iloc[:4000],self.y[:4000],self.ids[:4000],9)
            with self.assertRaises(ValueError):core.fit_support3072(self.x,self.y,self.ids+.5,9)

    def test_validation_bad_shape_nonfinite_or_moment_mutation_fails(self):
        model = Tiny(); optimizer = core.base.optimizer_for(model)
        data = torch.randn(3360,3); labels = torch.arange(3360)%2; valid = torch.randn(840,3)
        for prediction in [np.ones((840,10))/10,np.full((840,2),np.nan),np.full((840,2),.8)]:
            with patch.object(core.base,'predict_arrays',return_value=prediction):
                with self.assertRaises(RuntimeError):core.validation_prediction(model,optimizer,data,labels,valid)
        def mutated(*args,**kwargs):
            p = next(model.parameters())
            optimizer.state[p] = dict(step=torch.tensor(1.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
            return np.full((840,2),.5)
        with patch.object(core.base,'predict_arrays',mutated):
            with self.assertRaises(RuntimeError):core.validation_prediction(model,optimizer,data,labels,valid)

    def test_failure_logs_inconclusive_and_releases_model(self):
        refs = []
        def fresh(device):
            m = Tiny(); refs.append(weakref.ref(m)); return m,{}
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/'fit.json'
            with patch.object(core.base,'MitraPreprocessor',Prep),patch.object(core,'fresh_model',fresh), \
                 patch.object(core.base,'predict_arrays',return_value=np.full((840,2),.5)), \
                 patch.object(core.base,'release_memory',side_effect=gc.collect), \
                 patch.object(core,'episode_step',side_effect=TimeoutError('synthetic deadline')), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(TimeoutError):core.fit_support3072(self.x,self.y,self.ids,9,device='cpu',log_path=log)
            record = json.loads(log.read_text())
            self.assertEqual(record['status'],'failed_inconclusive')
            self.assertEqual(record['error_type'],'TimeoutError')
            self.assertEqual(len(record['inner_history']),1)
            self.assertEqual(record['refit_history'],[])
            gc.collect()
            self.assertIsNone(refs[0]())

    def test_source_mutation_is_rejected(self):
        expected = core.source_hashes()
        with patch.object(core,'source_hashes',return_value={**expected,'mitra_support3072.py':'changed'}):
            with self.assertRaises(RuntimeError):core.assert_sources(expected)


if __name__ == '__main__':
    unittest.main()
