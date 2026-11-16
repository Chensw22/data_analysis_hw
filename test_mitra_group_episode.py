"""CPU-only checks of group sampling and fresh selected-count refitting."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

import mitra_group_episode as core


class Prep:
    fitted=[]
    n_features=3
    def fit(self,x,y,seed):
        type(self).fitted.append(x.index.to_numpy().copy());return self
    def transform(self,x):return x.to_numpy(np.float32).copy()


class GroupEpisodeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.old_deadline=core.base.DEADLINE;core.base.DEADLINE=None
    def tearDown(self):core.base.DEADLINE=self.old_deadline

    def test_deterministic_query_kept_and_support_excludes_whole_query_groups(self):
        groups=np.repeat(np.arange(20),2).astype(str)
        rng=np.random.default_rng(17);expected_rng=np.random.default_rng(17)
        for _ in range(3):
            permutation=expected_rng.permutation(len(groups));query=permutation[8:12]
            expected=permutation[~np.isin(groups[permutation],groups[query])][:8]
            support,actual=core.sample_group_episode(groups,rng,8,4)
            np.testing.assert_array_equal(actual,query);np.testing.assert_array_equal(support,expected)
            self.assertFalse(set(groups[support])&set(groups[actual]))

    def test_shortage_fails_after_one_permutation_without_query_resampling(self):
        groups=np.repeat('same',20)
        rng=np.random.default_rng(3);repeat=np.random.default_rng(3)
        repeat.permutation(len(groups))
        with self.assertRaisesRegex(ValueError,'do not resample'):
            core.sample_group_episode(groups,rng,8,4)
        np.testing.assert_array_equal(rng.permutation(20),repeat.permutation(20))

    def test_inner_is_exact_first_sgkf5_and_group_disjoint(self):
        groups=np.repeat(np.arange(20),2);y=np.arange(40)%2
        expected=next(StratifiedGroupKFold(5,shuffle=True,random_state=19).split(np.zeros(40),y,groups.astype(str)))
        inner,valid=core.inner_split(y,groups,19)
        np.testing.assert_array_equal(inner,expected[0]);np.testing.assert_array_equal(valid,expected[1])
        self.assertFalse(set(groups[inner])&set(groups[valid]))
        with self.assertRaises(ValueError):core.inner_split(y,np.array(['']*40),19)

    def test_native10_ce_persistent_optimizer_and_gradient_clear_boundary(self):
        torch.manual_seed(8);model=torch.nn.Linear(3,10);data=torch.randn(40,3)
        targets=torch.arange(40)%2;ids=np.arange(40)+100;groups=np.repeat(np.arange(20),2).astype(str)
        optimizer=core.base.optimizer_for(model)
        rng=np.random.default_rng(4);repeat=np.random.default_rng(4)
        sampler=core.sample_group_episode
        with patch.object(core,'sample_group_episode',lambda g,r:sampler(g,r,8,4)), \
                patch.object(core.base,'model_forward',lambda model,s,y,q:model(q)):
            for step in (1,2):
                _,query=sampler(groups,repeat,8,4)
                expected=float(F.cross_entropy(model(data[query]),targets[query]).detach())
                row=core.episode_step(model,optimizer,data,targets,ids,groups,rng,step)
                self.assertAlmostEqual(row['training_full_head_ce'],expected,places=6)
                self.assertEqual(row['cumulative_optimizer_steps'],step)
                self.assertEqual(row['group_overlap_count'],0)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertTrue(all(int(s['step'])==step for s in optimizer.state.values()))

    def test_group_inner_selection_and_fresh_refit_only_selected_count_transfers(self):
        x=pd.DataFrame(np.arange(120).reshape(40,3)/120,index=np.arange(40)*3+101)
        ids=x.index.to_numpy();y=np.arange(40)%2;groups=np.repeat(np.arange(20),2).astype(str)
        lookup={float(v):g for v,g in zip(x.iloc[:,0].to_numpy(np.float32),groups)}
        sampler=core.sample_group_episode
        for selected in (0,1):
            with self.subTest(selected=selected):
                Prep.fitted=[];models=[];initial=[];validation=[];episodes=[]
                def fresh(device):
                    model=torch.nn.Linear(3,10);models.append(model)
                    initial.append({k:v.detach().clone() for k,v in model.state_dict().items()})
                    return model,{'toy':True}
                def forward(model,support,labels,query):
                    left={lookup[v] for v in support[:,0].tolist()}
                    right={lookup[v] for v in query[:,0].tolist()}
                    self.assertFalse(left&right)
                    episodes.append(len(models));return model(query)
                def predict(model,support,labels,query,chunk):
                    self.assertEqual(chunk,2318)
                    self.assertFalse({lookup[v] for v in support[:,0].tolist()}&{lookup[v] for v in query[:,0].tolist()})
                    self.assertTrue(all(p.grad is None for p in model.parameters()))
                    validation.append((len(support),len(query)))
                    return np.full((len(query),2),.5)
                losses=[1.]*51 if selected==0 else [1.]+[.9]*50
                with patch.object(core.base,'MitraPreprocessor',Prep),patch.object(core,'fresh_model',fresh), \
                        patch.object(core,'sample_group_episode',lambda g,r:sampler(g,r,8,4)), \
                        patch.object(core.base,'model_forward',forward),patch.object(core.base,'predict_arrays',predict), \
                        patch.object(core,'log_loss',side_effect=losses),contextlib.redirect_stdout(io.StringIO()):
                    artifact=core.fit_group_episode(x,y,ids,groups,9,'cpu')
                inner,valid=core.inner_split(y,groups,9)
                np.testing.assert_array_equal(Prep.fitted[0],ids[inner]);np.testing.assert_array_equal(Prep.fitted[1],ids)
                self.assertEqual(validation,[(len(inner),len(valid))]*51)
                self.assertEqual(episodes.count(1),50);self.assertEqual(episodes.count(2),selected)
                self.assertEqual(artifact.metadata['selected_steps'],selected)
                self.assertEqual(artifact.metadata['refit_optimizer_steps'],selected)
                self.assertEqual(artifact.metadata['inner_group_overlap_count'],0)
                self.assertEqual(len(artifact.labels),40);self.assertEqual(artifact.query_chunk,2318)
                for key in initial[0]:
                    torch.testing.assert_close(initial[0][key],initial[1][key],atol=0,rtol=0)
                    if selected==0:torch.testing.assert_close(artifact.state[key],initial[1][key],atol=0,rtol=0)

    def test_no_horizon_extension_or_duplicate_ids(self):
        x=pd.DataFrame(np.zeros((20,3)));y=np.arange(20)%2;groups=np.arange(20).astype(str)
        with patch.object(core,'fresh_model',side_effect=AssertionError('must not load')):
            with self.assertRaises(ValueError):core.fit_group_episode(x,y,np.arange(20),groups,7,max_steps=100)
            with self.assertRaises(ValueError):core.fit_group_episode(x,y,np.zeros(20),groups,7)


if __name__=='__main__':unittest.main()
