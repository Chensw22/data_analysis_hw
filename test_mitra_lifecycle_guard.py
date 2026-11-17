import gc
import json
import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

import mitra_lifecycle_guard as g


class Model(torch.nn.Linear):
    def __init__(self):
        super().__init__(3,2);self.register_buffer('counter',torch.tensor(1,dtype=torch.int64))


class GuardTests(unittest.TestCase):
    def test_canonical_mps_backend_accepts_index_zero_and_rejects_mixed_devices(self):
        data=SimpleNamespace(device=torch.device('mps:0'))
        target=SimpleNamespace(device=torch.device('mps:0'))
        query=SimpleNamespace(device=torch.device('mps:0'))
        with patch.object(g,'capture_state',return_value={'fixture':'unchanged'}), \
                patch.object(g,'synchronize'),patch.object(g,'observe',return_value={}), \
                patch.object(g.gc,'collect',return_value=0),patch.object(g,'empty_cache') as cache:
            proof=g.lifecycle_barrier(None,None,data,target,query,None,None,device='mps')
            self.assertTrue(proof['state_preserved']);cache.assert_called_once_with('mps')
            for bad in [torch.device('cpu'),torch.device('mps:1')]:
                query.device=bad
                with self.assertRaises(RuntimeError):g.lifecycle_barrier(None,None,data,target,query,None,None,device='mps')
            query.device=torch.device('mps:0')
            with self.assertRaises(RuntimeError):g.lifecycle_barrier(None,None,data,target,query,None,None,device='cpu')

    def state(self):
        torch.manual_seed(7);np.random.seed(9);random.seed(11)
        model=Model();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.3)
        x=torch.arange(18,dtype=torch.float32).reshape(6,3)/18;y=torch.arange(6)%2;q=x[:2].clone()
        torch.nn.functional.cross_entropy(model(x),y).backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        return model,optimizer,x,y,q,np.random.default_rng(1),np.random.default_rng(1)

    def test_complete_state_preserved_and_plain_json_proof(self):
        values=self.state()
        proof=g.lifecycle_barrier(*values,device='cpu')
        self.assertEqual(proof['before'],proof['after']);self.assertTrue(proof['state_preserved'])
        self.assertIsInstance(proof['gc_collected'],int)
        self.assertIn('buffers',proof['before']);self.assertIn('optimizer_state',proof['before'])
        json.dumps(proof)
        def no_tensor(obj):
            self.assertFalse(torch.is_tensor(obj))
            if isinstance(obj,dict):
                for value in obj.values():no_tensor(value)
            elif isinstance(obj,list):
                for value in obj:no_tensor(value)
        no_tensor(proof)
        self.assertEqual([r['stage'] for r in proof['observations']],['before_pre_hash','after_pre_hash',
            'barrier_after_synchronize1','barrier_after_gc_collect','barrier_after_empty_cache',
            'barrier_after_synchronize2','before_post_hash','after_post_hash'])

    def test_parameter_moment_storage_mode_input_and_rng_mutations_rejected(self):
        cases=['weight','buffer','moment','replace_moment','lr','training','input','python','numpy','sampler','reference','torch']
        for case in cases:
            with self.subTest(case=case):
                values=self.state();model,opt,x,y,q,rng,ref=values
                def mutate():
                    parameter=next(model.parameters());state=opt.state[parameter]
                    with torch.no_grad():
                        if case=='weight':parameter.add_(1)
                        if case=='buffer':model.counter.add_(1)
                        if case=='moment':state['exp_avg'].add_(1)
                        if case=='replace_moment':state['exp_avg']=state['exp_avg'].clone()
                        if case=='input':x.add_(1)
                    if case=='lr':opt.param_groups[0]['lr']=.2
                    if case=='training':model.eval()
                    if case=='python':random.random()
                    if case=='numpy':np.random.random()
                    if case=='sampler':rng.random()
                    if case=='reference':ref.random()
                    if case=='torch':torch.rand(1)
                    return 0
                with tempfile.TemporaryDirectory() as folder,patch.object(g.gc,'collect',side_effect=mutate):
                    path=Path(folder)/'proof.json'
                    with self.assertRaises(RuntimeError):g.lifecycle_barrier(*values,device='cpu',log_path=path)
                    proof=json.loads(path.read_text());self.assertFalse(proof['state_preserved'])
                    self.assertTrue(proof['different_sections'])

    def test_existing_grad_is_rejected_without_clearing_it(self):
        values=self.state();parameter=next(values[0].parameters());parameter.grad=torch.ones_like(parameter)
        with patch.object(g.gc,'collect') as collect:
            with self.assertRaises(RuntimeError):g.lifecycle_barrier(*values,device='cpu')
        collect.assert_not_called();self.assertIsNotNone(parameter.grad)

    def test_scalar_bfloat16_and_shared_storage_hashes(self):
        scalar=torch.tensor(1.,dtype=torch.bfloat16);g.tensor_bytes(scalar)
        x=torch.arange(12,dtype=torch.float32);a=g.tensor_record(x);b=g.tensor_record(x[2:])
        self.assertEqual(a['storage_id'],b['storage_id']);self.assertNotEqual(a['object_id'],b['object_id'])
        self.assertNotEqual(a['storage_offset'],b['storage_offset'])
        self.assertEqual(g.tensor_record(x),g.tensor_record(x))


if __name__=='__main__':unittest.main()
