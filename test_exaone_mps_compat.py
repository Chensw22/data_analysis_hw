import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

import exaone_mps_compat as compat


class AttentionCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def pair(self,dtype=torch.float32,scale=None):
        torch.manual_seed(42)
        native=compat.TensorAttention(24,6,dtype=dtype,score_scale=scale).eval()
        local=copy.deepcopy(native);local.__class__=compat.ChunkedTensorAttention
        return native,local

    def assert_math(self,native,local,query,context=None,**kwargs):
        with torch.no_grad():
            a=native(query.clone(),None if context is None else context.clone(),**kwargs)
            b=local(query.clone(),None if context is None else context.clone(),**kwargs)
        tolerance=2e-3 if query.dtype==torch.float16 else 3e-6
        torch.testing.assert_close(a,b,rtol=0,atol=tolerance)
        return a,b

    def test_self_attention_nondivisible_query_preserves_full_kv(self):
        native,local=self.pair()
        self.assert_math(native,local,torch.randn(2,513,24))

    def test_context_boundary_and_explicit_scale_multibatch_multihead(self):
        for rows in [256,257]:
            with self.subTest(rows=rows):
                native,local=self.pair(scale=.17)
                self.assert_math(native,local,torch.randn(2,513,24),torch.randn(2,rows,24))
        self.assertEqual(compat.flat_batch_limit(256),512);self.assertEqual(compat.flat_batch_limit(257),1)

    def test_short_context_more_than512_independent_batches(self):
        native,local=self.pair()
        self.assert_math(native,local,torch.randn(513,7,24),torch.randn(513,17,24))

    def test_fp16_first_head_and_cached_context_exact_transaction(self):
        native,local=self.pair(dtype=torch.float16,scale=.2)
        context=torch.randn(2,257,24,dtype=torch.float16);query=torch.randn(2,259,24,dtype=torch.float16)
        self.assert_math(native,local,query,context,reuse_first_context_head=True)
        self.assert_math(native,local,context,cache_context=True)
        torch.testing.assert_close(local._cached_key,native._cached_key,rtol=0,atol=0)
        torch.testing.assert_close(local._cached_value,native._cached_value,rtol=0,atol=0)
        self.assert_math(native,local,query,use_cached_context=True,reuse_first_context_head=True)
        old_key,old_value=local._cached_key,local._cached_value
        sdpa=compat.F.scaled_dot_product_attention;calls=[]
        def fail_later(*args,**kwargs):
            calls.append(1)
            if len(calls)==3:raise RuntimeError('injected attention failure')
            return sdpa(*args,**kwargs)
        with patch.object(compat.F,'scaled_dot_product_attention',side_effect=fail_later),torch.no_grad():
            with self.assertRaisesRegex(RuntimeError,'injected'):local(query,context,cache_context=True)
        self.assertEqual(len(calls),3)
        self.assertIs(local._cached_key,old_key);self.assertIs(local._cached_value,old_value)
        local.clear_cache();self.assertFalse(local.has_cached_context)

    def test_projection_cache_direct_and_instance_local_no_global_mutation(self):
        native,local=self.pair();q=torch.randn(2,259,24);context=torch.randn(2,257,24)
        with torch.no_grad():
            native.cache_context_projections(context,first_head_only=True)
            local.cache_context_projections(context,first_head_only=True)
        self.assert_math(native,local,q,use_cached_context=True,reuse_first_context_head=True)
        peer=self.pair()[0];model=nn.Sequential(native)
        original_forward=compat.TensorAttention.forward
        result=compat.install_local(model)
        self.assertIs(type(peer),compat.TensorAttention);self.assertIs(type(model[0]),compat.ChunkedTensorAttention)
        self.assertIs(compat.TensorAttention.forward,original_forward);self.assertTrue(result['instance_local'])
        with patch.object(compat,'ATTENTION_SHA','wrong'):
            with self.assertRaises(RuntimeError):compat.verify_vendor()

    def test_sdpa_full_context_and_fixed_call_dimensions(self):
        native=compat.F.scaled_dot_product_attention;calls=[]
        def observe(q,k,v,**options):
            calls.append((len(q),q.shape[-2],k.shape[-2],v.shape[-2],options.copy()))
            return native(q,k,v,**options)
        with patch.object(compat.F,'scaled_dot_product_attention',side_effect=observe):
            compat.chunked_sdpa(torch.randn(2,3,259,4),torch.randn(2,3,257,4),torch.randn(2,3,257,4),dropout_p=0.,scale=.12)
        self.assertEqual(len(calls),4)
        self.assertTrue(all(b==1 and q<=256 and k==v==257 and opts['scale']==.12 for b,q,k,v,opts in calls))


if __name__=='__main__':unittest.main()
