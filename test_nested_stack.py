"""Verify the nested stack's fold boundaries without training expensive models."""
import unittest
from unittest.mock import patch
import numpy as np
import nested_stack


class NestedBoundaries(unittest.TestCase):
    def test_outer_validation_never_enters_base_or_meta_training(self):
        train=np.arange(18);valid=np.arange(18,22);labels=train%2;calls=[]
        def fake_pair(matrices,fit_idx,y,predict_idx,seed,keep=False):
            self.assertFalse(set(fit_idx)&set(valid))
            self.assertFalse(set(fit_idx)&set(predict_idx))
            np.testing.assert_array_equal(y,labels[fit_idx])
            calls.append((np.asarray(fit_idx),np.asarray(predict_idx)))
            p=.25+.5*(np.asarray(predict_idx)%2)
            return np.column_stack([p,.9*p+.05]),None,None
        with patch.object(nested_stack,'pair_fit',fake_pair):
            p,_=nested_stack.fit('v35_nested_stack',{},train,labels,valid,42)
        self.assertEqual(len(calls),4)
        np.testing.assert_array_equal(np.sort(np.concatenate([p for _,p in calls[:3]])),train)
        np.testing.assert_array_equal(calls[-1][0],train)
        np.testing.assert_array_equal(calls[-1][1],valid)
        self.assertEqual(p.shape,(4,))
        self.assertTrue(np.isfinite(p).all())

    def test_endpoint_probabilities_have_finite_logits(self):
        self.assertTrue(np.isfinite(nested_stack.logits(np.array([0.,.5,1.]))).all())


if __name__=='__main__':
    unittest.main()
