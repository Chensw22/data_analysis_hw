import unittest
from unittest.mock import patch

import numpy as np
from mitra_seed_mean_ensemble import SeedMeanPredictor


class FakeModel:
    resident=0
    def __init__(self,p,seed,fail=False):
        self.p=np.asarray(p);self.metadata=dict(seed=seed,recipe=dict(max_steps=50))
        self.labels=np.array([0,1]);self.active=False;self.fail=fail
    def predict_proba(self,x):
        if FakeModel.resident:raise AssertionError('Concurrent engines')
        FakeModel.resident+=1;self.active=True
        if self.fail:raise RuntimeError('prediction failed')
        return self.p
    def release(self):
        if self.active:FakeModel.resident-=1;self.active=False


class Tests(unittest.TestCase):
    def test_fixed_mean_releases_each_engine_and_serializes_cpu(self):
        model=SeedMeanPredictor([FakeModel([[.8,.2]],20261112),FakeModel([[.2,.8]],20261212)])
        with patch('mitra_prior_adapt_ensemble.mitra.check_deadline'):
            np.testing.assert_allclose(model.predict_proba([0]),[[.5,.5]])
        self.assertEqual(FakeModel.resident,0)
        model.device='mps';self.assertEqual(model.__getstate__()['device'],'cpu')

    def test_only_correct_ordered_seeds_and_failure_release(self):
        with self.assertRaises(ValueError):
            SeedMeanPredictor([FakeModel([[.5,.5]],20261112),FakeModel([[.5,.5]],20261113)])
        model=SeedMeanPredictor([FakeModel([[.5,.5]],20261112,True),FakeModel([[.5,.5]],20261212)])
        with self.assertRaises(RuntimeError),patch('mitra_prior_adapt_ensemble.mitra.check_deadline'):
            model.predict_proba([0])
        self.assertEqual(FakeModel.resident,0)


if __name__=='__main__':unittest.main()
