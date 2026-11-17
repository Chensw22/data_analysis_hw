import os
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import torch
import mitra_contrasts_fixed15_online_first as c


class FixedContrastTests(unittest.TestCase):
    def test_formula_and_missing(self):
        x=c.base.synthetic_frame(8)
        x.loc[0,['FoodCourt','ShoppingMall']]=[3,4]
        x.loc[0,['RoomService','Spa','VRDeck']]=[1,2,3]
        x.loc[1,['FoodCourt','ShoppingMall']]=np.nan
        x.loc[2,['FoodCourt','ShoppingMall']]=[np.nan,5]
        got=c.add_contrasts(x)
        self.assertEqual(got.loc[0,c.EXTRA[0]],1)
        self.assertAlmostEqual(got.loc[0,c.EXTRA[1]],np.log1p(7)-np.log1p(6))
        self.assertTrue(np.isnan(got.loc[1,c.EXTRA[0]]))
        self.assertEqual(got.loc[2,c.EXTRA[0]],5-x.loc[2,['RoomService','Spa','VRDeck']].sum())

    def test_real15_prefix_unknown_and_serialization(self):
        x=c.base.synthetic_frame(128);y=np.arange(128)%2
        prep=c.ContrastPreprocessor().fit(x,y,c.SEED)
        self.assertEqual(prep.transform(x).shape,(128,15))
        q=c.base.synthetic_frame(9,123);q.loc[0,'HomePlanet']='unknown';q.loc[1,'Age']=np.nan
        expected=prep.transform(q)
        np.testing.assert_array_equal(expected[:,:13],prep.reference.transform(q))
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'prep.joblib';joblib.dump(prep,path)
            np.testing.assert_array_equal(joblib.load(path).transform(q),expected)

    def test_exact15_updates_no_inner_selection(self):
        x=c.base.synthetic_frame(8693);y=np.arange(8693)%2;ids=np.arange(8693)
        model=torch.nn.Linear(15,10)
        seen=[]
        def step(m,opt,data,targets,row_ids,rng,n):
            seen.append((id(opt),n,tuple(data.shape)))
            return dict(step=n)
        with patch.object(c.training,'fresh_model',return_value=(model,{'toy':True})),patch.object(c.training,'episode_step',side_effect=step):
            predictor=c.fit_fixed(x,y,ids,device='cpu')
        self.assertEqual([r[1] for r in seen],list(range(1,16)))
        self.assertEqual(len({r[0] for r in seen}),1)
        self.assertTrue(all(r[2]==(8693,15) for r in seen))
        self.assertEqual(predictor.support.shape,(8693,15))
        self.assertFalse(predictor.metadata['inner_selection_performed'])
        self.assertEqual(predictor.metadata['refit_optimizer_steps'],15)


if __name__=='__main__':unittest.main()
