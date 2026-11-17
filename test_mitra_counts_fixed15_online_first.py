import os
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import torch
import mitra_counts_fixed15_online_first as c


def fixture(n):
    x=c.base.synthetic_frame(n)
    raw=pd.DataFrame({'PassengerId':[f'{i//3:04d}_{i%3+1:02d}' for i in range(n)],
                      'Cabin':[f'G/{i//2}/S' if i%11 else None for i in range(n)],
                      'Transported':np.arange(n)%2})
    return x,raw,c.RosterCounts(raw,x)


class CountsTests(unittest.TestCase):
    def test_fixed_counts_subset_ids_missing_target(self):
        x,raw,roster=fixture(128)
        full=roster.transform(x)
        self.assertEqual(full.loc[0,'GroupSize'],3)
        self.assertTrue(np.isnan(full.loc[0,'CabinOccupants']))
        self.assertEqual(full.loc[2,'CabinOccupants'],2)
        order=[127,3,0]
        pd.testing.assert_frame_equal(roster.transform(x.loc[order]),full.loc[order])
        pd.testing.assert_frame_equal(roster.frame_by_ids(raw.PassengerId.loc[order]),x.loc[order])
        raw.Transported=1-raw.Transported
        pd.testing.assert_frame_equal(c.RosterCounts(raw,x).counts,roster.counts)
        bad=x.loc[order].copy();bad.loc[3,'Age']=1234
        with self.assertRaises(AssertionError):roster.transform(bad)
        with self.assertRaises(ValueError):roster.frame_by_ids(['missing'])

    def test_real15_prefix_and_portable_mapping(self):
        x,raw,roster=fixture(128)
        training=np.r_[np.arange(96),np.arange(124,128)]
        prep=c.CountsPreprocessor(roster).fit(x.iloc[training],np.arange(len(training))%2,c.SEED)
        self.assertEqual(prep.n_features,15)
        q=x.iloc[[100,127,3]]
        actual=prep.transform(q)
        np.testing.assert_array_equal(actual[:,:13],prep.reference.transform(q))
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'prep.joblib';joblib.dump(prep,path);restored=joblib.load(path)
            query=restored.roster.frame_by_ids(raw.PassengerId.iloc[[100,127,3]])
            np.testing.assert_array_equal(restored.transform(query),actual)
            self.assertEqual(restored.roster.roster_sha256,roster.roster_sha256)

    def test_fixed15_persistent_optimizer_no_inner_selection(self):
        x,_,roster=fixture(8693);y=np.arange(8693)%2;ids=np.arange(8693)
        model=torch.nn.Linear(15,10);seen=[]
        def step(m,opt,data,targets,row_ids,rng,n):
            seen.append((id(opt),n,tuple(data.shape)));return dict(step=n)
        with patch.object(c,'load_roster',return_value=roster),patch.object(c.training,'fresh_model',return_value=(model,{'toy':True})),patch.object(c.training,'episode_step',side_effect=step):
            predictor=c.fit_fixed(x,y,ids,device='cpu')
        self.assertEqual([r[1] for r in seen],list(range(1,16)))
        self.assertEqual(len({r[0] for r in seen}),1)
        self.assertTrue(all(r[2]==(8693,15) for r in seen))
        self.assertFalse(predictor.metadata['inner_selection_performed'])
        self.assertEqual(predictor.metadata['refit_optimizer_steps'],15)
        self.assertEqual(predictor.support.shape,(8693,15))


if __name__=='__main__':unittest.main()
