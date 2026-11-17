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
import mitra_peercryo_fixed15_ready as c


def fixture(n):
    x=c.base.synthetic_frame(n)
    raw=pd.DataFrame({'PassengerId':[f'{i//3:04d}_{i%3+1:02d}' for i in range(n)],
        'Cabin':[f'G/{i//2}/S' if i%11 else None for i in range(n)],
        'CryoSleep':([True,False,None]*((n+2)//3))[:n], 'Transported':np.arange(n)%2})
    return x,raw,c.PeerCryoRoster(raw,x)


class PeerCryoTests(unittest.TestCase):
    def test_self_exclusion_unknown_labels_and_subsets(self):
        x,raw,_=fixture(128)
        raw.loc[[3,4,5],'CryoSleep']=None
        raw.loc[127,'PassengerId']='9999_01'
        roster=c.PeerCryoRoster(raw,x);full=roster.transform(x)
        np.testing.assert_array_equal(full.loc[[0,1,2],'PeerKnownCryoCount'],[1,1,2])
        np.testing.assert_array_equal(full.loc[[0,1,2],'PeerCryoSleepRate'],[0,1,.5])
        self.assertTrue(full.loc[[3,4,5,127],'PeerCryoSleepRate'].isna().all())
        self.assertTrue(full.loc[[3,4,5,127],'PeerKnownCryoCount'].eq(0).all())
        raw.loc[0,'CryoSleep']=False
        altered=c.PeerCryoRoster(raw,x)
        self.assertEqual(altered.counts.loc[0,'PeerCryoSleepRate'],roster.counts.loc[0,'PeerCryoSleepRate'])
        raw.Transported=1-raw.Transported
        pd.testing.assert_frame_equal(c.PeerCryoRoster(raw,x).counts,altered.counts)
        pd.testing.assert_frame_equal(c.PeerCryoRoster(raw.drop(columns='Transported'),x).counts,altered.counts)
        order=[127,3,2,0]
        pd.testing.assert_frame_equal(roster.transform(x.loc[order]),full.loc[order])
        pd.testing.assert_frame_equal(roster.frame_by_ids(roster.passenger_ids.loc[order]),x.loc[order])
        pd.testing.assert_frame_equal(full[c.prior.COLUMNS],c.prior.RosterCounts(raw,x).transform(x))

    def test_real17_original15_prefix_and_portable_mapping(self):
        x,raw,roster=fixture(128)
        training=np.r_[np.arange(96),np.arange(124,128)]
        y=np.arange(len(training))%2
        prep=c.PeerCryoPreprocessor(roster).fit(x.iloc[training],y,c.SEED)
        original=c.prior.CountsPreprocessor(c.prior.RosterCounts(raw,x)).fit(x.iloc[training],y,c.SEED)
        self.assertEqual(prep.n_features,17)
        q=x.iloc[[100,127,3]];actual=prep.transform(q)
        np.testing.assert_array_equal(actual[:,:15],original.transform(q))
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'prep.joblib';joblib.dump(prep,path);restored=joblib.load(path)
            query=restored.roster.frame_by_ids(raw.PassengerId.iloc[[100,127,3]])
            np.testing.assert_array_equal(restored.transform(query),actual)

    def test_loader_never_reads_target(self):
        calls=[]
        def read_csv(path,**kwargs):
            selector=kwargs['usecols'];self.assertFalse(selector('Transported'));self.assertTrue(selector('CryoSleep'))
            calls.append(path)
            raise RuntimeError('stop before data access')
        with patch.object(c.pd,'read_csv',side_effect=read_csv):
            with self.assertRaisesRegex(RuntimeError,'stop before data access'):c.load_roster()
        self.assertEqual(len(calls),1)

    def test_fixed15_steps_real17_tensor(self):
        x,_,roster=fixture(8693);y=np.arange(8693)%2;ids=np.arange(8693)
        model=torch.nn.Linear(17,10);seen=[]
        def step(m,opt,data,targets,row_ids,rng,n):
            seen.append((id(opt),n,tuple(data.shape)));return dict(step=n)
        with patch.object(c,'load_roster',return_value=roster),patch.object(c.training,'fresh_model',return_value=(model,{'toy':True})),patch.object(c.training,'episode_step',side_effect=step):
            predictor=c.fit_fixed(x,y,ids,device='cpu')
        self.assertEqual([r[1] for r in seen],list(range(1,16)))
        self.assertEqual(len({r[0] for r in seen}),1)
        self.assertTrue(all(r[2]==(8693,17) for r in seen))
        self.assertFalse(predictor.metadata['inner_selection_performed'])
        self.assertEqual(predictor.metadata['refit_optimizer_steps'],15)
        self.assertEqual(predictor.support.shape,(8693,17))


if __name__=='__main__':unittest.main()
