"""Synthetic attribute/contract tests only; no model fit or real data parsing."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock,patch

import numpy as np
import pandas as pd

import consistent_repair_candidate as e
import iterate


def roster():
    planets=['Earth',None,'Europa','Mars',None,None,None,'Earth',None,None,None,None]
    cabins=['A/1/P',None,'B/2/P','C/3/P','C/4/S','D/5/P',None,'E/6/P','E/7/S',None,None,None]
    groups=['0001']*3+['0002']*2+['0003']*2+['0004']*3+['0005']*2
    frame=pd.DataFrame(dict(PassengerId=[f'{g}_{i:02}' for i,g in enumerate(groups)],HomePlanet=planets,
        CryoSleep=[True,True,False,False,None,True,False,'True',True,False,False,True],
        Cabin=cabins,Destination=[None]*12,Age=[np.nan]*12,VIP=[False]*12,Name=[None]*12))
    for name in e.SPEND:frame[name]=np.nan
    frame.loc[0,'RoomService']=17.;frame.loc[2,'FoodCourt']=9.
    return frame


class RepairTests(unittest.TestCase):
    def test_exact_three_rules_observed_and_unrelated_columns_unchanged(self):
        frame=roster();original=frame.copy(deep=True);base,cats=iterate.build_features(frame,False)
        (fixed,new_cats),masks,proof=e.repair_features(frame,iterate)
        pd.testing.assert_frame_equal(frame,original)
        self.assertEqual(list(fixed.columns),list(base.columns));self.assertEqual(len(fixed.columns),30)
        self.assertEqual(new_cats,cats);self.assertEqual(proof['model_columns'],e.COMPACT)
        self.assertEqual(fixed.loc[0,'RoomService'],17.)
        self.assertTrue(fixed.loc[0,e.SPEND[1:]].eq(0).all())
        self.assertEqual(fixed.loc[2,'FoodCourt'],9.)
        for row in [2,3,4,6,7,9,10]:
            for name in e.SPEND:
                if pd.isna(frame.loc[row,name]):self.assertTrue(pd.isna(fixed.loc[row,name]))
        self.assertEqual(fixed.loc[4,'HomePlanet'],'Mars')
        self.assertEqual(fixed.loc[1,'HomePlanet'],'__MISSING__')  # Earth/Europa disagreement.
        self.assertEqual(fixed.loc[6,'HomePlanet'],'__MISSING__')  # No known planet.
        self.assertEqual(fixed.loc[9,'HomePlanet'],'Earth')
        self.assertEqual(fixed.loc[1,'Side'],'P');self.assertEqual(fixed.loc[6,'Side'],'P')
        self.assertEqual(fixed.loc[9,'Side'],'__MISSING__')  # P/S disagreement.
        self.assertEqual(fixed.loc[10,'Side'],'__MISSING__');self.assertEqual(fixed.loc[11,'Side'],'__MISSING__')
        self.assertEqual(fixed.loc[1,'Deck'],'__MISSING__');self.assertTrue(pd.isna(fixed.loc[1,'CabinNumber']))
        untouched=[c for c in base if c not in e.ALLOWED]
        pd.testing.assert_frame_equal(base[untouched],fixed[untouched])
        pd.testing.assert_frame_equal(base.loc[~masks['any_repair'],e.COMPACT],fixed.loc[~masks['any_repair'],e.COMPACT])
        self.assertTrue(proof['observed_cells_unchanged']);self.assertTrue(proof['extra17_columns_unchanged'])
        self.assertEqual(proof['changed_rows'],int(masks['any_repair'].sum()))
        self.assertNotIn('PassengerId',fixed);self.assertNotIn('GroupId',fixed)

    def test_peer_evidence_includes_unlabeled_other_roster_partition_without_cascade(self):
        frame=roster().iloc[:2].copy();frame.loc[0,'HomePlanet']=None;frame.loc[1,'HomePlanet']='Mars'
        frame.loc[0,'Cabin']=None;frame.loc[1,'Cabin']='Z/99/S'
        (fixed,_),masks,_=e.repair_features(frame,iterate)
        self.assertEqual(fixed.loc[0,'HomePlanet'],'Mars');self.assertEqual(fixed.loc[0,'Side'],'S')
        self.assertEqual(fixed.loc[0,'Deck'],'__MISSING__');self.assertTrue(pd.isna(fixed.loc[0,'CabinNumber']))
        self.assertTrue(masks['HomePlanet'][0]);self.assertFalse(masks['HomePlanet'][1])

    def test_labels_rejected_before_feature_builder_and_builder_drift_rejected(self):
        builder=Mock();frame=roster();frame['Transported']=True
        with self.assertRaises(ValueError):e.repair_features(frame,builder)
        builder.build_features.assert_not_called()
        frame=roster();base,cats=iterate.build_features(frame,False);base.loc[0,'Age']=42.
        builder.build_features.return_value=(base,cats)
        with self.assertRaisesRegex(RuntimeError,'Baseline raw13'):e.repair_features(frame,builder)

    def test_no_observed_value_is_ever_replaced_in_conflicting_groups(self):
        frame=roster()
        (fixed,_),_,_=e.repair_features(frame,iterate)
        base,_=iterate.build_features(frame,False)
        for name in ['HomePlanet',*e.SPEND]:
            known=frame[name].notna();pd.testing.assert_series_equal(base.loc[known,name],fixed.loc[known,name])
        known=frame.Cabin.notna();pd.testing.assert_series_equal(base.loc[known,'Side'],fixed.loc[known,'Side'])

    def test_continue_exact_gain_and_descriptive_subsets(self):
        raw=dict(accuracy=.8,log_loss=.4,fold_accuracy=[.8,.8,.8])
        for gain,passed in [(.0009,False),(.001,True)]:
            candidate=dict(accuracy=.8+gain,log_loss=.4,fold_accuracy=[.8,.8,.8])
            self.assertEqual(e.continuation(candidate,raw)['passed'],passed)
        candidate=dict(accuracy=.81,log_loss=.41,fold_accuracy=[.9,.9,.9])
        self.assertFalse(e.continuation(candidate,raw)['passed'])
        report=e.subset_report(np.array([.2,.7]),np.array([.6,.3]),np.array([0,1]),{'any_repair':np.array([True,False]),'Side':np.array([False,False])})
        self.assertEqual(report['any_repair']['repaired'],1);self.assertEqual(report['unrepaired']['repaired'],1)
        self.assertIsNone(report['Side']['candidate_accuracy']);self.assertTrue(report['Side']['descriptive_only'])

    def test_load_boundary_uses_development_reader_and_attribute_only_csv(self):
        frame=roster();baseline=iterate.build_features(frame,False);dev=np.arange(6);y=dev%2
        frozen={'evidence':{}}
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder)
            stack.enter_context(patch.object(e,'ROOT',root));stack.enter_context(patch.object(e,'assert_frozen'))
            stack.enter_context(patch.object(e,'sha',return_value='fixture'))
            readdev=stack.enter_context(patch.object(e.audit,'load_audit_data',return_value=({False:baseline},dev,y,None,None)))
            stack.enter_context(patch.object(e.history,'read_component',return_value=(np.full(6,.5),{},{})))
            path=root/'artifacts/run_038';path.mkdir(parents=True)
            np.savez(path/'v6_no_id_features_seed20261112.npz',prob=np.full(6,.5))
            def attributes(path,**kwargs):
                self.assertIn('usecols',kwargs);self.assertFalse(kwargs['usecols']('Transported'));self.assertTrue(kwargs['usecols']('HomePlanet'))
                return frame.iloc[:6].copy() if Path(path).name=='train.csv' else frame.iloc[6:].copy()
            stack.enter_context(patch.object(e.pd,'read_csv',side_effect=attributes))
            result=e.load_inputs(iterate,frozen)
            readdev.assert_called_once_with(iterate);np.testing.assert_array_equal(result['y'],y)
            self.assertEqual(len(result['splits']),3)

    def test_parent_supervision_timeout_and_no_overwrite(self):
        process=Mock();process.poll.return_value=None
        with self.assertRaises(TimeoutError):e.supervise(process,time.monotonic()-1)
        process.kill.assert_called_once_with();process.wait.assert_called_once_with(timeout=5)
        with tempfile.TemporaryDirectory() as folder,patch.object(e,'OUTPUT',Path(folder)):
            with self.assertRaises(FileExistsError):e.run()

    def test_failed_child_preserves_original_error_and_relabels_outputs(self):
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder);out=root/'artifacts'/e.DIRECTORY;report=root/'reports'/e.DIRECTORY
            for key,val in [('ROOT',root),('OUTPUT',out),('REPORT',report)]:stack.enter_context(patch.object(e,key,val))
            e.write(root/e.PLAN,{'fixture':True})
            stack.enter_context(patch.object(e,'contract',return_value={}))
            process=Mock();process.poll.return_value=0
            def spawn(*args,**kwargs):
                e.write(out/'worker_result.json',dict(status='failed_inconclusive',error_type='RuntimeError',error='first native error'))
                np.savez(out/'oof.npz',partial=np.arange(2));return process
            stack.enter_context(patch.object(e.subprocess,'Popen',side_effect=spawn));stack.enter_context(patch.object(e,'supervise'))
            with self.assertRaisesRegex(RuntimeError,'first native error'):e.run()
            self.assertEqual(e.read(report/'failure.json')['worker_failure']['error'],'first native error')
            self.assertFalse((out/'oof.npz').exists());self.assertTrue((out/'oof.partial.npz').exists())
            self.assertFalse((report/'summary.json').exists())


if __name__=='__main__':unittest.main()
