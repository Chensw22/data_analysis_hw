import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import mitra_context_candidate as candidate


TINY=dict(dim=8,dim_output=10,n_layers=1,n_heads=2,task='CLASSIFICATION')


def roster_fixture(n=72):
    rng=np.random.default_rng(81)
    raw=pd.DataFrame({col:rng.lognormal(2,1,n) for col in
        ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']})
    groups=np.repeat(np.arange(n),np.tile([1,2,3,4],(n+3)//4)[:n])[:n]
    positions=pd.Series(groups).groupby(groups).cumcount().to_numpy()+1
    raw['PassengerId']=[f'{g:04d}_{p:02d}' for g,p in zip(groups,positions)]
    raw['Name']=[f'Person{i} Surname{i%17}' for i in range(n)]
    raw['HomePlanet']=np.array(['Earth','Europa','Mars'])[groups%3]
    raw['Destination']=np.array(['A','B','C'])[groups%3]
    raw['Cabin']=[f'{"ABC"[g%3]}/{g}/{"PS"[g%2]}' for g in groups]
    raw['Age']=rng.integers(0,70,n).astype(float)
    raw['VIP']=np.arange(n)%7==0
    raw['CryoSleep']=np.arange(n)%3==0
    for col in ['Age','HomePlanet','Destination','Cabin','CryoSleep']:
        if col!='Age':
            raw[col]=raw[col].astype(object)
        raw.loc[3,col]=np.nan
    return raw[candidate.ATTRIBUTES]


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidate.mitra.torch.set_num_threads(1)

    def test_targets_ignored_and_fixed_schema(self):
        raw=roster_fixture()
        original=candidate.build_roster(raw)
        for target in [np.arange(len(raw))%2,np.zeros(len(raw)),np.ones(len(raw))]:
            changed=candidate.build_roster(raw.assign(Transported=target))
            pd.testing.assert_frame_equal(changed['context'],original['context'])
            self.assertEqual(changed['attribute_sha256'],original['attribute_sha256'])
        self.assertEqual(list(original['context']),candidate.COLUMNS)
        self.assertEqual(len(original['context'].columns),20)
        self.assertNotIn('PassengerId',original['context'])
        self.assertNotIn('Name',original['context'])
        self.assertEqual(original['plain'].loc[3,'HomePlanet'],'__MISSING__')
        self.assertNotEqual(original['context'].loc[3,'HomePlanet'],'__MISSING__')
        pd.testing.assert_frame_equal(original['plain'],candidate.iterate.build_features(raw,False)[0])

    def test_train_only_twenty_column_preprocessing_unknown_missing_and_serialization(self):
        x=candidate.mitra.synthetic_frame(40)
        rng=np.random.default_rng(7)
        for col in candidate.EXTRA:
            x[col]=rng.uniform(0,10,len(x))
        prep=candidate.ContextPreprocessor().fit(x,np.arange(len(x))%2,seed=19)
        self.assertEqual(prep.columns,candidate.COLUMNS)
        self.assertEqual(prep.n_features,20)
        self.assertEqual(prep.transform(x).shape,(40,20))
        means=prep.upstream.pre_nan_mean.copy()
        vocab=copy.deepcopy(prep.vocab)
        query=x.iloc[:3].copy()
        query['HomePlanet']=['new_planet','__MISSING__',None]
        query['PeerMeanAge']=[np.nan,1e9,1e10]
        encoded=prep.encode(query)
        np.testing.assert_array_equal(encoded[:,0],[0,1,1])
        self.assertTrue(np.isfinite(prep.transform(query)).all())
        np.testing.assert_array_equal(prep.upstream.pre_nan_mean,means)
        self.assertEqual(prep.vocab,vocab)
        changed=x.copy()
        changed['PeerMeanAge']+=100
        self.assertFalse(np.array_equal(prep.transform(x),prep.transform(changed)))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'preprocessor.joblib'
            joblib.dump(prep,path)
            restored=joblib.load(path)
            np.testing.assert_array_equal(restored.transform(query),prep.transform(query))
        self.assertEqual(candidate.mitra.COMPACT,candidate.COLUMNS[:13])

    def test_persisted_manifest_permutation_subset_and_tiny_cpu_model_reload(self):
        roster=candidate.build_roster(roster_fixture())
        a=np.arange(48)
        labels=a%2
        prep=candidate.ContextPreprocessor().fit(roster['context'].iloc[a],labels,42)
        model=candidate.mitra.make_model(TINY)
        predictor=candidate.ContextPredictor(prep,TINY,model.state_dict(),
            prep.transform(roster['context'].iloc[a]),labels,device='cpu',roster=roster,query_chunk=2318)
        query=roster['plain'].iloc[[54,52,60]]
        expected=predictor.predict_proba(query)
        np.testing.assert_allclose(predictor.predict_proba_by_passenger_ids(roster['passenger_ids'][[54,52,60]]),expected,atol=1e-6)
        np.testing.assert_allclose(predictor.predict_proba(query.iloc[[2,0]]),expected[[2,0]],atol=1e-6)
        with self.assertRaises(AssertionError):
            predictor.predict_proba(query.reset_index(drop=True))
        changed=query.copy()
        changed.iloc[0,changed.columns.get_loc('Age')]=9999
        with self.assertRaises(AssertionError):
            predictor.predict_proba(changed)
        with self.assertRaises(ValueError):
            predictor.predict_proba_by_passenger_ids(['unknown'])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'tiny.joblib'
            joblib.dump(predictor,path)
            restored=joblib.load(path)
            self.assertEqual(restored.device,'cpu')
            self.assertEqual(restored.preprocessor.columns,candidate.COLUMNS)
            np.testing.assert_allclose(restored.predict_proba(query),expected,atol=1e-6,rtol=0)

    def test_control_uses_exact_original_frame_and_disjoint_ids(self):
        roster=candidate.build_roster(roster_fixture())
        received=[]
        class Fake:
            metadata={}
            def predict_proba(self,x):
                received.append(x.copy())
                return np.tile([.6,.4],(len(x),1))
            def release(self):
                pass
        original=candidate.RUN_OUT
        try:
            with tempfile.TemporaryDirectory() as folder, \
                 patch.object(candidate,'load_roster',return_value=roster), \
                 patch.object(candidate.mitra,'fit_mitra',return_value=Fake()) as fit, \
                 patch.object(candidate.mitra,'release_memory'), \
                 patch.object(candidate.mitra,'check_deadline'), \
                 patch.object(candidate.mitra,'snapshot',return_value={'cpu_test':True}):
                candidate.RUN_OUT=Path(folder)
                matrices={False:(roster['plain'],[])}
                p,_=candidate.fit(candidate.CONTROL,matrices,[0,1,2,3],[0,1,0,1],[4,5],7)
                pd.testing.assert_frame_equal(fit.call_args.args[0],roster['plain'].iloc[[0,1,2,3]])
                pd.testing.assert_frame_equal(received[0],roster['plain'].iloc[[4,5]])
                self.assertEqual(fit.call_args.args[4:6],('mps','zero-shot'))
                self.assertEqual(fit.call_args.kwargs['query_chunk'],2318)
                np.testing.assert_array_equal(p,[.4,.4])
                with self.assertRaisesRegex(ValueError,'disjoint'):
                    candidate.fit(candidate.NAME,matrices,[0,1],[0,1],[1,2],7)
                wrong={False:(roster['plain'].iloc[::-1],[])}
                with self.assertRaises(AssertionError):
                    candidate.fit(candidate.NAME,wrong,[0,1],[0,1],[2,3],7)
        finally:
            candidate.RUN_OUT=original

    def test_roster_loaded_once_with_target_excluding_allowlist(self):
        raw=roster_fixture(12970)
        original=candidate.ROSTER_CACHE
        candidate.ROSTER_CACHE=None
        try:
            with patch.object(candidate.pd,'read_csv',side_effect=[raw.iloc[:8693],raw.iloc[8693:]]) as read:
                first=candidate.load_roster()
                self.assertIs(candidate.load_roster(),first)
                self.assertEqual(read.call_count,2)
                for call in read.call_args_list:
                    self.assertEqual(call.kwargs['usecols'],candidate.ATTRIBUTES)
                    self.assertNotIn('Transported',call.kwargs['usecols'])
        finally:
            candidate.ROSTER_CACHE=original

    def test_resource_gate_requires_both_full_shape_complete_matched_probes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            probes={key:root/key for key in ['outer','final']}
            lock='synthetic-package==1\n'
            lockhash=candidate.hashlib.sha256(lock.encode()).hexdigest()
            for shape,n in [('outer',4636),('final',8693)]:
                candidate.write_json(probes[shape]/'result.json',dict(status='complete',support_rows=n,
                    query_rows=2318,actual_retained_features=20,full_support=True,source_hashes={'file':'sha'},
                    settings=candidate.SETTINGS,environment_sha256=lockhash,roster_attribute_sha256='attributes',
                    reload_passed=True,subset_passed=True))
            with patch.object(candidate,'ROOT',root),patch.object(candidate,'PROBES',probes), \
                 patch.object(candidate,'source_hashes',return_value={'file':'sha'}), \
                 patch.object(candidate.mitra,'environment_lock',return_value=lock), \
                 patch.object(candidate,'load_roster',return_value={'attribute_sha256':'attributes'}):
                self.assertEqual(set(candidate.require_resources()),{'outer','final'})
                path=probes['final']/'result.json'
                complete=json.loads(path.read_text())
                for field,value in [('status','failed_inconclusive'),('support_rows',8192),
                        ('actual_retained_features',13),('source_hashes',{}),('reload_passed',False)]:
                    candidate.write_json(path,{**complete,field:value})
                    with self.subTest(field=field),self.assertRaises(RuntimeError):
                        candidate.require_resources()

    def test_screen_freeze_failure_marks_partial_without_touching_existing_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            out=root/'artifacts/run_fixture'
            def screen(path):
                if path.exists():
                    raise RuntimeError('Existing run')
                path.mkdir(parents=True)
                (path/'screen_results.csv').write_text('version,accuracy\n')
                candidate.write_json(path/'selection.json',{'stage':'unverified'})
            with patch.object(candidate,'ROOT',root), \
                 patch.object(candidate.sys,'argv',['context','screen','--run','run_fixture']), \
                 patch.object(candidate,'configure_runtime'),patch.object(candidate,'require_resources'), \
                 patch.object(candidate.iterate,'screen',side_effect=screen), \
                 patch.object(candidate,'assert_frozen',side_effect=RuntimeError('Freeze changed')), \
                 patch.object(candidate.signal,'signal'),patch.object(candidate.signal,'setitimer'):
                with self.assertRaisesRegex(RuntimeError,'Freeze changed'):
                    candidate.main()
                self.assertFalse((out/'selection.json').exists())
                self.assertFalse((out/'screen_results.csv').exists())
                self.assertTrue((out/'screen_results.partial.csv').exists())
                self.assertEqual(json.loads((out/'failure.json').read_text())['status'],'failed_inconclusive')
                (out/'selection.json').write_text('historical')
                with self.assertRaisesRegex(RuntimeError,'Existing run'):
                    candidate.main()
                self.assertEqual((out/'selection.json').read_text(),'historical')


if __name__=='__main__':
    unittest.main()
