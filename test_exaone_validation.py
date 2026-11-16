"""CPU-only boundary tests: no checkpoint load, device work, or real scoring."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import exaone_validation as v


def forwards(rows,queries):
    return [dict(support_rows=rows,query_rows=min(1024,queries-start),views=1,input_dtype='float32')
            for start in range(0,queries,1024) for _ in range(8)]


def qualified():
    lock='fixture==1\n';current=dict(sources={'fixture.py':'sha'},environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),
        checkpoint_sha256='weight',checkpoint_bytes=83286800,settings=v.core.SETTINGS,design_sha256='design')
    r=dict(current,status='complete',mode='mps',resource_qualification=True,accuracy_computed=False,
        all_data_synthetic=True,budget_seconds=2400,backend='native_default_device_SDPA_in_locked_PyTorch',stages=[])
    for name,rows,q in [('AB_native_and_local_CPU',128,33),('C_tiny',128,33),('D_screen',4636,2318),('E_full_support',8693,2318),('F_full_export',8693,4277)]:
        stage=dict(stage=name,status='complete',support_rows=rows,query_rows=q,views=8,full_support=True,
            initial_forward_records=forwards(rows,q),same_native_cpu_preprocessing=True,
            model={'native_strict_state_keys_shapes_values':True},original_vs_adapter_max_difference=0.,partition_max_difference=0.,
            cpu_vs_mps_max_difference=0.,repeat_max_difference=0.,
            portable_reload=dict(max_difference=0.,query_rows=q,full_support_rows=rows,same_threshold_decisions=True,
                restored_cpu_context=True,preprocessing_exact=True,forward_records=forwards(rows,q)))
        r['stages'].append(stage)
    return r,current,lock


class ValidationTests(unittest.TestCase):
    def test_resource_requires_complete_all_views_and_reload(self):
        r,c,l=qualified();v.validate_resource(r,c,l)
        for mutation in ['incomplete','views','reload','environment','nan']:
            bad=copy.deepcopy(r)
            if mutation=='incomplete':bad['status']='incomplete'
            elif mutation=='views':bad['stages'][-1]['initial_forward_records'].pop()
            elif mutation=='reload':bad['stages'][-1]['portable_reload']['same_threshold_decisions']=False
            elif mutation=='environment':bad['environment_sha256']='changed'
            else:bad['stages'][1]['repeat_max_difference']=float('nan')
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):v.validate_resource(bad,c,l)

    def test_resource_refusal_precedes_label_access(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(v,'ROOT',Path(folder)),\
                patch.object(v.resource,'check_quota'),patch.object(v,'resource_gate',side_effect=RuntimeError('resource refused')),\
                patch.object(v.data,'load_data') as labels:
            with self.assertRaisesRegex(RuntimeError,'resource refused'):v.run('random',Path('/not/read'))
            labels.assert_not_called()

    def test_group_refusal_precedes_label_access(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(v,'ROOT',Path(folder)),\
                patch.object(v.resource,'check_quota'),patch.object(v,'resource_gate',return_value=({'evidence':{}},'')),\
                patch.object(v,'random_gate',side_effect=RuntimeError('random refused')),patch.object(v.data,'load_data') as labels:
            with self.assertRaisesRegex(RuntimeError,'random refused'):v.run('group',Path('/not/read'))
            labels.assert_not_called()

    def test_exact_saved_group_membership_not_regenerated(self):
        dev=np.arange(6);groups=np.array(['a','a','b','b','c','c']);passengers=np.array(['a_1','a_2','b_1','b_2','c_1','c_2'])
        cache=dict(dev_ids=dev,passenger_ids=passengers,group_ids=groups,fold_id=np.array([0,0,1,1,2,2]))
        inputs=dict(dev=dev,groups=groups,passengers=passengers)
        rows=[dict(train_ids=dev[cache['fold_id']!=k],validation_ids=dev[cache['fold_id']==k]) for k in range(3)]
        self.assertEqual(len(v.data.validate_saved_group(cache,rows,inputs)),3)
        bad=copy.deepcopy(rows);bad[0]['validation_ids']=bad[0]['validation_ids'][::-1]
        with self.assertRaises(RuntimeError):v.data.validate_saved_group(cache,bad,inputs)
        bad=copy.deepcopy(cache);bad['fold_id'][1],bad['fold_id'][2]=1,0
        mixed=[dict(train_ids=dev[bad['fold_id']!=k],validation_ids=dev[bad['fold_id']==k]) for k in range(3)]
        with self.assertRaises(RuntimeError):v.data.validate_saved_group(bad,mixed,inputs)

    def exercise(self,fail=False,final_fail=False):
        n=6;dev=np.arange(n);y=np.array([0,1,0,1,0,1]);splits=[(dev[dev//2!=k],dev[dev//2==k]) for k in range(3)]
        inputs=dict(dev=dev,y=y,features=pd.DataFrame({'feature':dev}),groups=np.array(['a','a','b','b','c','c']),
            passengers=np.array(['a1','a2','b1','b2','c1','c2']))
        made=[]
        class Fake:
            def __init__(self,x,labels,ids,seed,device):
                self.released=False;self.ids=ids;self.seed=seed;self.device=device
                if made and not made[-1].released:raise AssertionError('Previous engine still resident.')
                self.metadata=dict(training_rows=len(ids),training_ids_sha256=v.core.array_hash(ids),labels_sha256=v.core.array_hash(labels.astype(np.int64)),
                    settings=v.core.SETTINGS,preprocessing=dict(support_rows=len(ids),views=8))
                made.append(self)
            def predict_proba(self,q):
                if fail and len(made)==2:raise RuntimeError('injected fold failure')
                p=.1+.8*(q['feature'].to_numpy()%2);result=np.column_stack([1-p,p])
                self.metadata['last_prediction']=dict(probabilities_sha256=v.core.array_hash(result),forward_records=forwards(len(self.ids),len(q)))
                return result
            def release(self):self.released=True
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'artifacts';report=Path(directory)/'reports';out.mkdir();report.mkdir()
            original_write=v.resource.write
            def writing(path,value):
                if final_fail and Path(path).name=='summary.json' and value['status']=='complete':
                    raise RuntimeError('injected finalization failure')
                original_write(path,value)
            with patch.object(v.core,'fit_context',side_effect=Fake),patch.object(v,'assert_frozen'),\
                    patch.object(v.resource,'write',side_effect=writing),\
                    patch.object(v.core,'check_deadline'),patch.object(v.resource,'snapshot',return_value={}):
                if fail or final_fail:
                    with self.assertRaisesRegex(RuntimeError,'injected'):v.evaluate(inputs,splits,{'reference':.1+.8*y},out,report,{},'random')
                    self.assertFalse((out/'screen_results.csv').exists())
                    summary=json.loads((out/'summary.json').read_text());self.assertEqual(summary['status'],'failed_inconclusive')
                    self.assertEqual(summary['completed_folds'],3 if final_fail else 1)
                    self.assertFalse((out/'assessment.json').exists())
                    self.assertFalse((out/f'{v.core.NAME}_seed20261112.npz').exists())
                else:
                    result=v.evaluate(inputs,splits,{'reference':.1+.8*y},out,report,{},'random')
                    self.assertEqual(result['status'],'complete');self.assertTrue(result['continue_to_group'])
                    self.assertEqual(result['metrics']['accuracy'],1.);self.assertFalse(result['S_supported'])
                    with np.load(out/f'{v.core.NAME}_seed20261112.npz') as cache:np.testing.assert_array_equal(cache['dev_ids'],dev)
            self.assertTrue(all(m.released and m.seed==0 and m.device=='mps' for m in made))

    def test_all_fold_seed_zero_and_sequential_release(self):self.exercise()
    def test_partial_failure_no_winner_and_release(self):self.exercise(fail=True)
    def test_finalization_failure_removes_complete_artifacts(self):self.exercise(final_fail=True)

    def test_fixed_thresholds_and_group_anchors(self):
        self.assertTrue(v.assess('random',{'accuracy':.8098159332758125},{})['passed'])
        self.assertFalse(v.assess('random',{'accuracy':.8097},{})['passed'])
        refs={'v71_Mitra2_zeroshot':{'accuracy':.8169398907103825},'v44_seed_ensemble':{'accuracy':.814495254529767}}
        self.assertTrue(v.assess('group',{'accuracy':.81},refs)['passed'])
        self.assertFalse(v.assess('group',{'accuracy':.806},refs)['passed'])


if __name__=='__main__':unittest.main()
