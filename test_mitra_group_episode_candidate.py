"""Metadata-only fixed-cache, review and execution-boundary checks."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import mitra_group_episode_candidate as wrapper


class GroupWrapperTests(unittest.TestCase):
    def cache(self):
        dev=np.arange(6,dtype=np.int64)*2;groups=np.repeat(['a','b','c'],2)
        passengers=np.array([f'{g}_{i}' for i,g in enumerate(groups)])
        fold=np.repeat(np.arange(3,dtype=np.int8),2)
        cached=dict(dev_ids=dev,group_ids=groups,passenger_ids=passengers,fold_id=fold,prob=np.linspace(.1,.9,6))
        rows=[dict(train_ids=dev[fold!=i],validation_ids=dev[fold==i]) for i in range(3)]
        return cached,rows,dev,groups,passengers

    def test_cached_exact_ids_coverage_and_group_boundary(self):
        cached,rows,dev,groups,passengers=self.cache()
        probability,assignment,folds=wrapper.validate_cached_arrays(cached,rows,dev,groups,passengers)
        np.testing.assert_array_equal(probability,cached['prob'])
        np.testing.assert_array_equal(np.sort(np.concatenate([b for _,b in folds])),np.arange(6))
        bad=copy.deepcopy(rows);bad[0]['train_ids']=bad[0]['train_ids'][::-1]
        with self.assertRaisesRegex(RuntimeError,'fold IDs'):wrapper.validate_cached_arrays(cached,bad,dev,groups,passengers)
        for key in ('dev_ids','passenger_ids','group_ids'):
            bad=copy.deepcopy(cached);bad[key]=bad[key][::-1]
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                wrapper.validate_cached_arrays(bad,rows,dev,groups,passengers)
        bad=copy.deepcopy(cached);bad['prob'][0]=np.nan
        with self.assertRaises(RuntimeError):wrapper.validate_cached_arrays(bad,rows,dev,groups,passengers)
        crossed=groups.copy();crossed[2]='a';bad=copy.deepcopy(cached);bad['group_ids']=crossed
        with self.assertRaisesRegex(RuntimeError,'group boundary'):
            wrapper.validate_cached_arrays(bad,rows,dev,crossed,passengers)

    def test_review_matches_fixed_recipe_and_explicit_resource_extrapolation(self):
        value=dict(candidate=wrapper.NAME,recipe=wrapper.core.SETTINGS,screen=wrapper.SCREEN_SETTINGS,
            resource_reuse=wrapper.RESOURCE_REUSE,weekly_reserve_percent=30,
            automatic_export=False,adaptive_reused_development=True)
        with tempfile.TemporaryDirectory() as folder,patch.object(wrapper,'REVIEW',Path(folder)/'review.json'):
            wrapper.write(wrapper.REVIEW,value);self.assertEqual(wrapper.review(),value)
            for key,altered in [('weekly_reserve_percent',12),('automatic_export',True),('resource_reuse',{}),('candidate','v81')]:
                bad=copy.deepcopy(value);bad[key]=altered;wrapper.write(wrapper.REVIEW,bad)
                with self.subTest(key=key),self.assertRaises(RuntimeError):wrapper.review()

    def test_existing_output_blocks_before_data_loading(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(wrapper,'OUT',Path(folder)), \
                patch.object(wrapper,'load_inputs') as data,patch.object(wrapper,'review') as review:
            with self.assertRaisesRegex(RuntimeError,'no resume'):wrapper.run()
            data.assert_not_called();review.assert_not_called()

    def test_uncommitted_or_modified_sources_are_rejected(self):
        with patch.object(wrapper.subprocess,'check_output',return_value=b''):
            with self.assertRaisesRegex(RuntimeError,'committed'):wrapper.require_committed({'new.py':'sha'})
        with patch.object(wrapper.subprocess,'check_output',side_effect=[b'new.py\0',b' M new.py']):
            with self.assertRaisesRegex(RuntimeError,'committed versions'):wrapper.require_committed({'new.py':'sha'})

    def test_frozen_source_or_environment_change_rejected(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(wrapper,'ROOT',Path(folder)), \
                patch.object(wrapper,'review'),patch.object(wrapper.base,'verify_checkpoint'), \
                patch.object(wrapper.base,'environment_lock',return_value='env'):
            source=Path(folder)/'model.py';source.write_text('first')
            record=dict(sources={'model.py':wrapper.audit.file_hash(source)},historical_files={},
                environment_sha256=wrapper.hashlib.sha256(b'env').hexdigest())
            wrapper.assert_frozen(record)
            source.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'Frozen evidence'):wrapper.assert_frozen(record)


    def test_official_raw_hash_check_precedes_any_label_parser(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(wrapper,'ROOT',Path(folder)):
            root=Path(folder);(root/'data/raw').mkdir(parents=True);(root/'reports').mkdir()
            files={}
            for name in ['train.csv','test.csv']:
                path=root/'data/raw'/name;path.write_text('synthetic-only\n')
                files[name]={'official_sha256':wrapper.audit.file_hash(path)}
            wrapper.write(root/'reports/official_data_verification.json',dict(competition='spaceship-titanic',files=files))
            self.assertEqual(set(wrapper.verify_data_files()),{'data/raw/train.csv','data/raw/test.csv'})
            (root/'data/raw/train.csv').write_text('changed\n')
            with patch.object(wrapper.audit,'load_audit_data') as data:
                with self.assertRaisesRegex(RuntimeError,'Official raw data SHA'):
                    wrapper.load_inputs(None)
                data.assert_not_called()

    def test_complete_progress_and_failed_partial_without_whole_score(self):
        cached,rows,dev,groups,passengers=self.cache()
        _,assignment,folds=wrapper.validate_cached_arrays(cached,rows,dev,groups,passengers)
        y=np.arange(6)%2;raw=np.tile([.4,.6],3)
        matrix={False:(pd.DataFrame({'feature':np.arange(12)}),[])}
        inputs=(matrix,dev,y,groups,passengers,raw,assignment,folds,[],{})
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
                output=Path(folder)/'out';report=Path(folder)/'report';calls=[]
                class Fake:
                    metadata={'selected_steps':0}
                    def predict_proba(self,x):return np.array([[.6,.4],[.4,.6]])
                    def release(self):pass
                def train(*args):
                    index=len(calls);calls.append(index)
                    wrapper.write(args[-1],dict(status='failed_inconclusive' if fail and index==1 else 'fit_complete'))
                    if fail and index==1:raise RuntimeError('injected fold failure')
                    return Fake()
                for name,value in [('OUT',output),('REPORT',report)]:stack.enter_context(patch.object(wrapper,name,value))
                stack.enter_context(patch.object(wrapper,'review'))
                stack.enter_context(patch.object(wrapper,'verify_reference_basis',return_value=({},)))
                stack.enter_context(patch.object(wrapper,'verify_resource',return_value={}))
                stack.enter_context(patch.object(wrapper,'source_and_history',return_value=({},{})))
                stack.enter_context(patch.object(wrapper,'require_committed',return_value='test-commit'))
                stack.enter_context(patch.object(wrapper,'load_inputs',return_value=inputs))
                stack.enter_context(patch.object(wrapper,'assert_frozen'))
                stack.enter_context(patch.object(wrapper.base,'environment_lock',return_value='test-env'))
                stack.enter_context(patch.object(wrapper.base,'release_memory'))
                stack.enter_context(patch.object(wrapper.base.torch.backends.mps,'is_available',return_value=True))
                stack.enter_context(patch.object(wrapper.base.torch.mps,'set_per_process_memory_fraction'))
                stack.enter_context(patch.object(wrapper.base.torch,'set_num_threads'))
                stack.enter_context(patch.object(wrapper.core,'fit_group_episode',side_effect=train))
                stack.enter_context(patch.object(wrapper.signal,'signal'))
                stack.enter_context(patch.object(wrapper.signal,'setitimer'))
                stack.enter_context(patch('builtins.print'))
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'injected fold failure'):wrapper.run()
                    self.assertFalse((report/'summary.json').exists())
                    self.assertFalse((report/'group_metrics.csv').exists())
                    progress=json.loads((report/'progress.json').read_text())
                    self.assertEqual(progress['status'],'failed_inconclusive')
                    self.assertFalse(progress['whole_score_available'])
                    self.assertEqual(len(progress['completed_folds']),1)
                    self.assertTrue((output/'partial_oof.npz').exists())
                    self.assertTrue((report/'fold1_training.json').exists())
                else:
                    result=wrapper.run();self.assertEqual(result['status'],'complete')
                    progress=json.loads((report/'progress.json').read_text())
                    self.assertEqual(progress['status'],'complete')
                    self.assertTrue(progress['whole_score_available'])
                    self.assertEqual(len(progress['completed_folds']),3)
                    self.assertTrue((report/'fold2_training.json').exists())
                    self.assertFalse((output/'partial_oof.npz').exists())


if __name__=='__main__':unittest.main()
