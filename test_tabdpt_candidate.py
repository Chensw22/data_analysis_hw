import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tabdpt_candidate as candidate


class ResourceGateTest(unittest.TestCase):
    def test_complete_matching_large_cases_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            freeze=dict(settings=candidate.backend.SETTINGS,checkpoint_sha256=candidate.backend.WEIGHT_SHA,
                        sources={},environment_sha256=hashlib.sha256(b'env').hexdigest())
            (path/'freeze.json').write_text(json.dumps(freeze))
            (path/'environment.lock.txt').write_text('env')
            result=dict(status='complete',device='mps',synthetic_only=True,accuracy_computed=False,
                freeze_sha256=candidate.backend.sha(path/'freeze.json'),cases=[])
            for support,query,ensembles in [(128,17,8),(4636,2318,1),(8693,4277,1)]:
                result['cases'].append(dict(support_rows=support,query_rows=query,ensembles=ensembles,
                    actual_forward_support_rows=[support],full_reload_rows=query,
                    reload_threshold_identical=True,reload_max_abs_difference=0.,batching_max_abs_difference=0.))
            with patch.object(candidate,'PROBE',path), patch.object(candidate,'environment_lock',return_value='env'), \
                    patch.object(candidate.backend,'source_hashes',return_value={}), \
                    patch.object(candidate.backend,'verify_checkpoint'),patch.object(candidate.adapter,'verify_sources'):
                (path/'result.json').write_text(json.dumps(result))
                self.assertEqual(candidate.verify_resource(),freeze)
                variants=[]
                bad=copy.deepcopy(result);bad['status']='failed_inconclusive';variants.append(bad)
                bad=copy.deepcopy(result);bad['cases'].pop();variants.append(bad)
                bad=copy.deepcopy(result);bad['cases'][-1]['actual_forward_support_rows']=[1024];variants.append(bad)
                bad=copy.deepcopy(result);bad['cases'][-1]['batching_max_abs_difference']=1e-3;variants.append(bad)
                bad=copy.deepcopy(result);bad['cases'][-1]['reload_threshold_identical']=False;variants.append(bad)
                for bad in variants:
                    with self.subTest(bad=bad):
                        (path/'result.json').write_text(json.dumps(bad))
                        with self.assertRaises(RuntimeError):candidate.verify_resource()

    def test_fit_isolates_training_labels_and_disjoint_queries(self):
        class Fake:
            metadata={}
            def predict_proba(self,x):
                self.query=x.index.to_list()
                return np.tile([.4,.6],(len(x),1))
            def release(self):pass
        with tempfile.TemporaryDirectory() as directory:
            fake=Fake();frame=pd.DataFrame({'value':range(8)})
            with patch.object(candidate,'RUN_OUT',Path(directory)),patch.object(candidate,'DEADLINE',None), \
                    patch.object(candidate.adapter,'fit_context',return_value=fake) as fit:
                p,_=candidate.fit(candidate.NAME,{False:(frame,[])},[1,3,5,7],np.array([0,1,0,1]),[0,2],42)
                self.assertEqual(fit.call_args.args[0].index.to_list(),[1,3,5,7])
                np.testing.assert_array_equal(fit.call_args.args[1],[0,1,0,1])
                self.assertEqual(fake.query,[0,2]);np.testing.assert_array_equal(p,[.6,.6])
                with self.assertRaises(ValueError):
                    candidate.fit(candidate.NAME,{False:(frame,[])},[1,3],[0,1],[3,5],42)

    def test_screen_partition_guard_precedes_evaluation(self):
        with patch.object(candidate,'record_protocol') as record,patch.object(candidate,'BASE_EVALUATE') as evaluate:
            with self.assertRaises(ValueError):
                candidate.evaluate(candidate.NAME,None,None,np.arange(8693),20261112,Path('.'))
            record.assert_not_called();evaluate.assert_not_called()


if __name__=='__main__':unittest.main()
