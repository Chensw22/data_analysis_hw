"""Small mocked boundary checks; no native model/official labels are read."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import prepare_iltm_online_first as p


class PreparationTests(unittest.TestCase):
    def test_identity_failure_precedes_full_targets_and_leaves_no_csv(self):
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            out=Path(folder)/'artifact';out.mkdir();report=Path(folder)/'report';report.mkdir()
            p.write(out/'protocol.json',{});(out/'submission.csv').write_text('partial')
            stack.enter_context(patch.object(p,'OUT',out));stack.enter_context(patch.object(p,'REPORT',report))
            stack.enter_context(patch.dict(p.os.environ,{p.bridge.FLAG:'FALSE'}))
            stack.enter_context(patch.object(p,'assert_frozen',side_effect=RuntimeError('source changed')))
            load=stack.enter_context(patch.object(p,'load_full_inputs'))
            record=p.worker(p.time.monotonic()+30)
            load.assert_not_called();self.assertEqual(record['error'],'source changed')
            self.assertFalse((out/'submission.csv').exists());self.assertEqual(record['status'],'failed_inconclusive')

    def test_parent_snapshots_first_preserves_failure_and_no_retry(self):
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            out=Path(folder)/'artifact';report=Path(folder)/'report';order=[]
            stack.enter_context(patch.object(p,'OUT',out));stack.enter_context(patch.object(p,'REPORT',report))
            stack.enter_context(patch.dict(p.os.environ,{},clear=True))
            stack.enter_context(patch.object(p.bridge,'entry_environment',side_effect=lambda:order.append('env') or {p.bridge.FLAG:'FALSE'}))
            stack.enter_context(patch.object(p,'contract',side_effect=lambda:order.append('identity') or {}))
            process=Mock();process.poll.return_value=0
            def launch(*args,**kwargs):
                order.append('child');self.assertEqual(kwargs['env'][p.bridge.FLAG],'FALSE')
                self.assertEqual(p.os.environ[p.bridge.FLAG],'FALSE')
                p.write(out/'worker_result.json',{'status':'failed_inconclusive','error_type':'MemoryError','error':'first native failure'})
                (out/'submission.csv').write_text('partial');return process
            stack.enter_context(patch.object(p.subprocess,'Popen',side_effect=launch));stack.enter_context(patch.object(p.full,'supervise',return_value=1))
            result=p.run();self.assertEqual(order,['env','identity','child']);self.assertIn('first native failure',result['error'])
            self.assertFalse((out/'submission.csv').exists());self.assertFalse((out/'manifest.json').exists())
            self.assertFalse((report/'selection.json').exists())
            with self.assertRaises(FileExistsError):p.run()

    def test_full_saved_prediction_and_csv_reverification(self):
        import numpy as np
        import pandas as pd
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder);out=root/'artifact';out.mkdir();(root/'data/raw').mkdir(parents=True)
            stack.enter_context(patch.object(p,'ROOT',root));stack.enter_context(patch.object(p,'OUT',out))
            stack.enter_context(patch.object(p.full,'validate_native_completion'))
            ids=np.array([f'{i:04}_01' for i in range(4277)])
            matrix=np.tile([.7,.3],(4277,1));np.savez(out/'predictions.npz',first=matrix,reloaded=matrix,passenger_ids=ids)
            pd.DataFrame({'PassengerId':ids}).to_csv(root/'data/raw/test.csv',index=False)
            pd.DataFrame({'PassengerId':ids,'Transported':False}).to_csv(out/'submission.csv',index=False)
            (out/'model.joblib').write_bytes(b'mocked portable model')
            record=dict(status='worker_complete',release_complete=True,training_rows=8693,rows=4277,
                export_reload_inference_verified=True,model_sha256=p.sha(out/'model.joblib'),sha256=p.sha(out/'submission.csv'),
                prediction_cache_sha256=p.sha(out/'predictions.npz'),reload_max_absolute_difference=0.)
            p.verify_completed(record)
            wrong=pd.read_csv(out/'submission.csv');wrong.loc[0,'Transported']=True;wrong.to_csv(out/'submission.csv',index=False)
            record['sha256']=p.sha(out/'submission.csv')
            with self.assertRaisesRegex(RuntimeError,'boolean submission'):p.verify_completed(record)


if __name__=='__main__':unittest.main()
