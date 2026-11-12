import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import audit_top5_references as audit


class ReferenceAuditTests(unittest.TestCase):
    def test_old_holdout_target_values_are_not_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            # Invalid non-dev targets would fail binary conversion if included.
            train = pd.DataFrame(dict(PassengerId=['0001_01','0002_01','0003_01','0004_01'],
                Age=[10,20,30,40], Transported=['True','NEVER_PARSE','False','NEVER_PARSE']))
            train.to_csv(path/'train.csv', index=False)
            train.drop(columns='Transported').iloc[:1].to_csv(path/'test.csv', index=False)
            attributes, _, targets = audit.read_development_only(path/'train.csv',path/'test.csv',np.array([0,2]))
            self.assertNotIn('Transported', attributes)
            np.testing.assert_array_equal(targets,[1,0])
            self.assertEqual(len(attributes),4)

    def test_group_boundaries_and_full_oof_coverage(self):
        groups = np.repeat(np.arange(30).astype(str),2)
        y = np.arange(60)%2
        for kind, seed in audit.SPLITS:
            folds = audit.audit_splits(kind,seed,y,groups)
            np.testing.assert_array_equal(np.sort(np.concatenate([b for _,b in folds])),np.arange(60))
            for a,b in folds:
                self.assertFalse(set(a)&set(b))
                if kind=='group':
                    self.assertFalse(set(groups[a])&set(groups[b]))
            again = audit.audit_splits(kind,seed,y,groups)
            for first,second in zip(folds,again):
                np.testing.assert_array_equal(first[1],second[1])

    def test_foreign_monkeypatch_import_is_rejected(self):
        audit.assert_fresh_model_process({'numpy':object()})
        for name in ['seed_ensemble','mitra_candidate','spatial_expert','autogluon_candidate']:
            with self.assertRaisesRegex(RuntimeError,'fresh process'):
                audit.assert_fresh_model_process({name:object()})

    def test_history_mutation_is_detected_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            source=path/'immutable.json'
            source.write_text('original')
            before=audit.directory_hashes(path)
            audit.ensure_history_unchanged(path,before)
            source.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'Historical run changed'):
                audit.ensure_history_unchanged(path,before)
            self.assertEqual(source.read_text(),'changed')

    def test_mitra_validates_history_then_redirects_logs_before_fit(self):
        events=[]
        torch=SimpleNamespace(set_num_threads=lambda n:events.append(('threads',n)),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:True)),
            mps=SimpleNamespace(set_per_process_memory_fraction=lambda x:events.append(('fraction',x))))
        wrapper=SimpleNamespace(CHECKPOINT_DIR=Path('unused-test-path'),
            configure_runtime=lambda args:events.append(('configure',args.stage,args.run)),
            RUN_OUT=Path('artifacts/run_038'),DEADLINE=1,torch=torch,
            SETTINGS={'mps_memory_fraction':.65},FRAMEWORK=object(),fit=object())
        output=Path('artifacts/top5_reference_audit/mitra')
        with patch.object(audit.importlib,'import_module',return_value=wrapper), \
                patch.object(audit,'check_historical_sources'), \
                patch.object(audit,'assert_fresh_model_process'):
            _,got,_,_,_=audit.initialize_reference('mitra',output)
        self.assertIs(got,wrapper)
        self.assertEqual(events[0],('configure','confirm','run_038'))
        self.assertEqual(wrapper.RUN_OUT,output)
        self.assertIsNone(wrapper.DEADLINE)


if __name__=='__main__':
    unittest.main()
