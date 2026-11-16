from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import mitra_missing_flags_candidate as candidate


class CandidateTests(unittest.TestCase):
    def test_fixed_continuation_gate_boundaries_and_no_automatic_authorization(self):
        raw=dict(accuracy=.81,log_loss=.4,fold_accuracy=[.81,.81,.81])
        passing=dict(accuracy=.811,log_loss=.4,fold_accuracy=[.81,.81,.813])
        gate=candidate.continue_gate(passing,raw,[0,1,2])
        self.assertTrue(gate['passed']);self.assertFalse(gate['automatic_fullfit_or_upload'])
        for changed in [dict(accuracy=.810999),dict(log_loss=.400001),dict(fold_accuracy=[.80,.80,.84]),dict(accuracy=np.nan)]:
            self.assertFalse(candidate.continue_gate({**passing,**changed},raw,[0,1,2])['passed'])
        self.assertFalse(candidate.continue_gate(passing,raw,[0,1])['passed'])
        self.assertTrue(candidate.continue_gate({**passing,'accuracy':.811-1e-13},raw,[0,1,2])['passed'])

    def test_resource_evidence_requires_both_shapes_complete_exact_sources_and_reload(self):
        contract=dict(sources={'fixture.py':'sha'},environment_sha256='env')
        case=lambda rows:dict(status='complete',support_rows=rows,query_rows=2318,actual_support_rows=rows,
            actual_retained_features=20,subset_passed=True,reload_passed=True,cpu_mps_input_exact=True,
            schema_reload_exact=True,full_support=True,subset_max_difference=0.,reload_max_difference=0.)
        good=dict(status='complete',device='mps',settings=candidate.core.SETTINGS,sources=contract['sources'],
            environment_sha256='env',checkpoint_sha256=candidate.base.WEIGHT_SHA,budget_seconds=1200,
            no_accuracy_computed=True,cases=[case(4636),case(8693)])
        with tempfile.TemporaryDirectory() as folder,patch.object(candidate,'ROOT',Path(folder)), \
                patch.object(candidate.common,'require_committed'):
            root=Path(folder)
            for altered in [None,dict(status='incomplete'),dict(cases=[case(4636)]),dict(sources={}),
                    dict(cases=[case(4636),{**case(8693),'reload_max_difference':.001}]),
                    dict(cases=[case(4636),{**case(8693),'actual_retained_features':19}])]:
                record=good if altered is None else {**good,**altered}
                for name in [candidate.PROBE,candidate.PROBE_REPORT]:candidate.write(root/name,record)
                if altered is None:self.assertEqual(len(candidate.require_probe(contract)),2)
                else:
                    with self.assertRaises(RuntimeError):candidate.require_probe(contract)

    def test_wrong_dev_ids_are_rejected_before_any_target_read(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(candidate,'ROOT',Path(folder)):
            root=Path(folder)
            candidate.write(root/'reports/official_data_verification.json',dict(competition='spaceship-titanic',files={
                name:dict(official_sha256=hashlib.sha256(b'fixture').hexdigest()) for name in ['train.csv','test.csv']}))
            (root/'data/raw').mkdir(parents=True)
            for name in ['train.csv','test.csv']:(root/'data/raw'/name).write_text('fixture')
            (root/'artifacts/run_002').mkdir(parents=True)
            np.savez(root/'artifacts/run_002/split_indices.npz',development=np.arange(10))
            with patch.object(candidate.audit,'read_development_only') as read:
                with self.assertRaisesRegex(RuntimeError,'before reading labels'):candidate.load_inputs()
                read.assert_not_called()

    def test_source_and_environment_mutation_fail_freeze(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(candidate,'ROOT',Path(folder)), \
                patch.object(candidate.base,'environment_lock',return_value='env'),patch.object(candidate.base,'verify_checkpoint'):
            root=Path(folder);(root/'source.py').write_text('a')
            contract=dict(sources={'source.py':candidate.sha(root/'source.py')},evidence_files={},
                environment_sha256=hashlib.sha256(b'env').hexdigest(),settings=candidate.core.SETTINGS)
            candidate.assert_frozen(contract)
            (root/'source.py').write_text('b')
            with self.assertRaisesRegex(RuntimeError,'source/evidence'):candidate.assert_frozen(contract)

    def inputs(self):
        x=candidate.base.synthetic_frame(24);dev=np.arange(24);y=dev%2
        splits=[(np.delete(dev,np.arange(fold*8,(fold+1)*8)),np.arange(fold*8,(fold+1)*8)) for fold in range(3)]
        return dict(x=x,dev=dev,y=y,splits=splits,raw=np.full(24,.4),raw_record={},references={},
            passenger_ids=np.asarray([str(v) for v in dev]),evidence={})

    def test_screen_probe_precedes_labels_and_partial_failure_never_emits_winner(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
                root=Path(folder);stack.enter_context(patch.object(candidate,'ROOT',root))
                contract=dict(sources={},evidence_files={},environment_sha256='fixture',settings=candidate.core.SETTINGS)
                events=[];data=self.inputs()
                stack.enter_context(patch.object(candidate,'check_quota'))
                stack.enter_context(patch.object(candidate,'freeze_contract',return_value=(contract,'fixture')))
                stack.enter_context(patch.object(candidate,'require_probe',side_effect=lambda _:events.append('probe') or {}))
                stack.enter_context(patch.object(candidate,'load_inputs',side_effect=lambda:events.append('labels') or data))
                stack.enter_context(patch.object(candidate.common,'require_committed'))
                stack.enter_context(patch.object(candidate,'assert_frozen'));stack.enter_context(patch.object(candidate,'runtime'))
                stack.enter_context(patch.object(candidate.base,'release_memory'));stack.enter_context(patch('builtins.print'))
                def fit(values,fold,out):
                    if fail and fold==1:raise RuntimeError('second fold')
                    b=values['splits'][fold][1]
                    return np.where(values['y'][b],.8,.2),dict(schema={'fixture':True},seconds=0.)
                stack.enter_context(patch.object(candidate,'fit_fold',side_effect=fit))
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'second fold'):candidate.screen('private-session')
                else:
                    result=candidate.screen('private-session');self.assertEqual(result['status'],'complete')
                    self.assertEqual(result['missing_subset']['rows']+result['complete_subset']['rows'],24)
                self.assertEqual(events,['probe','labels'])
                for directory in [root/'artifacts'/candidate.RUN,root/'reports'/candidate.RUN]:
                    self.assertNotIn('private-session',(directory/'protocol.json').read_text())
                    progress=json.loads((directory/'progress.json').read_text())
                    self.assertEqual(progress['status'],'failed_inconclusive' if fail else 'complete')
                    for name in ['assessment.json','selection.json','screen_results.csv']:
                        self.assertEqual((directory/name).exists(),not fail)
                self.assertEqual((root/'artifacts'/candidate.RUN/'partial_oof.npz').exists(),fail)
                with self.assertRaisesRegex(RuntimeError,'no overwrite'):candidate.screen('private-session')


if __name__=='__main__':unittest.main()
