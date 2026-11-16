"""CPU/mock checks for the fixed-seed design, activation and failure boundaries."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,train_test_split

import mitra_seed_repeat_candidate as candidate


class SeedRepeatTests(unittest.TestCase):
    def setUp(self):
        self.previous_deadline=candidate.base.DEADLINE;candidate.base.DEADLINE=None
    def tearDown(self):candidate.base.DEADLINE=self.previous_deadline

    def inputs(self,n=24):
        dev=np.arange(n);y=dev%2
        splits=list(StratifiedKFold(3,shuffle=True,random_state=candidate.OUTER_SEED).split(dev,y))
        return dict(dev=dev,y=y,splits=splits,features=pd.DataFrame({'Age':dev}),
            old=np.where(y,.7,.3),old_record={'version':candidate.OLD},old_logs=[],evidence={})

    def test_outer_seed_remains_original_and_cached_alignment_is_strict(self):
        dev=np.arange(6954);y=dev%2
        with patch.object(candidate.audit,'DEV_HASH',candidate.audit.row_hash(dev)):
            actual=candidate.fixed_splits(dev,y)
        expected=list(StratifiedKFold(3,shuffle=True,random_state=20261112).split(dev,y))
        wrong=list(StratifiedKFold(3,shuffle=True,random_state=20261212).split(dev,y))
        for (a,b),(aa,bb) in zip(actual,expected):
            np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)
        self.assertFalse(np.array_equal(actual[0][1],wrong[0][1]))
        old=np.where(y,.8,.2)
        candidate.common.cached_fold_scores(old,np.ones(3),y,actual)
        with self.assertRaises(RuntimeError):candidate.common.cached_fold_scores(np.roll(old,1),np.ones(3),y,actual)
        bad=copy.deepcopy(actual);bad[1]=bad[0]
        with self.assertRaises(RuntimeError):candidate.make_fold_assignment(dev,bad)

    def test_fit_changes_only_training_seed_not_outer_membership(self):
        inputs=self.inputs();fold=1;a,b=inputs['splits'][fold];seed=20261213
        ids=inputs['dev'][a];labels=inputs['y'][a]
        inner,valid=train_test_split(np.arange(len(a)),test_size=.2,stratify=labels,random_state=seed)
        step=lambda i:dict(step=i,validation_log_loss=1. if i==0 else .9,finite_gradients=True,disjoint_ids=True)
        metadata=dict(status='fit_complete',seed=seed,training_rows=len(a),training_ids_sha256=candidate.audit.row_hash(ids),
            full_prediction_support=True,fresh_refit_from_original_checkpoint=True,
            inner_train_ids_sha256=candidate.audit.row_hash(ids[inner]),inner_validation_ids_sha256=candidate.audit.row_hash(ids[valid]),
            inner_history=[step(i) for i in range(51)],refit_history=[step(1)],selected_steps=1,
            inner_optimizer_steps=50,refit_optimizer_steps=1,recipe=candidate.core.SETTINGS)
        class Fake:
            def __init__(self):self.metadata=copy.deepcopy(metadata);self.released=False
            def predict_proba(self,x):
                np.testing.assert_array_equal(x.index,inputs['dev'][b])
                return np.tile([.4,.6],(len(x),1))
            def release(self):self.released=True
        fake=Fake()
        with tempfile.TemporaryDirectory() as folder,patch.object(candidate.core,'fit_full_episode',return_value=fake) as fit, \
                patch.object(candidate.base,'release_memory'):
            p,row=candidate.fit_fold(inputs,fold,Path(folder))
            args,kwargs=fit.call_args
            np.testing.assert_array_equal(args[0].index,ids);np.testing.assert_array_equal(args[1],labels)
            np.testing.assert_array_equal(args[2],ids);self.assertEqual(args[3],seed)
            self.assertEqual(kwargs['max_steps'],50);self.assertEqual(kwargs['device'],'mps')
            self.assertEqual(row['outer_split_seed'],20261112);self.assertEqual(row['actual_training_seed'],seed)
            self.assertTrue(fake.released);self.assertEqual(p.shape,(len(b),))
            with self.assertRaisesRegex(RuntimeError,'no resume'):candidate.fit_fold(inputs,fold,Path(folder))

    def test_exact_mean_resource_gate_never_selects_solo_or_relaxes_one_failed_condition(self):
        old=dict(accuracy=.82,log_loss=.38,fold_accuracy=[.81,.82,.83])
        equal=copy.deepcopy(old)
        self.assertTrue(candidate.continuation(equal,old,[0,1,2])['passed'])
        for change in [dict(accuracy=.82-1e-10),dict(log_loss=.38+1e-10),dict(fold_accuracy=[.80,.81,.84])]:
            result=candidate.continuation({**equal,**change},old,[0,1,2]);self.assertFalse(result['passed'])
            self.assertEqual(result['sole_primary_candidate'],candidate.NAME)
            self.assertFalse(result['diagnostic_solo_upload_allowed']);self.assertFalse(result['automatic_fullfit_or_upload'])
        self.assertFalse(candidate.continuation(equal,old,[0,1])['passed'])
        self.assertTrue(candidate.continuation({**equal,'accuracy':.82-1e-13,'log_loss':.38+1e-13},old,[0,1,2])['passed'])
        np.testing.assert_allclose(candidate.cached.fixed_mean(np.array([.2,.8]),np.array([.6,.4])),[.4,.6],rtol=0,atol=2e-16)

    def activation_fixture(self,root):
        summary=dict(status='complete',candidate=candidate.V85_NAME,kind='group',folds=[dict(fold=i) for i in range(3)])
        decision_path='reports/v85_final_decision.json';rank_path='reports/rank_at_activation.json'
        candidate.write(root/candidate.V85_SUMMARY,summary)
        candidate.write(root/decision_path,dict(candidate=candidate.V85_NAME,status='archived_no_upload',
            v85_process_complete=True,submission_attempted=False,completed_at_utc='2026-11-16T11:59:00Z',rationale='Completed review elects not to upload.'))
        candidate.write(root/rank_path,dict(account_best_rank=112,team_count=2146,rank_fraction=112/2146,observed_at_utc='2026-11-16T12:00:00Z'))
        paths=[candidate.V85_SUMMARY,decision_path,rank_path]
        activation=dict(status='activate_fixed_fallback',run=candidate.RUN,primary_candidate=candidate.NAME,
            diagnostic_solo=candidate.SOLO,review_sha256=candidate.REVIEW_SHA,v85_process_complete=True,
            top5_goal_met=False,mps_exclusive_confirmed=True,rationale='Root activates the already fixed fallback.',
            v85_online_disposition='not_submitted',v85_disposition_decision_path=decision_path,
            ranking_evidence_path=rank_path,evidence_files={p:candidate.sha(root/p) for p in paths})
        candidate.write(root/candidate.ACTIVATION,activation)
        return activation,decision_path,rank_path

    def test_activation_requires_completed_v85_concrete_disposition_and_unmet_goal(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(candidate,'ROOT',Path(folder)), \
                patch.object(candidate.common,'require_committed',return_value='commit'), \
                patch.object(candidate,'utc_now',return_value=candidate.aware_timestamp('2026-11-16T12:01:00Z')):
            root=Path(folder);activation,decision_path,rank_path=self.activation_fixture(root)
            self.assertEqual(candidate.activation_evidence()[0],activation)
            for change in [dict(v85_process_complete=False),dict(top5_goal_met=True),dict(mps_exclusive_confirmed=False),dict(v85_disposition_decision_path='missing')]:
                candidate.write(root/candidate.ACTIVATION,{**activation,**change})
                with self.assertRaises(RuntimeError):candidate.activation_evidence()
            pending=json.loads((root/decision_path).read_text());pending['status']='pending_online'
            candidate.write(root/decision_path,pending);activation['evidence_files'][decision_path]=candidate.sha(root/decision_path)
            candidate.write(root/candidate.ACTIVATION,activation)
            with self.assertRaisesRegex(RuntimeError,'Pending'):candidate.activation_evidence()
            activation,decision_path,rank_path=self.activation_fixture(root)
            candidate.write(root/rank_path,dict(account_best_rank=90,teams=2146,rank_fraction=90/2146,observed_at_utc='2026-11-16T12:00:00Z'))
            activation['evidence_files'][rank_path]=candidate.sha(root/rank_path);candidate.write(root/candidate.ACTIVATION,activation)
            with self.assertRaisesRegex(RuntimeError,'top5 goal'):candidate.activation_evidence()

    def test_online_activation_requires_actual_complete_attempt_and_blocks_v85_top5(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(candidate,'ROOT',Path(folder)), \
                patch.object(candidate.common,'require_committed',return_value='commit'), \
                patch.object(candidate,'utc_now',return_value=candidate.aware_timestamp('2026-11-16T12:01:00Z')):
            root=Path(folder);activation,_,rank_path=self.activation_fixture(root)
            attempt_path=candidate.V85_ATTEMPTS+'/attempt.json'
            attempt=dict(version=candidate.V85_NAME,evaluation={'Status':'PENDING'},last_checked_at_utc='2026-11-16T11:59:00Z',leaderboard_snapshot={'rank_fraction':.06})
            candidate.write(root/attempt_path,attempt);activation['v85_online_disposition']='online_complete'
            activation['evidence_files'][attempt_path]=candidate.sha(root/attempt_path);candidate.write(root/candidate.ACTIVATION,activation)
            with self.assertRaisesRegex(RuntimeError,'COMPLETE'):candidate.activation_evidence()
            attempt['evaluation']['Status']='COMPLETE';candidate.write(root/attempt_path,attempt)
            activation['evidence_files'][attempt_path]=candidate.sha(root/attempt_path);candidate.write(root/candidate.ACTIVATION,activation)
            candidate.activation_evidence()
            attempt['leaderboard_snapshot']['rank_fraction']=.04;candidate.write(root/attempt_path,attempt)
            activation['evidence_files'][attempt_path]=candidate.sha(root/attempt_path);candidate.write(root/candidate.ACTIVATION,activation)
            with self.assertRaisesRegex(RuntimeError,'already reached'):candidate.activation_evidence()

    def test_activation_ranking_is_fresh_aware_and_after_final_disposition(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(candidate,'ROOT',Path(folder)), \
                patch.object(candidate.common,'require_committed',return_value='commit'), \
                patch.object(candidate,'utc_now',return_value=candidate.aware_timestamp('2026-11-16T12:01:00Z')):
            root=Path(folder)
            for observed,expected in [
                    ('2026-11-16T11:45:59Z','900seconds'),
                    ('2026-11-16T12:02:01Z','900seconds'),
                    ('2026-11-16T11:58:59Z','predates'),
                    ('2026-11-16T12:00:00','timezone-aware'),
                    ('invalid','timezone-aware')]:
                with self.subTest(observed=observed):
                    activation,decision_path,rank_path=self.activation_fixture(root)
                    rank=json.loads((root/rank_path).read_text());rank['observed_at_utc']=observed
                    candidate.write(root/rank_path,rank)
                    activation['evidence_files'][rank_path]=candidate.sha(root/rank_path)
                    candidate.write(root/candidate.ACTIVATION,activation)
                    with self.assertRaisesRegex(RuntimeError,expected):candidate.activation_evidence()
            activation,decision_path,rank_path=self.activation_fixture(root)
            decision=json.loads((root/decision_path).read_text());decision.pop('completed_at_utc')
            candidate.write(root/decision_path,decision)
            activation['evidence_files'][decision_path]=candidate.sha(root/decision_path)
            candidate.write(root/candidate.ACTIVATION,activation)
            with self.assertRaisesRegex(RuntimeError,'timezone-aware'):candidate.activation_evidence()
            activation,_,_=self.activation_fixture(root)
            activation['v85_online_disposition']='online_complete'
            for index,checked in enumerate(['2026-11-16T11:59:00Z','2026-11-16T12:00:01Z']):
                path=candidate.V85_ATTEMPTS+f'/attempt{index}.json'
                record=dict(version=candidate.V85_NAME,evaluation={'Status':'COMPLETE'},
                    last_checked_at_utc=checked,leaderboard_snapshot={'rank_fraction':.06})
                candidate.write(root/path,record);activation['evidence_files'][path]=candidate.sha(root/path)
            candidate.write(root/candidate.ACTIVATION,activation)
            with self.assertRaisesRegex(RuntimeError,'predates'):candidate.activation_evidence()
            record.pop('last_checked_at_utc');candidate.write(root/path,record)
            activation['evidence_files'][path]=candidate.sha(root/path)
            candidate.write(root/candidate.ACTIVATION,activation)
            with self.assertRaisesRegex(RuntimeError,'timezone-aware'):candidate.activation_evidence()

    def test_partial_failure_has_no_mean_winner_and_success_reports_all_components(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
                root=Path(folder);stack.enter_context(patch.object(candidate,'ROOT',root))
                for name in candidate.OWN_SOURCES:
                    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# fixture\n')
                inputs=self.inputs()
                review=dict(provenance_sha256={},fixed_design=dict(scoring_disclosure='adaptive',randomness_disclosure='whole program'),
                    fixed_continue_to_fullfit_heuristic={'fixed':'fixture'})
                stack.enter_context(patch.object(candidate,'verify_review',return_value=review))
                stack.enter_context(patch.object(candidate,'activation_evidence',return_value=({},{})))
                quota=stack.enter_context(patch.object(candidate,'check_quota'))
                stack.enter_context(patch.object(candidate,'verify_runtime',return_value=({'sources':{}},'fixture-env')))
                stack.enter_context(patch.object(candidate,'load_cached_inputs',return_value=inputs))
                stack.enter_context(patch.object(candidate.common,'require_committed',return_value='commit'))
                stack.enter_context(patch.object(candidate,'assert_frozen'))
                stack.enter_context(patch.object(candidate.base.torch.backends.mps,'is_available',return_value=True))
                stack.enter_context(patch.object(candidate.base.torch.mps,'set_per_process_memory_fraction'))
                stack.enter_context(patch.object(candidate.base.torch,'set_num_threads'))
                stack.enter_context(patch.object(candidate.base,'release_memory'))
                stack.enter_context(patch.object(candidate.signal,'signal'))
                stack.enter_context(patch.object(candidate.signal,'setitimer'))
                stack.enter_context(patch('builtins.print'))
                def fit(values,fold,out):
                    path=out/f'{candidate.SOLO}_seed{candidate.TRAINING_SEEDS[fold]}_fixture.json'
                    candidate.write(path,{'status':'failed' if fail and fold==1 else 'fit_complete'})
                    if fail and fold==1:raise RuntimeError('injected secondfold failure')
                    b=values['splits'][fold][1]
                    return np.where(values['y'][b],.8,.2),dict(fold=fold,outer_split_seed=20261112,
                        actual_training_seed=candidate.TRAINING_SEEDS[fold],selected_steps=1)
                stack.enter_context(patch.object(candidate,'fit_fold',side_effect=fit))
                session=Path('/private/never-write-this-session.jsonl')
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'secondfold'):candidate.screen(session)
                else:
                    result=candidate.screen(session)
                    self.assertEqual(set(result['metrics']),{candidate.OLD,candidate.SOLO,candidate.NAME})
                    self.assertEqual(result['sole_possible_future_export'],candidate.NAME)
                    self.assertFalse(result['solo_upload_allowed'])
                quota.assert_called_once_with(session)
                out=root/'artifacts'/candidate.RUN;report=root/'reports'/candidate.RUN
                for directory in [out,report]:
                    progress=json.loads((directory/'progress.json').read_text())
                    if fail:
                        self.assertEqual(progress['status'],'failed_inconclusive')
                        self.assertFalse(progress['pooled_mean_available'])
                        for name in ['screen_results.csv','assessment.json','selection.json']:
                            self.assertFalse((directory/name).exists())
                    else:
                        self.assertEqual(progress['status'],'complete')
                        self.assertTrue(progress['pooled_mean_available'])
                        self.assertEqual(json.loads((directory/'selection.json').read_text())['version'],candidate.NAME)
                        self.assertEqual(len(pd.read_csv(directory/'screen_results.csv')),3)
                    content=(directory/'protocol.json').read_text()
                    self.assertNotIn(str(session),content);self.assertNotIn('used_percent',content)
                self.assertEqual((out/'partial_solo_oof.npz').exists(),fail)
                with patch.object(candidate,'verify_review') as verify:
                    with self.assertRaisesRegex(RuntimeError,'no overwrite'):candidate.screen(session)
                    verify.assert_not_called()


if __name__=='__main__':unittest.main()
