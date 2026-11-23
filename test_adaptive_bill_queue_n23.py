"""Offline verification of sequencing and replay for the fourth frozen export."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import await_half_roster_slot_n23 as queue


class StopPolling(Exception):
    pass


class AdaptiveQueueTest(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        root = Path(stack.enter_context(TemporaryDirectory()))
        stack.enter_context(patch.object(queue, 'ROOT', root))
        self.job = replace(queue.ADAPTIVE, predecessor=queue.HALF.out/'queue_status.json')
        queue.HALF.out.mkdir(parents=True)
        self.job.out.mkdir()
        self.saved = dict(phase='completed', ref='12', score=.817, sha256=self.job.predecessor_digest)
        queue.write(self.job.predecessor, self.saved)
        self.digest = 'a'*64
        self.revision = '1'*40
        self.rows = [SimpleNamespace(ref='12', status='COMPLETE', public_score='.817',
                                     description='previous sha:' + self.job.predecessor_digest[:12])]
        fake_api = SimpleNamespace(competition_submissions=lambda *args, **kwargs: list(self.rows))
        stack.enter_context(patch.object(queue, 'verify', return_value=dict(
            sha256=self.digest, parents={'v131':'baseline'}, version='adaptive')))
        self.api = stack.enter_context(patch.object(queue, 'available', return_value=(fake_api, {'numAllowedNow':1})))
        stack.enter_context(patch.object(queue, 'snapshot', return_value=({'account_best_rank':100}, .838)))
        self.submit = stack.enter_context(patch.object(queue, 'submit'))
        self.sleep = stack.enter_context(patch.object(queue.time, 'sleep', side_effect=StopPolling))

    def test_waits_for_third_and_handles_failure(self):
        queue.write(self.job.predecessor, dict(phase='awaiting_predecessor'))
        with self.assertRaises(StopPolling):
            queue.run(self.revision, self.job)
        self.assertEqual(queue.read(self.job.out/'queue_status.json')['phase'], 'awaiting_predecessor')
        queue.write(self.job.predecessor, dict(phase='evaluation_failed'))
        queue.run(self.revision, self.job)
        self.assertEqual(queue.read(self.job.out/'queue_status.json')['phase'], 'predecessor_failed')
        self.api.assert_not_called()
        self.submit.assert_not_called()

    def test_wrong_third_digest_or_ambiguous_official_record(self):
        with self.assertRaises(ValueError):
            queue.confirm_predecessor(self.job, {**self.saved, 'sha256': self.digest}, self.rows[0])
        self.rows[0].description = 'different export'
        with self.assertRaises(RuntimeError):
            queue.run(self.revision, self.job)
        self.submit.assert_not_called()

    def test_pending_remote_and_repeated_run_never_upload_twice(self):
        with self.assertRaises(StopPolling):
            queue.run(self.revision, self.job)
        self.submit.assert_called_once_with(self.job.name, 'gate.joblib', self.job.notice)
        self.rows.append(SimpleNamespace(ref='13', status='PENDING', public_score='',
                                         description=self.job.name+' adaptive git:x sha:'+self.digest[:12]))
        with self.assertRaises(StopPolling):
            queue.run(self.revision, self.job)
        self.assertEqual(queue.read(self.job.out/'queue_status.json')['phase'], 'awaiting_score')
        self.assertEqual(self.submit.call_count, 1)


if __name__ == '__main__':
    unittest.main()
