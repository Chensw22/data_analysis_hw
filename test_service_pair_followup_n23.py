import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import await_service_pair_followup_n23 as queue


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(TemporaryDirectory()))
        for attr, path in [('ROOT', root), ('WORK', root/'work'), ('OUT', root/'first'),
                           ('REPORT', root/'reports/first')]:
            self.stack.enter_context(patch.object(queue, attr, path))
        self.stack.enter_context(patch.object(queue, 'verify', lambda n: {'sha256': queue.FILES[n][1]}))
        put(queue.OUT/'queue_status.json', {'phase': 'completed'})
        put(queue.ROOT/'reports/current_best_submission.json', {'public_score': .81692, 'leaderboard_snapshot': {}})

    def first(self, value, rank):
        put(queue.REPORT/'queued_result.json', {'ref':'x', 'score':value, 'rank':{'account_best_rank':rank}})
        put(queue.REPORT/'online_attempts/x.json', {'file_sha256':queue.DIGEST, 'submission_ref':'x',
            'evaluation':{'Status':'COMPLETE', 'Public Score':str(value)}})

    def check_branch(self, score, expected):
        self.first(score, 110)
        calls = []
        api = SimpleNamespace(competition_leaderboard_view=lambda *a, **k: [SimpleNamespace(score=.84)]*3)
        self.stack.enter_context(patch.object(queue, 'available', lambda: (api, {'numAllowedNow':1})))
        def submit(name, model, notice):
            calls.append(name)
            put(queue.ROOT/'reports'/name/'online_attempts/new.json', {'file_sha256':queue.FILES[name][1]})
        self.stack.enter_context(patch.object(queue, 'submit', submit))
        def record(name, manifest, receipt, original, third):
            self.assertEqual((name, original['score'], third), (expected, score, .84))
            queue.state('completed')
        self.stack.enter_context(patch.object(queue, 'record', record))
        queue.run()
        queue.state('recording')
        queue.run()
        queue.run()
        self.assertEqual(calls, [expected])

    def test_linear_if_below_best(self):
        self.check_branch(.81691, queue.LINEAR)

    def test_union_if_tied(self):
        self.check_branch(.81692, queue.UNION)

    def test_uncertain_attempt_and_reached_goal(self):
        self.stack.enter_context(patch.object(queue, 'available', side_effect=AssertionError('unexpected Kaggle call')))
        self.first(.816, 110)
        put(queue.WORK/'followup_status.json', {'phase':'submitting'})
        with self.assertRaisesRegex(RuntimeError, 'previous upload'):
            queue.run()
        self.first(.851, 3)
        queue.run()
        self.assertEqual(queue.read(queue.WORK/'followup_status.json')['phase'], 'goal_reached')
