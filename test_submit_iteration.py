import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import submit_iteration as upload
from submit_iteration import count_attempts


class AccountingTest(unittest.TestCase):
    def test_failed_and_unresolved_attempts_consume_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, batch, states in [('a', 'new', ['submitted', 'attempt_started']),
                                        ('b', 'new', ['needs_manual_reconciliation']),
                                        ('old', 'old', ['submitted'])]:
                path = root/name; (path/'online_attempts').mkdir(parents=True)
                (path/'authorization.json').write_text(json.dumps({'batch_id': batch}))
                for index, state in enumerate(states):
                    (path/'online_attempts'/f'{index}.json').write_text(json.dumps({'state': state}))
            self.assertEqual(count_attempts(root, 'new'), 3)
            self.assertEqual(count_attempts(root, 'absent'), 0)

    def test_corrupt_authorization_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'candidate'; path.mkdir()
            (path/'authorization.json').write_text('{broken')
            with self.assertRaises(json.JSONDecodeError):
                count_attempts(Path(directory), 'new')

    def test_preupload_refusals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(path, data):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            auth = root/'reports'/upload.BATCH/'authorization.json'
            write(auth, dict(batch_id=upload.BATCH, maximum_upload_attempts=10,
                             weekly_reserve_percent=30))
            source = root/'source.py'; source.write_text('frozen')
            evidence = root/'evidence.json'; write(evidence, {'done': True})
            decision_path = root/'decision.json'
            decision = dict(proposal='candidate', version='v74',
                            evidence_files={'evidence.json': upload.workflow.sha256(evidence)})
            write(decision_path, decision)
            out = root/'artifacts/candidate'; out.mkdir(parents=True)
            (out/'model.joblib').write_bytes(b'model')
            (out/'submission.csv').write_text('csv')
            bound = dict(batch_id=upload.BATCH, proposal='candidate', version='v74',
                source_authorization_sha256=upload.workflow.sha256(auth),
                maximum_upload_attempts_for_this_file=1, file_sha256='csvhash',
                model_sha256=upload.workflow.sha256(out/'model.joblib'),
                decision_sha256=upload.workflow.sha256(decision_path))
            manifest = dict(status='complete', export_reload_inference_verified=True,
                sha256='csvhash', provenance=dict(source_hashes={'source.py': upload.workflow.sha256(source)},
                decision_path='decision.json', decision_sha256=bound['decision_sha256'],
                evidence_files=decision['evidence_files']))
            def sync():
                for folder in [out, root/'reports/candidate']:
                    for name, data in [('authorization.json',bound), ('manifest.json',manifest),
                                       ('selection.json', {'version':'v74'})]:
                        write(folder/name, data)
            sync()
            with patch.object(upload, 'ROOT', root), patch.object(upload, 'AUTH', auth), \
                    patch.object(upload.workflow, 'validate_submission', return_value={'sha256':'csvhash'}), \
                    patch.object(upload.subprocess, 'check_output', return_value='') as git:
                self.assertEqual(upload.verify('candidate')[1], 0)
                with patch.object(upload, 'count_attempts', return_value=10):
                    with self.assertRaisesRegex(RuntimeError, 'ten'):
                        upload.verify('candidate')
                attempt = root/'reports/previous/online_attempts/one.json'
                write(attempt, {'file_sha256':'csvhash'})
                with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                    upload.verify('candidate')
                attempt.unlink()
                bound['file_sha256']='changed'; sync()
                with self.assertRaisesRegex(RuntimeError, 'CSV changed'):
                    upload.verify('candidate')
                bound['file_sha256']='csvhash'; sync()
                git.return_value=' M code.py'
                with self.assertRaisesRegex(RuntimeError, 'Commit changes'):
                    upload.verify('candidate')
                git.return_value=''
                source.write_text('different committed source')
                with self.assertRaisesRegex(RuntimeError, 'source changed'):
                    upload.verify('candidate')


if __name__ == '__main__':
    unittest.main()
