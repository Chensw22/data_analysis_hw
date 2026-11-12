"""Submission boundary tests; never access Kaggle or consume submission quota."""
from pathlib import Path
import tempfile
import unittest
from kaggle_workflow import validate_submission


class SubmissionValidationTests(unittest.TestCase):
    def check_csv(self, body):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'test.csv').write_text('PassengerId\n0001_01\n0002_01\n')
            (root/'submission.csv').write_text(body)
            return validate_submission(root/'submission.csv', root/'test.csv', expected_rows=2)

    def test_valid_booleans(self):
        self.assertTrue(self.check_csv('PassengerId,Transported\n0001_01,True\n0002_01,False\n')['valid'])

    def test_reject_wrong_id_order(self):
        with self.assertRaises(ValueError):
            self.check_csv('PassengerId,Transported\n0002_01,True\n0001_01,False\n')

    def test_reject_probabilities_and_missing_labels(self):
        for label in ['0.8', '', 'NaN', '1']:
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.check_csv(f'PassengerId,Transported\n0001_01,{label}\n0002_01,False\n')

    def test_reject_wrong_schema(self):
        with self.assertRaises(ValueError):
            self.check_csv('PassengerId,prediction\n0001_01,True\n0002_01,False\n')


if __name__ == '__main__':
    unittest.main()
