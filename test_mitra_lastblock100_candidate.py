import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import mitra_lastblock100_candidate as candidate


class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        candidate.configure()

    def tearDown(self):
        candidate.original.mitra.DEADLINE = None

    def test_only_learning_change_is_max_steps_and_old_sources_unchanged(self):
        ignored = {'budgets','screen_budget_seconds','stability_budget_seconds','export_budget_seconds'}
        changes = {k for k in candidate.SETTINGS
                   if k not in ignored and candidate.SETTINGS[k] != candidate.ORIGINAL_SETTINGS[k]}
        self.assertEqual(changes, {'max_steps'})
        self.assertEqual(candidate.ORIGINAL_SETTINGS['max_steps'],50)
        self.assertEqual(candidate.SETTINGS['max_steps'],100)
        self.assertEqual(candidate.original.NAME,candidate.NAME)
        self.assertIs(candidate.mitra,candidate.original.mitra)
        self.assertIn(candidate.NAME,candidate.original.iterate.CONFIGS)
        self.assertNotIn('v75_Mitra_lastblock50',candidate.original.iterate.CONFIGS)
        basis = candidate.verify_basis()  # Real frozen run041 source hashes.
        self.assertFalse(basis['resume_previous_fit'])
        self.assertIn('never automatically extend to 200',basis['stopping_rule'])

    def test_formal_fit_passes100_and_cap_is_only_an_annotation(self):
        original=candidate.original
        x=original.mitra.synthetic_frame(24)
        calls=[]
        class Fake:
            metadata={'selection_at_cap':True,'selected_steps':100}
            def predict_proba(self, frame):return np.tile([.3,.7],(len(frame),1))
            def release(self):pass
        def learner(*args, **kwargs):
            calls.append(args)
            return Fake()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(candidate,'_ORIGINAL_FIT_LASTBLOCK',side_effect=learner), \
             patch.object(original.torch.backends.mps,'is_available',return_value=True), \
             patch.object(original.torch.mps,'set_per_process_memory_fraction'):
            candidate.configure_runtime(directory)
            p, model=candidate.fit(candidate.NAME,{False:(x,[])},np.arange(20),
                np.arange(20)%2,np.arange(20,24),17)
            self.assertEqual(calls[0][-1],100)
            np.testing.assert_array_equal(p,np.full(4,.7))
            self.assertEqual(model.metadata['convergence_assessment'],'inconclusive_at_cap')
            self.assertFalse(model.metadata['automatic_further_extension_allowed'])
            saved=json.loads(next(Path(directory).glob('*.json')).read_text())
            self.assertEqual(saved['selected_steps'],100)
            self.assertEqual(saved['convergence_assessment'],'inconclusive_at_cap')

    def test_original_stepzero_and_fresh_refit_cpu_invariants_still_hold(self):
        # Execute the existing real tiny-Tab2D invariants through the configured
        # new wrapper. Their two-step override keeps CPU tests small; formal fit
        # propagation of100 is independently asserted above.
        from test_mitra_lastblock_candidate import CandidateTests
        suite=unittest.TestSuite(CandidateTests(name) for name in [
            'test_inner_isolation_selected_count_and_fresh_refit',
            'test_step_zero_is_exact_original_zero_shot_and_earliest_tie'])
        result=unittest.TestResult()
        suite.run(result)
        self.assertTrue(result.wasSuccessful(),str(result.errors+result.failures))

    def test_fresh_process_freeze_and_second_process_assert_without_training(self):
        script = '''
import json, sys
from pathlib import Path
import mitra_lastblock100_candidate as c
c.configure()
p=Path(sys.argv[1])
if sys.argv[2]=='freeze':
    p.mkdir()
    (p/'protocol.json').write_text(json.dumps(dict(sources={},configs=c.original.iterate.CONFIGS,blend=[])))
    c.record_protocol(p)
c.assert_frozen(p)
d=json.loads((p/'protocol.json').read_text())
assert 'mitra_lastblock100_candidate.py' in d['sources']
assert 'test_mitra_lastblock100_candidate.py' in d['sources']
assert 'reports/top5_lastblock50_review.json' in d['sources']
assert d['lastblock_policy']['settings']['max_steps']==100
assert 'steps0..100' in d['lastblock_policy']['selection']
assert d['bounded_lastblock_extension']['only_learning_change']==dict(max_steps_before=50,max_steps_after=100)
print('freeze_assert_ok')
'''
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'run'
            for mode in ['freeze','assert']:
                completed=subprocess.run([sys.executable,'-c',script,str(out),mode],
                    cwd=candidate.ROOT,capture_output=True,text=True,timeout=60)
                self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
                self.assertIn('freeze_assert_ok',completed.stdout)

    def test_stability_cli_uses_generic_validator_without_editing_it(self):
        import top5_validate
        before=candidate.original.mitra.sha256(Path(top5_validate.__file__))
        with patch.object(sys,'argv',['wrapper','stability','--run','run_042']), \
             patch.object(top5_validate,'validate') as validate:
            candidate.main()
        validate.assert_called_once_with('mitra_lastblock100_candidate','run_042')
        self.assertEqual(before,candidate.original.mitra.sha256(Path(top5_validate.__file__)))


if __name__ == '__main__':
    unittest.main()
