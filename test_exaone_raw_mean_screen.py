"""Synthetic cached probabilities only; never read candidate datasets or load models."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import exaone_raw_mean_screen as m


def fixture():
    dev=np.arange(6);y=np.array([0,1,0,1,0,1]);splits=[(dev[dev//2!=k],dev[dev//2==k]) for k in range(3)]
    inputs=dict(dev=dev,y=y,passengers=np.array(['a1','a2','b1','b2','c1','c2'],dtype=object),groups=np.array(['a','a','b','b','c','c']))
    raw=.2+.6*y;exaone=.1+.8*y
    component=dict(raw=raw,exaone=exaone,splits=splits,raw_metrics=m.data.metrics(raw,y,splits),exaone_metrics=m.data.metrics(exaone,y,splits))
    return inputs,{'random':copy.deepcopy(component),'group':copy.deepcopy(component)}


class MeanScreenTests(unittest.TestCase):
    def test_only_equal_probability_mean(self):
        np.testing.assert_array_equal(m.fixed_mean([.1,.9],[.3,.7]),[.2,.8])
        for a,b in [([.1],[.2,.3]),([float('nan')],[.2]),([1.1],[.2]),([],[])]:
            with self.assertRaises(ValueError):m.fixed_mean(a,b)

    def test_gate_requires_pooled_accuracy_loss_and_two_folds(self):
        raw=dict(accuracy=.8,log_loss=.4,fold_accuracy=[.8,.8,.8])
        self.assertTrue(m.gate(dict(raw),raw)['passed'])
        for changes in [dict(accuracy=.799),dict(log_loss=.401),dict(fold_accuracy=[.81,.79,.79])]:
            with self.subTest(changes=changes):self.assertFalse(m.gate(dict(raw,**changes),raw)['passed'])
        self.assertTrue(m.gate(dict(raw,fold_accuracy=[.8,.8,.79]),raw)['passed'])

    def test_wrong_dev_fold_or_passenger_ids_refused(self):
        inputs,phases=fixture();part=phases['random']
        cache=dict(dev_ids=inputs['dev'],fold_id=m.expected_fold_ids(inputs['dev'],part['splits']),
            passenger_ids=inputs['passengers'],group_ids=inputs['groups'],prob=part['exaone'],fold_accuracy=np.ones(3))
        m.aligned_cache(cache,inputs,part['splits'])
        for key in ['dev_ids','fold_id','passenger_ids','group_ids']:
            bad=copy.deepcopy(cache);bad[key]=bad[key][::-1]
            with self.subTest(key=key),self.assertRaises(RuntimeError):m.aligned_cache(bad,inputs,part['splits'])
        with self.assertRaises(RuntimeError):m.expected_fold_ids(inputs['dev'],part['splits'][:2])

    def test_uncommitted_refused_before_labels_or_mean(self):
        with patch.object(m.validation.resource,'check_quota'),\
                patch.object(m.data.common,'require_committed',side_effect=RuntimeError('uncommitted')),\
                patch.object(m,'prepare_inputs') as inputs,patch.object(m,'fixed_mean') as mean:
            with self.assertRaisesRegex(RuntimeError,'uncommitted'):m.run(Path('/not/read'))
            inputs.assert_not_called();mean.assert_not_called()

    def exercise(self,failure=None):
        inputs,phases=fixture();original_evaluate=m.evaluate_phase;original_write=m.validation.resource.write;count=0
        def evaluate(i,c):
            nonlocal count
            count+=1
            if failure=='second_phase' and count==2:raise RuntimeError('injected incomplete group')
            return original_evaluate(i,c)
        def write(path,value):
            if failure=='finalization' and 'exaone_mean_group_20261117' in str(path) and Path(path).name=='summary.json' and value.get('status')=='complete':
                raise RuntimeError('injected publish failure')
            return original_write(path,value)
        with tempfile.TemporaryDirectory() as directory,patch.object(m,'ROOT',Path(directory)),\
                patch.object(m,'assert_own'),patch.object(m.validation,'assert_frozen'),\
                patch.object(m,'evaluate_phase',side_effect=evaluate),patch.object(m.validation.resource,'write',side_effect=write),\
                patch.object(m.core,'fit_context',side_effect=AssertionError('Models forbidden')):
            if failure:
                with self.assertRaisesRegex(RuntimeError,'injected'):m.execute(inputs,phases,{},'fixture==1\n',{},'test_commit')
                for run in m.RUNS.values():
                    for kind in ['artifacts','reports']:
                        folder=Path(directory)/kind/run
                        for file in ['screen_results.csv','group_metrics.csv','assessment.json',f'{m.NAME}_seed20261112.npz']:
                            self.assertFalse((folder/file).exists())
                        summary=json.loads((folder/'summary.json').read_text());self.assertEqual(summary['status'],'failed_inconclusive')
            else:
                result=m.execute(inputs,phases,{},'fixture==1\n',{},'test_commit')
                self.assertEqual(set(result),{'random','group'})
                for phase,run in m.RUNS.items():
                    self.assertTrue(result[phase]['continue_to_root_decision']);self.assertFalse(result[phase]['S_supported'])
                    folder=Path(directory)/'artifacts'/run
                    with np.load(folder/f'{m.NAME}_seed20261112.npz',allow_pickle=False) as cache:
                        np.testing.assert_array_equal(cache['passenger_ids'],inputs['passengers'])
                        self.assertEqual(cache['passenger_ids'].dtype.kind,'U')
                with self.assertRaisesRegex(RuntimeError,'overwrite'):m.execute(inputs,phases,{},'',{},'test_commit')

    def test_complete_two_phases_and_unicode_cache(self):self.exercise()
    def test_partial_pair_cannot_publish_for_tiers(self):self.exercise('second_phase')
    def test_finalization_failure_cleans_both_phase_results(self):self.exercise('finalization')


if __name__=='__main__':unittest.main()
