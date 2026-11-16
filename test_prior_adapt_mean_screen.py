import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import train_test_split

import prior_adapt_mean_screen as screen


class PriorAdaptTests(unittest.TestCase):
    def test_uncommitted_plan_or_sources_stop_before_labels_or_mean(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(screen,'ROOT',Path(directory)), \
             patch.object(screen.common,'require_committed',side_effect=RuntimeError('uncommitted')), \
             patch.object(screen.audit,'read_development_only') as read, \
             patch.object(screen,'fixed_mean') as mean:
            with self.assertRaisesRegex(RuntimeError,'uncommitted'): screen.screen()
            read.assert_not_called();mean.assert_not_called()

    def test_fixed_half_mean_no_weight_or_threshold_search_interface(self):
        np.testing.assert_allclose(screen.fixed_mean([.2,.9],[.8,.3]),[.5,.6])
        with self.assertRaises(TypeError): screen.fixed_mean([.2],[.8],weights=[.2,.8])
        for raw,adapt in [([],[]),([.2],[.8,.3]),([np.nan],[.2]),([-.1],[.5])]:
            with self.assertRaises(ValueError): screen.fixed_mean(raw,adapt)

    def test_adaptation_uses_fit_complete_and_exact_folds_and_selected_steps(self):
        train,query,y=np.arange(12),np.array([20,21,22]),np.tile([0,1],6)
        inner,valid=train_test_split(np.arange(12),test_size=.2,stratify=y,random_state=screen.SEED)
        history=[dict(step=i,validation_log_loss=float(abs(i-2)+1),finite_gradients=True,disjoint_ids=True)
                 for i in range(51)]
        log=dict(status='fit_complete',seed=screen.SEED,training_rows=12,prediction_rows=3,
            training_ids_sha256=screen.audit.row_hash(train),prediction_ids_sha256=screen.audit.row_hash(query),
            inner_train_ids_sha256=screen.audit.row_hash(train[inner]),
            inner_validation_ids_sha256=screen.audit.row_hash(train[valid]),
            full_prediction_support=True,fresh_refit_from_original_checkpoint=True,
            inner_optimizer_steps=50,refit_optimizer_steps=2,selected_steps=2,inner_history=history,
            refit_history=[dict(step=i,finite_gradients=True,disjoint_ids=True) for i in (1,2)])
        screen.check_adaptation_log(log,train,query,y,screen.SEED)
        for key,value in [('status','complete'),('selected_steps',3),('prediction_ids_sha256','wrong')]:
            altered=copy.deepcopy(log);altered[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                screen.check_adaptation_log(altered,train,query,y,screen.SEED)

    def test_continuation_is_not_required_to_beat_stronger_component(self):
        components=[dict(accuracy=.816,log_loss=.380),dict(accuracy=.818,log_loss=.375)]
        best=dict(accuracy=.820);folds=[.81,.82,.82]
        result=dict(accuracy=.817,log_loss=.377)
        self.assertTrue(screen.exploratory_continuation(result,components,best,folds)['passed'])
        for altered in [dict(accuracy=.815,log_loss=.377),dict(accuracy=.817,log_loss=.381)]:
            self.assertFalse(screen.exploratory_continuation(altered,components,best,folds)['passed'])
        self.assertFalse(screen.exploratory_continuation(result,components,best,folds[:2])['passed'])

    def test_paired_counts_are_directional(self):
        y=np.array([0,1,0,1,0,1]);raw=np.array([.8,.8,.2,.2,.2,.8]);candidate=np.array([.2,.8,.8,.8,.2,.8])
        splits=[(None,np.array([0,1])),(None,np.array([2,3])),(None,np.array([4,5]))]
        pair=screen.paired(candidate,raw,y,splits)
        self.assertEqual(pair['overall']['repaired'],2)
        self.assertEqual(pair['overall']['broken'],1)
        self.assertEqual(pair['overall']['disagreements'],3)
        self.assertEqual(len(pair['folds']),3)


if __name__=='__main__': unittest.main()
