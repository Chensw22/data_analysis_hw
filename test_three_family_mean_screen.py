import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import three_family_mean_screen as screen


class FixedMeanTests(unittest.TestCase):
    def test_incomplete_component_stops_before_development_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root/screen.PLAN;plan.parent.mkdir(parents=True)
            plan.write_text(json.dumps(dict(candidate=screen.NAME,reserved_run=screen.RUN,
                components=[dict(version=v,run=r,weight=1/3) for r,v,_ in screen.COMPONENTS],known_inputs={})))
            with patch.object(screen,'ROOT',root), \
                 patch.object(screen,'require_committed',return_value='a'*40), \
                 patch.object(screen.audit,'read_development_only') as read:
                with self.assertRaisesRegex(RuntimeError,'Incomplete component'): screen.screen()
                read.assert_not_called()
            self.assertFalse((root/'artifacts'/screen.RUN).exists())

    def test_exactly_three_equal_means_without_weight_option(self):
        p = np.array([[.1,.9],[.2,.7],[.9,.2]])
        np.testing.assert_allclose(screen.fixed_mean(p), [.4,.6],rtol=0,atol=1e-15)
        with self.assertRaises(ValueError): screen.fixed_mean(p[:2])
        with self.assertRaises(TypeError): screen.fixed_mean(p,weights=[.1,.2,.7])

    def test_band_uses_global_reference_not_only_component_best(self):
        components = [dict(accuracy=.81),dict(accuracy=.815),dict(accuracy=.816)]
        screen.require_practical_band(components,dict(accuracy=.82))
        with self.assertRaises(RuntimeError):
            screen.require_practical_band(components,dict(accuracy=.821))

    def test_fold_ids_and_labels_must_align_for_each_hash_scheme(self):
        train,query,labels = np.array([2,5,8,10]),np.array([0,3]),np.array([0,1,0,1])
        for kind in ('mitra','tabdpt','kumo'):
            record = dict(status='complete',seed=screen.SEED,training_rows=4,prediction_rows=2,
                training_ids_sha256=screen.ids_hash(train,kind),
                prediction_ids_sha256=screen.ids_hash(query,kind),
                labels_sha256=screen.ids_hash(labels,kind),
                full_training_context=True,full_prediction_support=True)
            screen.check_fold_log(record,kind,train,query,labels,screen.SEED)
            with self.assertRaises(RuntimeError):
                screen.check_fold_log(record,kind,train[::-1],query,labels,screen.SEED)
            if kind!='mitra':
                with self.assertRaises(RuntimeError):
                    screen.check_fold_log(record,kind,train,query[::-1],labels,screen.SEED)
                with self.assertRaises(RuntimeError):
                    screen.check_fold_log(record,kind,train,query,1-labels,screen.SEED)

    def test_cache_misalignment_and_partial_probabilities_rejected(self):
        y = np.array([0,1,0,1,0,1]);p = np.array([.1,.9,.2,.8,.3,.7])
        splits = [(None,np.array([0,1])),(None,np.array([2,3])),(None,np.array([4,5]))]
        np.testing.assert_array_equal(screen.cached_fold_scores(p,np.ones(3),y,splits),np.ones(3))
        with self.assertRaises(RuntimeError): screen.cached_fold_scores(p[::-1],np.ones(3),y,splits)
        with self.assertRaises(RuntimeError): screen.cached_fold_scores(p[:-1],np.ones(3),y,splits)


if __name__=='__main__': unittest.main()
