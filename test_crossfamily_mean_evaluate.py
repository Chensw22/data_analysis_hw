import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import accuracy_score,log_loss

import crossfamily_mean_evaluate as candidate


class CrossfamilyTests(unittest.TestCase):
    def test_exact_equal_mean_no_implicit_partial_or_weight_search(self):
        a=np.array([.1,.9,.4]);b=np.array([.7,.1,.8])
        np.testing.assert_array_equal(candidate.fixed_mean(a,b),.5*a+.5*b)
        with self.assertRaises(RuntimeError):candidate.fixed_mean(a,b[:2])
        with self.assertRaises(RuntimeError):candidate.fixed_mean(a,np.array([.2,np.nan,.3]))

    def test_group_gate_unchanged_and_both_random_checks_additional(self):
        rows=[dict(seed=s,accuracy=.85,vs_mitra={'gain':.02},vs_v44={'gain':.015}) for s in candidate.GROUP_SEEDS]
        random=[dict(seed=s,vs_mitra={'gain':0.}) for s in candidate.RANDOM_SEEDS]
        intervals={'new_mean_vs_v44':{'lower95':.002}}
        old=candidate.group.make_decision(.85,rows,intervals)
        result=candidate.make_decision(.85,random,rows,intervals)
        self.assertTrue(result['A_eligible'] and result['S_supported'])
        for key,value in old['A_checks'].items():self.assertEqual(result['A_checks'][key],value)
        self.assertEqual(result['S_checks'],old['S_checks'])
        random[1]['vs_mitra']['gain']=-1/6954
        result=candidate.make_decision(.85,random,rows,intervals)
        self.assertFalse(result['A_eligible']);self.assertFalse(result['S_supported'])
        with self.assertRaises(RuntimeError):candidate.make_decision(.85,random[:1],rows,intervals)

    def test_oof_sha_and_all_four_row_identities(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'oof.npz'
            dev=np.arange(12);groups=np.repeat(['a','b','c','d'],3)
            passengers=np.array([f'p{i}' for i in dev]);assignment=np.tile([0,1,2],4)
            y=dev%2;p=np.where(y,.8,.2)
            original=dict(prob=p,dev_ids=dev,group_ids=groups,passenger_ids=passengers,fold_id=assignment)
            np.savez(path,**original)
            entry=dict(path='oof.npz',sha256=candidate.digest(path),accuracy=1.,log_loss=float(log_loss(y,p)))
            plan=dict(kind='random',references={'mitra':entry})
            with patch.object(candidate,'ROOT',root):
                np.testing.assert_array_equal(candidate.load_probability(plan,'mitra',dev,y,groups,passengers,assignment),p)
                for field in ['dev_ids','group_ids','passenger_ids','fold_id']:
                    changed={**original,field:np.roll(original[field],1)}
                    np.savez(path,**changed);entry['sha256']=candidate.digest(path)
                    with self.subTest(field=field),self.assertRaises(RuntimeError):
                        candidate.load_probability(plan,'mitra',dev,y,groups,passengers,assignment)
                np.savez(path,**original);entry['sha256']='wrong'
                with self.assertRaisesRegex(RuntimeError,'SHA'):
                    candidate.load_probability(plan,'mitra',dev,y,groups,passengers,assignment)

    def test_legacy_screen_requires_matching_provenance_and_fold_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dev=np.arange(24);y=dev%2
            groups=np.array([f'g{i}' for i in dev]);passengers=np.array([f'p{i}' for i in dev])
            assignment=candidate.assignment_for('screen',20261112,y,groups)
            p=np.where(y,.7,.3);folds=[accuracy_score(y[assignment==i],p[assignment==i]>=.5) for i in range(3)]
            np.savez(root/'screen.npz',prob=p,fold_accuracy=folds)
            protocol=dict(development_ids_sha256=candidate.audit.row_hash(dev),group_ids_sha256=candidate.audit.string_hash(groups),
                passenger_ids_sha256=candidate.audit.string_hash(passengers))
            (root/'audit.json').write_text(json.dumps(protocol))
            entry=dict(path='screen.npz',sha256=candidate.digest(root/'screen.npz'),audit_protocol='audit.json',
                fold_accuracy=folds,accuracy=1.,log_loss=float(log_loss(y,p)))
            plan=dict(kind='screen',references={'v44':entry})
            with patch.object(candidate,'ROOT',root):
                candidate.load_probability(plan,'v44',dev,y,groups,passengers,assignment)
                entry['fold_accuracy']=[.5]*3
                with self.assertRaises(AssertionError):candidate.load_probability(plan,'v44',dev,y,groups,passengers,assignment)
                protocol['passenger_ids_sha256']='wrong';(root/'audit.json').write_text(json.dumps(protocol))
                with self.assertRaisesRegex(RuntimeError,'row-order'):
                    candidate.load_probability(plan,'v44',dev,y,groups,passengers,assignment)

    def test_group_fit_context_and_seed_must_match_query_partition(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dev=np.arange(12);y=dev%2
            groups=np.repeat(['a','b','c','d','e','f'],2)
            passengers=np.array([f'p{i}' for i in dev]);assignment=np.repeat([0,1,2],4)
            p=np.where(y,.8,.2);path=root/'group.npz'
            np.savez(path,prob=p,dev_ids=dev,group_ids=groups,passenger_ids=passengers,fold_id=assignment)
            folds=[dict(fit_seed=20261115+fold,group_overlap=0,
                train_ids_sha256=candidate.audit.row_hash(dev[a]),query_ids_sha256=candidate.audit.row_hash(dev[b]))
                for fold,(a,b) in enumerate(candidate.group.checked_partition(assignment,groups))]
            entry=dict(path='group.npz',sha256=candidate.digest(path),accuracy=1.,
                log_loss=float(log_loss(y,p)),legacy_reused=False,folds=folds)
            plan=dict(kind='group',seed=20261115,references={'mitra':entry})
            with patch.object(candidate,'ROOT',root):
                candidate.load_probability(plan,'mitra',dev,y,groups,passengers,assignment)
                folds[1]['fit_seed']=0
                with self.assertRaisesRegex(RuntimeError,'fit seed/train/query'):
                    candidate.load_probability(plan,'mitra',dev,y,groups,passengers,assignment)

    def test_frozen_checks_precede_any_candidate_scoring_and_changes_failclosed(self):
        with patch.object(candidate,'assert_frozen',side_effect=RuntimeError('Input changed')), \
             patch.object(candidate,'fixed_mean') as mean,patch.object(candidate,'score_row') as score:
            with self.assertRaisesRegex(RuntimeError,'Input changed'):candidate.evaluate()
            mean.assert_not_called();score.assert_not_called()
        with patch.object(candidate,'check_policy',return_value={}), \
             patch.object(candidate,'inventory',side_effect=RuntimeError('Bad original hash')), \
             patch.object(candidate,'PHASE',Path('/nonexistent/crossfamily_synthetic_test')), \
             patch.object(candidate,'fixed_mean') as mean,patch.object(candidate,'score_row') as score:
            with self.assertRaisesRegex(RuntimeError,'Bad original hash'):candidate.freeze()
            mean.assert_not_called();score.assert_not_called()


if __name__=='__main__':unittest.main()
