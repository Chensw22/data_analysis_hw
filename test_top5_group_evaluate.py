import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import top5_group_evaluate as candidate


class GroupPhaseTests(unittest.TestCase):
    def fixture(self):
        dev=np.arange(12,dtype=np.int64)+20
        y=np.arange(12)%2
        groups=np.repeat(np.arange(6).astype(str),2)
        passengers=np.array([f'{g}_{i}' for i,g in enumerate(groups)])
        x=pd.DataFrame({'feature':np.arange(40,dtype=float)})
        assignment=np.repeat([0,1,2,0,1,2],2).astype(np.int8)
        return {False:(x,[])},dev,y,groups,passengers,assignment

    def test_strict_group_partition_rejects_overlap_missing_fold_and_bad_shape(self):
        *_,groups,passengers,assignment=self.fixture()
        folds=candidate.checked_partition(assignment,groups)
        np.testing.assert_array_equal(np.sort(np.concatenate([b for _,b in folds])),np.arange(12))
        broken=assignment.copy();broken[0]=1
        for value in [broken,np.zeros(12,dtype=int),assignment[:-1]]:
            with self.assertRaises(RuntimeError):candidate.checked_partition(value,groups)

    def test_oof_hash_and_ids_are_required_even_when_accuracy_is_identical(self):
        _,dev,y,groups,passengers,assignment=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'oof.npz'
            np.savez(path,prob=np.full(12,.4),dev_ids=dev,group_ids=groups,
                     passenger_ids=passengers,fold_id=assignment)
            expected=candidate.digest(path)
            p=candidate.artifact_probability(path,expected,dev,groups,passengers,assignment)
            self.assertEqual(float(((p>=.5)==y).mean()),.5)
            with self.assertRaises(RuntimeError):
                candidate.artifact_probability(path,expected,dev[::-1],groups,passengers,assignment)
            with self.assertRaises(RuntimeError):
                candidate.artifact_probability(path,expected,dev,groups,passengers,(assignment+1)%3)
            # Same decisions/accuracy but different bytes must fail original SHA.
            np.savez(path,prob=np.full(12,.41),dev_ids=dev,group_ids=groups,
                     passenger_ids=passengers,fold_id=assignment)
            with self.assertRaises(RuntimeError):
                candidate.artifact_probability(path,expected,dev,groups,passengers,assignment)

    def test_development_reader_never_parses_old_holdout_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            train,test=Path(directory)/'train.csv',Path(directory)/'test.csv'
            train.write_text('PassengerId,Age,Transported\na_1,10,True\nb_1,20,DO_NOT_PARSE\nc_1,30,False\n')
            test.write_text('PassengerId,Age\nd_1,40\n')
            attrs,_,y=candidate.audit.read_development_only(train,test,np.array([0,2]))
            self.assertNotIn('Transported',attrs.columns)
            np.testing.assert_array_equal(y,[1,0])
            self.assertEqual(len(attrs),3)

    def test_policy_rejects_removing_a_candidate_or_seed(self):
        scope=dict(candidates=[s['version'] for s in candidate.CANDIDATES.values()],
            references=[s['version'] for s in candidate.REFERENCES.values()],
            group_split_seeds=candidate.SEEDS,new_primary_seeds=candidate.SEEDS[1:],folds_per_seed=3)
        policy=dict(fixed_scope_before_new_scores=scope,bootstrap_recipe=dict(replicates=2000,seed=20261119))
        with tempfile.TemporaryDirectory() as directory,patch.object(candidate,'ROOT',Path(directory)):
            root=Path(directory);(root/'reports').mkdir()
            policy_path=root/candidate.POLICY;policy_path.write_text(json.dumps(policy))
            auth=dict(eligibility_policy=candidate.POLICY,eligibility_policy_sha256=candidate.digest(policy_path),
                fixed_candidates=scope['candidates'],fixed_references=scope['references'],
                budgets_seconds=dict(refs=dict(mitra=60,v44=60),candidates=dict(mirror=60,context=60)))
            (root/candidate.AUTH).write_text(json.dumps(auth))
            candidate.check_policy()
            policy['fixed_scope_before_new_scores']['candidates']=scope['candidates'][:1]
            policy_path.write_text(json.dumps(policy))
            auth['eligibility_policy_sha256']=candidate.digest(policy_path)
            (root/candidate.AUTH).write_text(json.dumps(auth))
            with self.assertRaises(RuntimeError):candidate.check_policy()

    def test_fixed_three_split_A_and_S_conditions(self):
        rows=[dict(seed=s,accuracy=.85,vs_mitra=dict(gain=.012),vs_v44=dict(gain=.011)) for s in candidate.SEEDS]
        intervals={'new_mean_vs_v44':{'lower95':.002}}
        good=candidate.make_decision(.84,rows,intervals)
        self.assertTrue(good['A_eligible']);self.assertTrue(good['S_supported'])
        self.assertFalse(good['automatic_upload_ready'])
        for invalid in [rows[:2],rows[::-1],[rows[0],rows[1],rows[1]]]:
            with self.assertRaises(RuntimeError):candidate.make_decision(.84,invalid,intervals)
        import copy
        for index,ref,gain in [(0,'mitra',-.0001),(1,'mitra',0),(2,'v44',-.0101)]:
            bad=copy.deepcopy(rows);bad[index][f'vs_{ref}']['gain']=gain
            self.assertFalse(candidate.make_decision(.84,bad,intervals)['A_eligible'])
        low=copy.deepcopy(rows)
        low[1]['vs_mitra']['gain']=low[2]['vs_mitra']['gain']=.00099
        self.assertFalse(candidate.make_decision(.84,low,intervals)['A_eligible'])
        self.assertFalse(candidate.make_decision(.83,rows,intervals)['S_supported'])
        self.assertFalse(candidate.make_decision(.84,rows,{'new_mean_vs_v44':{'lower95':0}})['S_supported'])

    def test_entire_queue_required_before_eligibility(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(candidate,'PHASE',Path(directory)):
            with self.assertRaisesRegex(RuntimeError,'Entire fixed queue'):
                candidate.complete_queue({})
            with patch.object(candidate,'assert_phase',return_value={}), \
                 patch.object(candidate.audit,'load_audit_data',side_effect=AssertionError('no data until queue complete')):
                with self.assertRaisesRegex(RuntimeError,'Entire fixed queue'):candidate.finalize()
            self.assertFalse((Path(directory)/'eligibility.json').exists())

    def test_execution_identity_checked_even_if_its_new_hash_matches_summary(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(candidate,'ROOT',Path(directory)), \
             patch.object(candidate,'PHASE',Path(directory)/'phase'):
            phase=candidate.PHASE;phase.mkdir();(phase/'protocol.json').write_text('{}')
            phase_sha=candidate.digest(phase/'protocol.json')
            budgets={'refs':{'mitra':60,'v44':60},'candidates':{'mirror':60,'context':60}}
            record=dict(budgets_seconds=budgets,legacy_caches={})
            for kind,specs in [('refs',candidate.REFERENCES),('candidates',candidate.CANDIDATES)]:
                for key,spec in specs.items():
                    folder=phase/kind/key;folder.mkdir(parents=True)
                    env='v44' if (kind,key)==('refs','v44') else 'mitra'
                    execution=dict(kind=kind,key=key,version=spec['version'],phase_sha256=phase_sha,
                        environment=env,budget_seconds=60,execution='CPU4' if env=='v44' else 'MPS')
                    (folder/'protocol.json').write_text(json.dumps(execution))
                    rows=[dict(seed=s,oof_path=str((folder/f'group_{s}.npz').relative_to(Path(directory))),
                        legacy_reused=False,legacy_evidence=None,folds=[{'fold':i} for i in range(3)]) for s in candidate.SEEDS]
                    (folder/'summary.json').write_text(json.dumps(dict(status='complete',kind=kind,key=key,
                        version=spec['version'],phase_sha256=phase_sha,results=rows,
                        execution_protocol_sha256=candidate.digest(folder/'protocol.json'))))
            self.assertEqual(len(candidate.complete_queue(record)),4)
            folder=phase/'candidates/context'
            bad=json.loads((folder/'protocol.json').read_text());bad['version']='wrong_model'
            (folder/'protocol.json').write_text(json.dumps(bad))
            summary=json.loads((folder/'summary.json').read_text())
            summary['execution_protocol_sha256']=candidate.digest(folder/'protocol.json')
            (folder/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(RuntimeError,'Execution protocol identity'):
                candidate.complete_queue(record)

    def test_uncommitted_or_untracked_phase_sources_cannot_freeze(self):
        with patch.object(candidate.subprocess,'check_output',return_value=b''):
            with self.assertRaisesRegex(RuntimeError,'Commit every phase'):
                candidate.require_committed(['source.py'])
        with patch.object(candidate.subprocess,'check_output',side_effect=[b'source.py\0',b' M source.py\n']):
            with self.assertRaisesRegex(RuntimeError,'committed versions'):
                candidate.require_committed(['source.py'])

    def test_fake_execution_only_passes_fold_training_targets_and_uses_fixed_seeds(self):
        matrices,dev,y,groups,passengers,assignment=self.fixture()
        calls=[]
        def fake_fit(name,mm,train,labels,query,seed):
            positions=np.searchsorted(dev,train)
            np.testing.assert_array_equal(labels,y[positions])
            self.assertEqual(len(train),8);self.assertEqual(len(query),4)
            self.assertFalse(set(train)&set(query))
            calls.append(seed)
            return np.full(len(query),.4),SimpleNamespace()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(candidate,'ROOT',Path(directory)), \
             patch.object(candidate,'PHASE',Path(directory)/'phase'):
            phase=candidate.PHASE;phase.mkdir();(phase/'protocol.json').write_text('{}')
            hashes={}
            for seed in candidate.SEEDS:
                file=phase/f'folds_{seed}.npz'
                np.savez(file,fold_id=assignment,dev_ids=dev,group_ids=groups,passenger_ids=passengers)
                hashes[str(seed)]=candidate.digest(file)
            record=dict(budgets_seconds={'refs':{'v44':60}},legacy_caches={},fold_hashes=hashes,
                data=candidate.data_identity(matrices,dev,y,groups,passengers))
            framework=SimpleNamespace(dataset=lambda: self.fail('Never use full-label dataset()'))
            with patch.object(candidate,'assert_phase',return_value=record), \
                 patch.object(candidate.audit,'initialize_reference',return_value=(framework,SimpleNamespace(),fake_fit,{},'v44_seed_ensemble')), \
                 patch.object(candidate.audit,'load_audit_data',return_value=(matrices,dev,y,groups,passengers)):
                candidate.execute('refs','v44')
            self.assertEqual(calls,[s+f for s in candidate.SEEDS for f in range(3)])
            result=json.loads((phase/'refs/v44/summary.json').read_text())
            self.assertEqual(result['status'],'complete')
            self.assertEqual([r['seed'] for r in result['results']],candidate.SEEDS)


if __name__=='__main__':unittest.main()
