import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
from contextlib import ExitStack

import numpy as np
import pandas as pd

import prepare_top5_submission as preparation


class FakePredictor:
    altered = False
    original = None
    def __init__(self, rows):
        self.labels = np.arange(rows)%2
        self.device='mps'
        self.metadata={'test_double':True,'full_support':True}
    def release(self):
        pass
    def predict_proba(self, frame, **kwargs):
        if self.original is not None and self.original() is not None:
            raise AssertionError('Original model still resident at reload inference.')
        if self.device != 'mps':
            raise AssertionError('Reload did not restore MPS backend.')
        p=np.full(len(frame),.6)
        if self.altered:
            p[0]=.4
        return np.column_stack([1-p,p])
    def __getstate__(self):
        state=dict(self.__dict__)
        state['device']='cpu'
        return state


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.root_patch=patch.object(preparation,'ROOT',self.root)
        self.root_patch.start()
        self.run='run_fixture'
        self.run_out=self.root/'artifacts'/self.run
        self.run_out.mkdir(parents=True)
        self.name=preparation.SPECS['mitra_mirror_ensemble'][0]
        self.files=['mitra_mirror_ensemble.py','top5_validate.py','audit_top5_references.py',
            'prepare_top5_submission.py','test_prepare_top5_submission.py','kaggle_workflow.py']
        for filename in self.files:
            (self.root/filename).write_text('# isolated fixture\n')
        preparation.write(self.root/preparation.POLICY,{'fixture':'policy'})
        preparation.write(self.root/preparation.AUTH,dict(eligibility_policy=preparation.POLICY,
            eligibility_policy_sha256=preparation.sha(self.root/preparation.POLICY),
            no_duplicate_or_blind_retry_upload=True))
        sources={f:preparation.sha(self.root/f) for f in self.files+[preparation.POLICY,preparation.AUTH]}
        preparation.write(self.run_out/'protocol.json',{'sources':sources})
        self.binding=dict(original_screen_protocol_sha256=preparation.sha(self.run_out/'protocol.json'),
            validation_policy_sha256=preparation.sha(self.root/preparation.POLICY),
            authorization_sha256=preparation.sha(self.root/preparation.AUTH))
        self.eligibility=dict(status='complete',candidate=self.name,A_eligible=True,
            A_checks={'all_conditions':True},screen_accuracy=.82,S_supported=False,**self.binding)
        preparation.write(self.run_out/'stability/eligibility.json',self.eligibility)
        preparation.write(self.run_out/'stability/protocol.json',dict(candidate=self.name,
            own_source_sha256=sources['top5_validate.py'],**self.binding))
        FakePredictor.altered=False
        FakePredictor.original=None

    def tearDown(self):
        self.root_patch.stop()
        self.temporary.cleanup()
        FakePredictor.original=None

    def runtime(self):
        train=pd.DataFrame({'Transported':np.arange(8693)%2==0})
        test=pd.DataFrame({'PassengerId':[f'{i:04d}_01' for i in range(4277)]})
        matrix=pd.DataFrame({'Age':np.arange(8693+4277,dtype=float)})
        data=SimpleNamespace(SEEDS=[20261112],dataset=unittest.mock.Mock(return_value=(
            train,test,{False:(matrix,[])},None)))
        core=SimpleNamespace(environment_lock=lambda:'fake==1\n',release_memory=lambda:None,DEADLINE=None)
        candidate=SimpleNamespace(NAME=self.name,mitra=core,
            configure_runtime=unittest.mock.Mock(),assert_frozen=unittest.mock.Mock())
        def fit(name,matrices,a,y,b,seed):
            self.assertEqual(name,self.name)
            self.assertEqual(len(a),8693)
            self.assertEqual(len(y),8693)
            self.assertEqual(len(b),4277)
            self.assertFalse(np.intersect1d(a,b).size)
            model=FakePredictor(len(a))
            FakePredictor.original=weakref.ref(model)
            return np.full(len(b),.6),model
        candidate.fit=unittest.mock.Mock(side_effect=fit)
        return candidate,data

    def execute(self,candidate,data,proposal='proposal_fixture',duplicate=False,validation='random'):
        def validate(path):
            frame=pd.read_csv(path,dtype=str)
            self.assertEqual(len(frame),4277)
            self.assertEqual(list(frame),['PassengerId','Transported'])
            self.assertTrue(frame.Transported.isin(['True','False']).all())
            return dict(rows=4277,sha256=preparation.sha(path),valid=True)
        real_import=preparation.importlib.import_module
        def load(name,*args,**kwargs):
            return candidate if name=='mitra_mirror_ensemble' else real_import(name,*args,**kwargs)
        with patch.object(preparation.importlib,'import_module',side_effect=load), \
             patch.dict(sys.modules,{'iterate':data}), \
             patch.object(preparation.subprocess,'check_output',return_value='f'*40+'\n'), \
             patch.object(preparation,'validate_csv',side_effect=validate), \
             patch.object(preparation.signal,'signal',return_value=None), \
             patch.object(preparation.signal,'setitimer'), \
             patch.object(preparation,'duplicate_attempts',return_value=['old/attempt.json'] if duplicate else []):
            return preparation.prepare('mitra_mirror_ensemble',self.run,proposal,validation=validation)

    def group_fixture(self):
        import top5_group_evaluate as group
        # Only metadata fixtures and mock environment verification; actual queue
        # structural checks and make_decision are exercised without labels/models.
        old=self.run_out
        self.run='run_040'
        self.run_out=self.root/'artifacts'/self.run
        old.rename(self.run_out)
        preparation.write(self.run_out/'stability/eligibility.json',{**self.eligibility,'A_eligible':False})
        for filename in ['top5_group_evaluate.py','test_top5_group_evaluate.py','mitra_context_candidate.py']:
            (self.root/filename).write_text('# group fixture\n')
        preparation.write(self.root/group.POLICY,{'fixture':'new group policy'})
        preparation.write(self.root/group.AUTH,dict(no_duplicate_or_blind_retry_upload=True,
            all_fixed_queue_must_complete_before_submission=True,revision_disclosure='Adaptive group revision; old failure preserved.'))
        sources={f:preparation.sha(self.root/f) for f in
            ['top5_group_evaluate.py','test_top5_group_evaluate.py',group.POLICY,group.AUTH]}
        phase=self.root/'artifacts/top5_group_phase'
        phase.mkdir()
        record=dict(sources=sources,historical_files={},screen_evidence={},legacy_caches={},
            budgets_seconds={'refs':{'mitra':900,'v44':900},'candidates':{'mirror':900,'context':900}})
        for key,spec in group.CANDIDATES.items():
            folder=self.root/'artifacts'/spec['run']
            folder.mkdir(exist_ok=True)
            if not (folder/'protocol.json').exists():
                preparation.write(folder/'protocol.json',{'sources':{}})
            pd.DataFrame([dict(version=spec['version'],accuracy=.82)]).to_csv(folder/'screen_results.csv',index=False)
            for filename in ['protocol.json','screen_results.csv']:
                path=folder/filename
                record['historical_files'][str(path.relative_to(self.root))]=preparation.sha(path)
            record['screen_evidence'][key]=dict(version=spec['version'],accuracy=.82)
        preparation.write(phase/'protocol.json',record)
        phasehash=preparation.sha(phase/'protocol.json')
        summary_hashes={}
        for kind,specs in [('refs',group.REFERENCES),('candidates',group.CANDIDATES)]:
            for key,spec in specs.items():
                folder=phase/kind/key
                folder.mkdir(parents=True)
                env='v44' if (kind,key)==('refs','v44') else 'mitra'
                preparation.write(folder/'protocol.json',dict(kind=kind,key=key,version=spec['version'],
                    phase_sha256=phasehash,environment=env,budget_seconds=900,execution='CPU4' if env=='v44' else 'MPS'))
                rows=[]
                accuracy=.82 if kind=='candidates' else (.81 if key=='mitra' else .815)
                for seed in group.SEEDS:
                    path=folder/f'group_{seed}.npz'
                    np.savez(path,prob=np.array([.2,.8]))
                    rows.append(dict(seed=seed,accuracy=accuracy,oof_path=str(path.relative_to(self.root)),
                        oof_sha256=preparation.sha(path),legacy_reused=False,legacy_evidence=None,
                        folds=[dict(fold=i) for i in range(3)]))
                preparation.write(folder/'summary.json',dict(status='complete',kind=kind,key=key,
                    version=spec['version'],phase_sha256=phasehash,results=rows,
                    execution_protocol_sha256=preparation.sha(folder/'protocol.json')))
                summary_hashes[f'{kind}/{key}']=preparation.sha(folder/'summary.json')
        decisions={}
        for key,spec in group.CANDIDATES.items():
            rows=[dict(seed=seed,accuracy=.82,vs_mitra=dict(gain=.01,reference_accuracy=.81),
                       vs_v44=dict(gain=.005,reference_accuracy=.815)) for seed in group.SEEDS]
            intervals={'new_mean_vs_v44':{'lower95':-.01}}
            decision=group.make_decision(.82,rows,intervals)
            decisions[key]=dict(**decision,candidate=spec['version'],original_screen_accuracy=.82,
                results=rows,paired_cluster_sensitivity=intervals)
        preparation.write(phase/'eligibility.json',dict(status='complete',fixed_queue_complete=True,
            phase_sha256=phasehash,policy_sha256=preparation.sha(self.root/group.POLICY),
            authorization_sha256=preparation.sha(self.root/group.AUTH),component_summary_sha256=summary_hashes,
            candidates=decisions))
        (phase/'eligibility.sha256').write_text(preparation.sha(phase/'eligibility.json')+'\n')
        stack=ExitStack()
        stack.enter_context(patch.object(group,'ROOT',self.root))
        stack.enter_context(patch.object(group,'PHASE',phase))
        gate=stack.enter_context(patch.object(group,'assert_phase',return_value=record))
        return stack,phase,group,gate

    def test_eligibility_requires_exact_status_identity_and_every_binding(self):
        preparation.eligibility_evidence(self.run,self.name)
        for field,value in [('status','partial'),('candidate','other'),('A_eligible',False),
                ('A_eligible',1),('A_checks',{'one':False}),
                ('original_screen_protocol_sha256','wrong'),('validation_policy_sha256','wrong'),
                ('authorization_sha256','wrong')]:
            with self.subTest(field=field,value=value):
                changed=copy.deepcopy(self.eligibility)
                changed[field]=value
                preparation.write(self.run_out/'stability/eligibility.json',changed)
                with self.assertRaises(RuntimeError):
                    preparation.eligibility_evidence(self.run,self.name)
        preparation.write(self.run_out/'stability/eligibility.json',self.eligibility)
        (self.root/'top5_validate.py').write_text('# changed\n')
        with self.assertRaisesRegex(RuntimeError,'source not frozen'):
            preparation.eligibility_evidence(self.run,self.name)

    def test_failed_gate_never_imports_candidate_or_reads_full_labels(self):
        changed={**self.eligibility,'A_eligible':False}
        preparation.write(self.run_out/'stability/eligibility.json',changed)
        candidate,data=self.runtime()
        with self.assertRaisesRegex(RuntimeError,'A-eligible'):
            self.execute(candidate,data)
        candidate.configure_runtime.assert_not_called()
        data.dataset.assert_not_called()
        self.assert_failed()

    def assert_failed(self):
        for area in ['artifacts','reports']:
            folder=self.root/area/'proposal_fixture'
            self.assertTrue((folder/'failure.json').exists())
            for filename in ['manifest.json','authorization.json','submission.csv']:
                self.assertFalse((folder/filename).exists(),filename)

    def test_complete_roundtrip_binds_one_upload_only_and_full_shape(self):
        candidate,data=self.runtime()
        manifest=self.execute(candidate,data)
        self.assertEqual(manifest['status'],'complete')
        self.assertEqual(manifest['fit_rows'],8693)
        self.assertEqual(manifest['reload_rows'],4277)
        self.assertEqual(manifest['reload_max_absolute_difference'],0)
        self.assertTrue(manifest['export_reload_inference_verified'])
        out=self.root/'artifacts/proposal_fixture'
        authorization=preparation.read(out/'authorization.json')
        self.assertEqual(authorization['maximum_upload_attempts_for_this_file'],1)
        self.assertEqual(authorization['file_sha256'],preparation.sha(out/'submission.csv'))
        self.assertEqual(authorization['eligibility_sha256'],preparation.sha(self.run_out/'stability/eligibility.json'))
        candidate.configure_runtime.assert_called_once_with(out/'fit_logs',stage='stability')
        data.dataset.assert_called_once()
        with self.assertRaisesRegex(RuntimeError,'Proposal exists'):
            self.execute(candidate,data)
        self.assertEqual(data.dataset.call_count,1)

    def test_reload_mismatch_and_duplicate_attempts_leave_no_authorization(self):
        for duplicate in [False,True]:
            with self.subTest(duplicate=duplicate):
                if duplicate:
                    import shutil
                    shutil.rmtree(self.root/'artifacts/proposal_fixture')
                    shutil.rmtree(self.root/'reports/proposal_fixture')
                FakePredictor.altered=not duplicate
                candidate,data=self.runtime()
                with self.assertRaisesRegex(RuntimeError,'already has an upload attempt' if duplicate else 'reload inference differs'):
                    self.execute(candidate,data,duplicate=duplicate)
                self.assert_failed()

    def test_tolerance_never_allows_changed_threshold_and_nested_attempts_found(self):
        with self.assertRaisesRegex(RuntimeError,'threshold'):
            preparation.check_reload(np.array([.5-1e-7]),np.array([.5+1e-7]))
        with self.assertRaisesRegex(RuntimeError,'Incomplete'):
            preparation.probability_vector([.2],4277)
        record=self.root/'reports/batch/nested/online_attempts/one.json'
        preparation.write(record,dict(file_sha256='same'))
        self.assertEqual(preparation.duplicate_attempts('same'),[str(record.relative_to(self.root))])
        self.assertEqual(preparation.duplicate_attempts('different'),[])

    def test_group_complete_export_binds_new_queue_and_preserves_old_false(self):
        stack,phase,group,gate=self.group_fixture()
        prior=preparation.sha(self.run_out/'stability/eligibility.json')
        candidate,data=self.runtime()
        with stack:
            manifest=self.execute(candidate,data,validation='group')
        out=self.root/'artifacts/proposal_fixture'
        self.assertEqual(manifest['validation_track'],'group')
        self.assertEqual(manifest['provenance']['eligibility_sha256'],preparation.sha(phase/'eligibility.json'))
        self.assertEqual(preparation.sha(out/'eligibility.json'),preparation.sha(phase/'eligibility.json'))
        self.assertEqual(preparation.sha(out/'group_phase_protocol.json'),preparation.sha(phase/'protocol.json'))
        self.assertEqual(preparation.sha(out/'validation_policy.json'),preparation.sha(self.root/group.POLICY))
        self.assertNotEqual(preparation.sha(out/'validation_policy.json'),preparation.sha(self.root/preparation.POLICY))
        self.assertEqual(preparation.sha(self.run_out/'stability/eligibility.json'),prior)
        self.assertFalse(manifest['provenance']['prior_random_eligibility']['A_eligible'])
        self.assertFalse((out/'stability_protocol.json').exists())
        authorization=preparation.read(out/'authorization.json')
        self.assertEqual(authorization['validation_track'],'group')
        self.assertEqual(authorization['eligibility_sha256'],preparation.sha(phase/'eligibility.json'))
        self.assertEqual(len(authorization['fixed_queue_component_summary_sha256']),4)
        gate.assert_called_with('mitra')
        data.dataset.assert_called_once()

    def test_group_incomplete_or_tampered_evidence_rejected_before_labels(self):
        stack,phase,group,_=self.group_fixture()
        changes=['missing_component','summary','oof','candidate','ineligible',
                 'genuinely_ineligible','paired_gain','sidecar']
        snapshot={p:p.read_bytes() for p in phase.rglob('*') if p.is_file()}
        with stack:
            for change in changes:
                with self.subTest(change=change):
                    for path,content in snapshot.items():
                        path.write_bytes(content)
                    out=self.root/'artifacts/proposal_fixture'
                    report=self.root/'reports/proposal_fixture'
                    if out.exists():
                        import shutil
                        shutil.rmtree(out);shutil.rmtree(report)
                    if change=='missing_component':
                        (phase/'candidates/context/summary.json').unlink()
                    elif change=='summary':
                        path=phase/'refs/v44/summary.json'
                        value=preparation.read(path);value['results'][0]['accuracy']=.1
                        preparation.write(path,value)
                    elif change=='oof':
                        with (phase/'refs/mitra/group_20261115.npz').open('ab') as handle:
                            handle.write(b'changed')
                    else:
                        path=phase/'eligibility.json';value=preparation.read(path)
                        selected=value['candidates']['mirror']
                        if change=='candidate':selected['candidate']='other'
                        if change=='ineligible':selected['A_eligible']=False
                        if change=='genuinely_ineligible':
                            for row in selected['results']:
                                row['accuracy']=.81
                                row['vs_mitra']['gain']=0.
                                row['vs_v44']['gain']=-.005
                            selected.update(group.make_decision(selected['original_screen_accuracy'],
                                selected['results'],selected['paired_cluster_sensitivity']))
                            summary_path=phase/'candidates/mirror/summary.json'
                            summary=preparation.read(summary_path)
                            for row in summary['results']:
                                row['accuracy']=.81
                            preparation.write(summary_path,summary)
                            value['component_summary_sha256']['candidates/mirror']=preparation.sha(summary_path)
                        if change=='paired_gain':selected['results'][1]['vs_mitra']['gain']=.2
                        if change=='sidecar':value['fixed_queue_complete']=False
                        preparation.write(path,value)
                        if change!='sidecar':
                            (phase/'eligibility.sha256').write_text(preparation.sha(path)+'\n')
                    candidate,data=self.runtime()
                    with self.assertRaises(RuntimeError):
                        self.execute(candidate,data,validation='group')
                    data.dataset.assert_not_called()
                    candidate.configure_runtime.assert_not_called()
                    self.assert_failed()

    def test_group_rejects_wrong_run_and_no_automatic_fallback_to_old_track(self):
        with self.assertRaisesRegex(RuntimeError,'fixed module/run'):
            preparation.group_eligibility_evidence('mitra_mirror_ensemble','run_043',self.name)
        with self.assertRaises(ValueError):
            preparation.validation_evidence('mitra_mirror_ensemble',self.run,self.name,'unknown')


if __name__=='__main__':
    unittest.main()
