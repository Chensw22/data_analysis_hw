"""Synthetic metadata and mocked-export tests: no real labels, weights or MPS."""
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import weakref

import numpy as np
import pandas as pd
import torch

import prepare_group_episode_submission as exporter
from mitra_group_episode_candidate import validate_cached_arrays


def row_hash(value):return hashlib.sha256(np.asarray(value,dtype=np.int64).tobytes()).hexdigest()
def group_hash(value):return hashlib.sha256(json.dumps(list(map(str,value)),separators=(',',':')).encode()).hexdigest()


def training_log(rows,seed,recipe,ids):
    return dict(status='fit_complete',recipe=recipe,training_rows=rows,seed=seed,
        training_ids_sha256=row_hash(ids),selected_steps=1,inner_optimizer_steps=50,
        refit_optimizer_steps=1,fresh_refit_from_original_checkpoint=True,full_prediction_support=True,
        query_chunk=2318,parameters=75670026,inner_history=[dict(step=i) for i in range(51)],
        refit_history=[dict(step=1)],inner_group_overlap_count=0)


class Predictor:
    previous=None
    altered=False
    def __init__(self,y,metadata):
        self.labels=np.array(y);self.support=np.zeros((8693,1),dtype=np.float32)
        self.metadata=metadata;self.query_chunk=2318;self.state={'toy':torch.zeros(1)}
        self.device='mps';self._engine=None
    def release(self):self._engine=None
    def __getstate__(self):return {**self.__dict__,'device':'cpu','_engine':None}
    def predict_proba(self,query):
        old=self.previous() if self.previous is not None else None
        if old is not None and old is not self:raise AssertionError('Old predictor retained across full reload.')
        if self.device!='mps' or len(query)!=4277:raise AssertionError('Full reload/backend contract differs.')
        p=np.full(4277,.6)
        if old is None and self.altered:p[0]=.4
        return np.column_stack([1-p,p])


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.root_patch=patch.object(exporter,'ROOT',self.root);self.root_patch.start()
        self.recipe=dict(max_steps=50,episode_support=1024,episode_query=256,query_chunk=2318,
            trainable_parameters=75670026,checkpoint_sha256='w'*64,learning_rate=3e-6,
            weight_decay=.3,fresh_full_refit=True)
        self.environment='fixture==1\n';self.env_sha=hashlib.sha256(self.environment.encode()).hexdigest()
        for name in exporter.OWN_SOURCES+['mitra_group_episode.py',exporter.MODULE+'.py',
                'mitra_candidate.py','mitra_full_episode.py','mitra_full_episode_probe.py']:
            (self.root/name).write_text('# fixture\n')
        exporter.write(self.root/exporter.AUTH,dict(batch_id=exporter.BATCH,competition='spaceship-titanic',
            maximum_upload_attempts=10,weekly_reserve_percent=30,no_duplicate_or_blind_retry_upload=True))
        fake_torch=SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:True)),
            set_num_threads=Mock(),mps=SimpleNamespace(set_per_process_memory_fraction=Mock()))
        self.candidate=SimpleNamespace(NAME=exporter.NAME,SCREEN_SETTINGS={'kind':'group'},
            assert_frozen=Mock(),validate_cached_arrays=validate_cached_arrays,
            audit=SimpleNamespace(row_hash=row_hash,DEV_HASH=row_hash(np.arange(6954))),
            core=SimpleNamespace(SETTINGS=self.recipe,group_hash=group_hash,
                inner_split=lambda y,g,s:(np.arange(6954),np.arange(6954,8693))),
            base=SimpleNamespace(environment_lock=lambda:self.environment,WEIGHT_SHA='w'*64,
                torch=fake_torch,MitraPredictor=Predictor,release_memory=Mock(),check_deadline=Mock(),DEADLINE=None))
        self.group_fixture();self.resource_fixture();self.decision_fixture()
        Predictor.previous=None;Predictor.altered=False

    def tearDown(self):
        self.root_patch.stop();self.tmp.cleanup();Predictor.previous=None;Predictor.altered=False

    def group_fixture(self):
        out=self.root/exporter.GROUP;report=self.root/exporter.GROUP_REPORT
        reference=self.root/'reference';reference.mkdir();self.candidate.REFERENCE=reference
        source={name:exporter.sha(self.root/name) for name in ['mitra_group_episode.py',exporter.MODULE+'.py']}
        protocol=dict(candidate=exporter.NAME,recipe=self.recipe,screen=self.candidate.SCREEN_SETTINGS,
            sources=source,historical_files={})
        for folder in [out,report]:exporter.write(folder/'protocol.json',protocol)
        dev=np.arange(6954);assignment=np.repeat(np.arange(3,dtype=np.int8),2318)
        groups=np.array([f'{i:04d}' for i in dev]);passengers=np.array([f'{g}_01' for g in groups])
        raw=np.full(6954,.6);p=np.full(6954,.7)
        identifiers=dict(dev_ids=dev,fold_id=assignment,group_ids=groups,passenger_ids=passengers)
        np.savez(out/'folds.npz',**identifiers)
        np.savez(out/'group_20261112.npz',**identifiers,prob=p,reference_prob=raw)
        np.savez(reference/'group_20261112.npz',**identifiers,prob=raw)
        rows=[]
        for fold in range(3):
            a=dev[assignment!=fold];b=dev[assignment==fold]
            np.savez(reference/f'group_20261112_fold{fold}_rows.npz',train_ids=a,validation_ids=b)
            row=dict(fold=fold,fit_seed=20261112+fold,train_rows=len(a),query_rows=len(b),
                train_ids_sha256=row_hash(a),query_ids_sha256=row_hash(b),group_overlap_count=0,
                accuracy=.8,reference_accuracy=.79,selected_steps=1)
            log=training_log(len(a),20261112+fold,self.recipe,a)
            log['training_group_ids_sha256']=group_hash(groups[a])
            for folder in [out,report]:
                exporter.write(folder/f'fold{fold}.json',row);exporter.write(folder/f'fold{fold}_training.json',log)
            rows.append(row)
        summary=dict(status='complete',candidate=exporter.NAME,kind='group',seed=20261112,
            development_rows=6954,no_old_holdout_targets_used=True,automatic_eligibility=False,
            random_tier_update=False,protocol_sha256=exporter.sha(out/'protocol.json'),
            oof_sha256=exporter.sha(out/'group_20261112.npz'),folds=rows,accuracy=.8)
        progress=dict(status='complete',whole_score_available=True,completed_folds=rows)
        for folder in [out,report]:
            exporter.write(folder/'summary.json',summary);exporter.write(folder/'progress.json',progress)
        exporter.write(report/'review.json',{'fixture':True})
        pd.DataFrame(rows).to_csv(report/'group_metrics.csv',index=False)

    def resource_fixture(self):
        out=self.root/exporter.RESOURCE;report=self.root/exporter.RESOURCE_REPORT;out.mkdir(parents=True)
        (out/'environment.lock.txt').write_text(self.environment)
        (out/'model.joblib').write_bytes(b'NOT A REAL MODEL; NEVER LOAD')
        sources={}
        for name in ['mitra_candidate.py','mitra_full_episode.py','mitra_full_episode_probe.py']:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((self.root/name).read_bytes());sources[name]=exporter.sha(target)
        manifest=dict(status='complete',selection={'version':'v81_Mitra_full_episode50'},fit_rows=8693,
            rows=4277,reload_rows=4277,export_reload_inference_verified=True,reload_threshold_decisions_equal=True,
            reload_max_absolute_difference=0.,model_sha256=exporter.sha(out/'model.joblib'),
            environment_sha256=self.env_sha,provenance={'source_hashes':sources})
        for folder in [out,report]:
            exporter.write(folder/'manifest.json',manifest)
            exporter.write(folder/'authorization.json',{'model_sha256':manifest['model_sha256']})
            exporter.write(folder/'selection.json',manifest['selection'])
        fit=training_log(8693,20261112,self.recipe,np.arange(8693))
        fit.update(inner_training_rows=6954,inner_validation_rows=1739,
            checkpoint_sha256='w'*64,precision='MPS bf16 autocast with FP32 weights',prediction_rows=4277)
        for name in ['fit_metadata.json','reload_metadata.json']:exporter.write(out/name,fit)
        exporter.write(out/'preparation_protocol.json',dict(environment_sha256=self.env_sha,sources_frozen_before_full_labels=True))

    def decision_fixture(self):
        self.decision=dict(batch_id=exporter.BATCH,batch_authorization_sha256=exporter.sha(self.root/exporter.AUTH),
            module=exporter.MODULE,version=exporter.NAME,source_phase=exporter.PHASE,proposal=exporter.PROPOSAL,
            validation_kind='group',group_accuracy=.8,source_protocol_sha256=exporter.sha(self.root/exporter.GROUP/'protocol.json'),
            evidence_files={p:exporter.sha(self.root/p) for p in exporter.required_evidence_paths()},
            exploratory=True,prior_A_eligible=None,prior_S_supported=False,threshold=.5,maximum_attempts=1,
            resource_envelope=exporter.RESOURCE_ENVELOPE,rationale='One group-recipe exploratory candidate.',
            resource_extrapolation_disclosure='Small change from completed v81 inner6954/1739; fail if outside envelope.')
        exporter.write(self.root/exporter.DECISION,self.decision)

    def runtime_data(self):
        y=np.arange(8693)%2;groups=np.array([f'{i:04d}' for i in range(8693)])
        train=pd.DataFrame({'PassengerId':[f'{g}_01' for g in groups],'Transported':y})
        test=pd.DataFrame({'PassengerId':[f'T{i:04d}_01' for i in range(4277)]})
        x=pd.DataFrame({'feature':np.arange(12970)})
        def fit(frame,labels,ids,received_groups,seed,**kwargs):
            self.assertEqual(len(frame),8693);np.testing.assert_array_equal(labels,y)
            np.testing.assert_array_equal(ids,np.arange(8693));np.testing.assert_array_equal(received_groups,groups)
            self.assertEqual(seed,20261112);self.assertEqual(kwargs['max_steps'],50);self.assertEqual(kwargs['device'],'mps')
            metadata=training_log(8693,seed,self.recipe,ids)
            metadata.update(inner_training_rows=6954,inner_validation_rows=1739,training_group_ids_sha256=group_hash(groups))
            model=Predictor(labels,metadata);Predictor.previous=weakref.ref(model)
            return model
        self.candidate.core.fit_group_episode=Mock(side_effect=fit)
        return train,test,x,y,groups

    def patches(self,stack,data=None,duplicate=False):
        original_import=exporter.importlib.import_module
        stack.enter_context(patch.object(exporter,'require_committed',return_value='a'*40))
        stack.enter_context(patch.object(exporter.importlib,'import_module',side_effect=lambda name,*a,**k:
            self.candidate if name==exporter.MODULE else original_import(name,*a,**k)))
        stack.enter_context(patch.object(exporter.signal,'signal',return_value='old-handler'))
        stack.enter_context(patch.object(exporter.signal,'setitimer'))
        stack.enter_context(patch.object(exporter,'duplicate_attempts',return_value=['old'] if duplicate else []))
        if data is not None:
            reader=stack.enter_context(patch.object(exporter,'read_full_fit_data',return_value=data))
        else:reader=None
        def validate(path):
            table=pd.read_csv(path,dtype=str)
            self.assertEqual(table.shape,(4277,2));self.assertTrue(table.Transported.isin(['True','False']).all())
            return dict(rows=4277,sha256=exporter.sha(path),valid=True)
        stack.enter_context(patch.object(exporter,'validate_csv',side_effect=validate))
        return reader

    def test_complete_group_identity_resource_and_committed_decision_gate(self):
        with ExitStack() as stack:
            self.patches(stack)
            evidence=exporter.decision_evidence()
        self.assertEqual(evidence['group_summary']['accuracy'],.8)
        self.assertEqual(evidence['fullfit_resource']['inner_support'],6954)
        self.assertIn(exporter.DECISION,evidence['sources'])
        self.assertIn(f'{exporter.GROUP_REPORT}/summary.json',evidence['sources'])

    def test_bad_decision_or_uncommitted_files_never_read_full_labels(self):
        for change in [dict(validation_kind='random'),dict(screen_accuracy=.8),dict(group_accuracy=.81),
                       dict(prior_S_supported=True),dict(evidence_files={}),dict(maximum_attempts=2)]:
            with self.subTest(change=change),ExitStack() as stack:
                exporter.write(self.root/exporter.DECISION,{**self.decision,**change})
                reader=self.patches(stack,self.runtime_data())
                with self.assertRaises(RuntimeError):exporter.prepare()
                reader.assert_not_called();self.candidate.core.fit_group_episode.assert_not_called()
        exporter.write(self.root/exporter.DECISION,self.decision)
        with ExitStack() as stack:
            reader=self.patches(stack,self.runtime_data())
            stack.enter_context(patch.object(exporter,'require_committed',side_effect=RuntimeError('uncommitted decision')))
            with self.assertRaisesRegex(RuntimeError,'uncommitted'):exporter.prepare()
            reader.assert_not_called()

    def test_incomplete_group_or_reordered_cached_ids_rejected(self):
        out=self.root/exporter.GROUP;report=self.root/exporter.GROUP_REPORT
        original=exporter.read(out/'progress.json')
        for folder in [out,report]:exporter.write(folder/'progress.json',{**original,'status':'incomplete'})
        with self.assertRaisesRegex(RuntimeError,'three-fold'):
            exporter.verify_group_evidence(self.candidate,self.decision)
        for folder in [out,report]:exporter.write(folder/'progress.json',original)
        with np.load(out/'folds.npz') as saved:values={k:saved[k].copy() for k in saved.files}
        values['dev_ids']=values['dev_ids'][::-1];np.savez(out/'folds.npz',**values)
        with self.assertRaisesRegex(RuntimeError,'identity'):
            exporter.verify_group_evidence(self.candidate,self.decision)

    def test_full_export_releases_first_model_and_keeps_group_fields(self):
        with ExitStack() as stack:
            reader=self.patches(stack,self.runtime_data())
            manifest=exporter.prepare()
        reader.assert_called_once();self.candidate.core.fit_group_episode.assert_called_once()
        self.assertEqual(manifest['validation_kind'],'group')
        self.assertEqual(manifest['selection']['group_accuracy'],.8)
        self.assertNotIn('screen_accuracy',manifest['selection']);self.assertNotIn('source_run',manifest['selection'])
        self.assertIsNone(manifest['selection']['A_eligible']);self.assertFalse(manifest['selection']['S_supported'])
        self.assertEqual(manifest['reload_max_absolute_difference'],0.)
        self.assertEqual(manifest['authorization']['maximum_upload_attempts_for_this_file'],1)
        for name in ['manifest.json','authorization.json','selection.json']:
            self.assertEqual(exporter.read(self.root/'artifacts'/exporter.PROPOSAL/name),
                             exporter.read(self.root/'reports'/exporter.PROPOSAL/name))
        import submit_iteration as submitter
        with patch.object(submitter,'ROOT',self.root),patch.object(submitter,'AUTH',self.root/exporter.AUTH), \
                patch.object(submitter.workflow,'validate_submission',return_value=dict(rows=4277,
                    sha256=exporter.sha(self.root/'artifacts'/exporter.PROPOSAL/'submission.csv'))), \
                patch.object(submitter.subprocess,'check_output',return_value=''):
            bound,used=submitter.verify(exporter.PROPOSAL)
            self.assertEqual(bound['version'],exporter.NAME);self.assertEqual(used,0)
        with patch.object(exporter,'decision_evidence') as evidence:
            with self.assertRaisesRegex(RuntimeError,'no overwrite'):exporter.prepare()
            evidence.assert_not_called()

    def test_failed_reload_duplicate_and_resource_overflow_leave_no_csv(self):
        for defect in ['reload','duplicate','envelope']:
            with self.subTest(defect=defect),ExitStack() as stack:
                proposal=f'{exporter.PROPOSAL}_{defect}'
                stack.enter_context(patch.object(exporter,'PROPOSAL',proposal))
                self.decision['proposal']=proposal;exporter.write(self.root/exporter.DECISION,self.decision)
                Predictor.altered=defect=='reload';self.patches(stack,self.runtime_data(),duplicate=defect=='duplicate')
                if defect=='envelope':
                    stack.enter_context(patch.object(self.candidate.core,'inner_split',return_value=(np.arange(7001),np.arange(7001,8693))))
                with self.assertRaises(RuntimeError):exporter.prepare()
                for base in ['artifacts','reports']:
                    folder=self.root/base/proposal
                    self.assertEqual(exporter.read(folder/'failure.json')['status'],'failed_incomplete_not_upload_eligible')
                    for name in ['submission.csv','manifest.json','authorization.json','selection.json']:
                        self.assertFalse((folder/name).exists())
                self.assertFalse((self.root/'artifacts'/proposal/'pending').exists())
                if defect=='envelope':self.candidate.core.fit_group_episode.assert_not_called()


if __name__=='__main__':unittest.main()
