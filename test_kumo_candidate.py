"""Mocked wrapper isolation and complete-eight-view evidence gates."""
import copy
import hashlib
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import kumo_candidate as candidate
from test_kumo_full8_probe import context,comparison


def test_registers_runner_feature_flag_without_shadowing_full_support_policy(monkeypatch):
    for name in ('fit','evaluate','CONFIGS','BLEND'):
        monkeypatch.setattr(candidate.iterate,name,getattr(candidate.iterate,name))
    candidate.configure()
    assert candidate.iterate.fit is candidate.fit
    assert candidate.iterate.evaluate is candidate.evaluate
    config=candidate.iterate.CONFIGS[candidate.NAME]
    assert config['context'] is False
    assert config['support_context']=='all_training_rows_no_sampling'
    assert candidate.iterate.BASELINE in candidate.iterate.CONFIGS


def full8_evidence(tmp_path,monkeypatch):
    path=tmp_path/'full8';path.mkdir();base=tmp_path/'base';base.mkdir()
    for name in ('freeze.json','result.json'):(base/name).write_text('{}')
    lock='fake env\n';(path/'environment.lock.txt').write_text(lock)
    freeze=dict(settings=candidate.full8.SETTINGS,backend_settings=candidate.backend.SETTINGS,
        device='mps',checkpoint_sha256=candidate.backend.WEIGHT_SHA,sources={},
        environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),
        base_probe_freeze_sha256=candidate.backend.sha(base/'freeze.json'),
        base_probe_result_sha256=candidate.backend.sha(base/'result.json'))
    candidate.backend.write(path/'freeze.json',freeze)
    result=dict(status='complete',device='mps',synthetic_only=True,accuracy_computed=False,
        freeze_sha256=candidate.backend.sha(path/'freeze.json'),support_rows=8693,query_rows=4277,views=8,
        first_probability_rows=4277,reload_probability_rows=4277,reload_check=comparison(),
        forwarded_contexts=[context(8693) for _ in range(8)],reload_forwarded_contexts=[context(8693) for _ in range(8)])
    candidate.backend.write(path/'result.json',result)
    monkeypatch.setattr(candidate,'PROBE',path);monkeypatch.setattr(candidate.full8,'PROBE',base)
    monkeypatch.setattr(candidate.full8,'verify_base_resource',lambda:None)
    monkeypatch.setattr(candidate.full8,'source_hashes',lambda:{})
    monkeypatch.setattr(candidate,'environment_lock',lambda:lock)
    return path,freeze,result


def test_exact_full8_both_passes_and_current_environment_required(tmp_path,monkeypatch):
    path,freeze,result=full8_evidence(tmp_path,monkeypatch)
    assert candidate.verify_resource()==freeze
    changes=[]
    bad=copy.deepcopy(result);bad['views']=1;changes.append(bad)
    bad=copy.deepcopy(result);bad['reload_probability_rows']=17;changes.append(bad)
    bad=copy.deepcopy(result);bad['reload_forwarded_contexts'].pop();changes.append(bad)
    bad=copy.deepcopy(result);bad['forwarded_contexts'][0]['output_device']='cpu';changes.append(bad)
    bad=copy.deepcopy(result);bad['reload_check']['max_abs_difference']=1e-3;changes.append(bad)
    for bad in changes:
        candidate.backend.write(path/'result.json',bad)
        with pytest.raises(RuntimeError):candidate.verify_resource()
    candidate.backend.write(path/'result.json',result)
    monkeypatch.setattr(candidate,'environment_lock',lambda:'changed')
    with pytest.raises(RuntimeError):candidate.verify_resource()


def test_fit_passes_training_rows_labels_seed_only_and_blocks_overlap(tmp_path,monkeypatch):
    class Fake:
        def __init__(self):self.metadata={};self.labels=np.array([0,1,0,1]);self.released=False
        def predict_proba(self,x):self.query=x.index.tolist();return np.tile([.4,.6],(len(x),1))
        def release(self):self.released=True
    fake=Fake();x=pd.DataFrame({'value':range(8)})
    monkeypatch.setattr(candidate,'RUN_OUT',tmp_path);monkeypatch.setattr(candidate,'DEADLINE',None)
    with patch.object(candidate.adapter,'fit_context',return_value=fake) as train:
        p,_=candidate.fit(candidate.NAME,{False:(x,[])},[1,3,5,7],[0,1,0,1],[0,2],42)
        assert train.call_args.args[0].index.tolist()==[1,3,5,7]
        assert train.call_args.args[1]==[0,1,0,1] and train.call_args.args[3]==42
        assert train.call_args.kwargs=={'device':'mps'}
        assert fake.query==[0,2] and fake.released
        np.testing.assert_array_equal(p,[.6,.6])
        for queries in ([3,5],[-1,0],[8,9],[0,0],[.5,2.5]):
            with pytest.raises(ValueError):candidate.fit(candidate.NAME,{False:(x,[])},[1,3],[0,1],queries,43)
        with pytest.raises(RuntimeError,match='exists'):
            candidate.fit(candidate.NAME,{False:(x,[])},[1,3,5,7],[0,1,0,1],[0,2],42)


def test_screen_guard_and_source_freeze_precede_baseline():
    calls=[]
    with patch.object(candidate,'record_protocol',side_effect=lambda _:calls.append('freeze')), \
            patch.object(candidate,'BASE_EVALUATE',side_effect=lambda *a:calls.append('baseline')):
        with pytest.raises(ValueError):candidate.evaluate(candidate.NAME,None,None,np.arange(8693),20261112,None)
        assert calls==[]
        candidate.evaluate(candidate.iterate.BASELINE,None,None,np.arange(6954),candidate.iterate.SEEDS[0],None)
        assert calls==['freeze','baseline']


def test_export_budget_and_mps_required_before_runtime(tmp_path,monkeypatch):
    monkeypatch.setattr(candidate,'verify_resource',lambda:None)
    monkeypatch.setattr(candidate,'source_hashes',lambda:{})
    monkeypatch.setattr(candidate.full8,'require_committed',lambda _:None)
    monkeypatch.setattr(candidate.backend.torch.backends.mps,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='no CPU fallback'):candidate.configure_runtime(tmp_path,'export')
    assert candidate.SETTINGS['screen_budget_seconds']==candidate.SETTINGS['export_budget_seconds']==7200
