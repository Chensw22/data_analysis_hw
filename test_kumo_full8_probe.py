"""Metadata-only gating tests: no model inference or accelerator work."""
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import kumo_full8_probe as probe


def context(rows):
    return dict(rows=rows,columns=13,input_device='cpu',input_dtype='torch.float32',output_device='mps')


def comparison():return dict(max_abs_difference=0.,threshold_disagreements=0,tolerance=2e-5)


def base_evidence(tmp_path,monkeypatch):
    path=tmp_path/'base';path.mkdir()
    lock='synthetic-environment\n';(tmp_path/'requirements-kumo.lock.txt').write_text(lock)
    (path/'environment.lock.txt').write_text(lock)
    cases=[(128,17,8),(4636,2318,1),(8693,4277,1)]
    freeze=dict(settings=probe.backend.SETTINGS,device='mps',checkpoint_sha256=probe.backend.WEIGHT_SHA,
        code_commit=probe.backend.CODE_COMMIT,model_revision=probe.backend.MODEL_REVISION,
        cases=[dict(support=s,query=q,views=v) for s,q,v in cases],sources={},
        environment_sha256=hashlib.sha256(lock.encode()).hexdigest())
    probe.backend.write(path/'freeze.json',freeze)
    result=dict(status='complete',device='mps',synthetic_only=True,accuracy_computed=False,
        freeze_sha256=probe.backend.sha(path/'freeze.json'),cases=[])
    for s,q,v in cases:
        result['cases'].append(dict(support_rows=s,query_rows=q,views=v,
            forwarded_contexts=[context(s) for _ in range(v)],probability_rows=q,
            partition_check=comparison(),reload_check=comparison()))
    probe.backend.write(path/'result.json',result)
    monkeypatch.setattr(probe,'PROBE',path);monkeypatch.setattr(probe,'ROOT',tmp_path)
    monkeypatch.setattr(probe.backend,'source_hashes',lambda:{})
    monkeypatch.setattr(probe.backend,'environment',lambda:lock)
    monkeypatch.setattr(probe.backend,'verify_checkpoint',lambda:None)
    monkeypatch.setattr(probe.adapter,'verify_sources',lambda:{})
    return path,freeze,result,lock


def test_exact_completed_base_cases_required_before_full8(tmp_path,monkeypatch):
    path,freeze,result,_=base_evidence(tmp_path,monkeypatch)
    assert probe.verify_base_resource()==freeze
    changes=[]
    bad=copy.deepcopy(result);bad['status']='failed_inconclusive';changes.append(bad)
    bad=copy.deepcopy(result);bad['cases'].pop();changes.append(bad)
    bad=copy.deepcopy(result);bad['cases'][-1]['forwarded_contexts'][0]['rows']=1024;changes.append(bad)
    bad=copy.deepcopy(result);bad['cases'][-1]['reload_check']['threshold_disagreements']=1;changes.append(bad)
    bad=copy.deepcopy(result);bad['cases'][-1]['partition_check']['max_abs_difference']=float('nan');changes.append(bad)
    for bad in changes:
        (path/'result.json').write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):probe.verify_base_resource()
    probe.backend.write(path/'result.json',result)
    (path/'environment.lock.txt').write_text('changed')
    with pytest.raises(RuntimeError):probe.verify_base_resource()


def test_existing_destination_and_staging_prevent_any_gpu_or_fit(tmp_path):
    with patch.object(probe,'verify_base_resource') as verify:
        with pytest.raises(RuntimeError,match='exists'):probe.run(tmp_path)
        verify.assert_not_called()
    with patch.object(probe,'BASE',tmp_path/'stage'),patch.object(probe,'ROOT',tmp_path), \
            patch.object(probe.subprocess,'check_output') as command:
        with pytest.raises(RuntimeError,match='relocate'):probe.require_committed({})
        command.assert_not_called()


def test_untracked_or_changed_formal_sources_fail_closed(tmp_path):
    with patch.object(probe,'BASE',tmp_path),patch.object(probe,'ROOT',tmp_path):
        with patch.object(probe.subprocess,'check_output',return_value=b''):
            with pytest.raises(RuntimeError,match='committed'):probe.require_committed({'model.py':'digest'})
        with patch.object(probe.subprocess,'check_output',side_effect=[b'model.py\0',b' M model.py']):
            with pytest.raises(RuntimeError,match='differ'):probe.require_committed({'model.py':'digest'})
