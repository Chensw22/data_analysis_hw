"""CPU tiny-model and provenance checks; no real weights, data or MPS work."""
from unittest.mock import patch

import joblib
import numpy as np
import pytest
import torch

import kumo_adapter as adapter
from test_kumo_resource_probe import tiny_model


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(2)


def tiny_predictor(monkeypatch):
    backend=adapter.backend
    prototype=tiny_model()
    state={k:v.detach().cpu().clone() for k,v in prototype.models[backend.sdm.Task.classification].state_dict().items()}
    def rebuild(weights,device):
        assert device=='cpu'
        model=tiny_model();model.models[backend.sdm.Task.classification].load_state_dict(weights)
        return model
    monkeypatch.setattr(backend,'new_model',rebuild)
    x,y,z=backend.synthetic(24,5,17)
    with patch.object(backend,'verify_checkpoint'),patch.object(backend.torch,'load',return_value=state):
        predictor=adapter.fit_context(x,y,np.arange(24),71,'cpu')
    return predictor,x,y,z


def test_full8_official_semantics_context_and_cpu_roundtrip(monkeypatch,tmp_path):
    model,x,y,z=tiny_predictor(monkeypatch)
    first=model.predict_proba(z)
    assert model._engine is None and model.labels.tolist()==y.tolist()
    assert len(model.last_forward_contexts)==8
    assert {row['rows'] for row in model.last_forward_contexts}=={24}
    assert model.metadata['prediction_calls']==1
    path=tmp_path/'tiny.joblib';joblib.dump(model,path);restored=joblib.load(path)
    assert restored.device=='cpu' and restored._engine is None
    assert all(t.device.type=='cpu' for t in restored.state.values())
    second=restored.predict_proba(z)
    adapter.backend.compare(first,second,atol=0.)
    assert restored.metadata['prediction_calls']==2 and restored.seed==71
    # Direct property assignment is the generic exporter's documented API.
    restored.device='mps';assert restored.device=='mps';restored.device='cpu'


def test_training_context_seed_state_and_source_tampering_rejected(monkeypatch):
    for mutation in ('ids','labels','context','state','seed','views','source'):
        model,x,y,z=tiny_predictor(monkeypatch)
        if mutation=='ids':model.ids[0]+=100
        if mutation=='labels':model.labels[0]=1-model.labels[0]
        if mutation=='context':model.x.numerical[0,0]+=1
        if mutation=='state':next(t for t in model.state.values() if t.is_floating_point()).view(-1)[0]+=1
        if mutation=='seed':model.seed+=1
        if mutation=='views':model.estimators=1
        if mutation=='source':model.sources['kumo_adapter.py']='invalid'
        with pytest.raises(RuntimeError):model.predict_proba(z)


def test_failed_prediction_releases_engine_and_query_labels_ignored(monkeypatch):
    model,x,y,z=tiny_predictor(monkeypatch)
    a=model.predict_proba(z);z['Transported']=1
    b=model.predict_proba(z);adapter.backend.compare(a,b,atol=0.)
    monkeypatch.setattr(adapter.backend,'probabilities',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('injected')))
    with pytest.raises(RuntimeError,match='injected'):model.predict_proba(z)
    assert model._engine is None


def test_bad_training_ids_labels_rejected_before_checkpoint(monkeypatch):
    x,y,z=adapter.backend.synthetic(12,3,1)
    with patch.object(adapter.backend,'verify_checkpoint') as verify:
        for yy,ids in [(np.ones(12),np.arange(12)),(y,np.zeros(12,dtype=int)),(y,np.arange(12)+.2),(y[:-1],np.arange(12))]:
            with pytest.raises(ValueError):adapter.fit_context(x,yy,ids,42,'cpu')
        verify.assert_not_called()
