"""Synthetic boundary and serialization tests; no competition rows or MPS work."""
import joblib
import numpy as np
import pytest
import torch

import kumo_resource_probe as k
from sdm.models.kumo.tabular.model import _KumoTabular


def tiny_model():
    torch.manual_seed(17)
    model=k.KumoTabular(task='classification',size='small',pretrained=False,device='meta')
    model.models[k.sdm.Task.classification]=_KumoTabular(num_classes=10,num_quantiles=999,
        cell_channels=16,num_embedding_layers=1,num_embedding_heads=4,num_inducing_points=8,
        group_size=3,num_frequencies=4,num_readout_tokens=4,icl_channels=32,
        num_icl_layers=2,num_icl_heads=4,num_icl_key_value_heads_for_query=2,device='cpu')
    return model.eval().requires_grad_(False)


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def test_official_cpu_recipe_equals_transfer_callback_and_query_partitions():
    x,y,z=k.synthetic(32,9,42);m=tiny_model();a,b,c=k.features(x),k.labels(y),k.features(z)
    raw,_=k.probabilities(m,a,b,c,'cpu',8,19,bridge=False)
    bridged,observed=k.probabilities(m,a,b,c,'cpu',8,19)
    k.compare(raw,bridged,atol=0.)
    assert len(observed)==8 and {d['rows'] for d in observed}=={32}
    assert all(d['input_dtype']=='torch.float32' and d['input_device']=='cpu' for d in observed)
    left,_=k.probabilities(m,a,b,k.features(z.iloc[:4]),'cpu',8,19)
    right,_=k.probabilities(m,a,b,k.features(z.iloc[4:]),'cpu',8,19)
    k.compare(raw,np.concatenate([left,right]))
    assert c.device.type=='cpu' and a.device.type=='cpu'


def test_semantics_drop_unrequested_labels_and_keep_missing_and_unknown():
    x,y,z=k.synthetic(20,5,10)
    x['Transported']=y
    first=k.features(x);x['Transported']=1-y;second=k.features(x)
    assert set(first.column_names)==set(k.COMPACT)
    assert set(first.categorical_columns)==set(k.CATEGORICAL)
    torch.testing.assert_close(first.numerical,second.numerical,equal_nan=True)
    assert 'unseen_synthetic_planet' in z['HomePlanet'].values
    with pytest.raises(ValueError):k.labels(np.ones(20))
    with pytest.raises(ValueError):k.labels(np.array([0,1,2]))


def test_cpu_artifact_roundtrip_rebuilds_engine_and_checks_state(monkeypatch,tmp_path):
    x,y,z=k.synthetic(24,5,7);model=tiny_model()
    state={name:value.detach().cpu().clone() for name,value in model.models[k.sdm.Task.classification].state_dict().items()}
    calls=[]
    def rebuild(values,device):
        assert device=='cpu';calls.append(device)
        result=tiny_model();result.models[k.sdm.Task.classification].load_state_dict(values)
        return result
    monkeypatch.setattr(k,'new_model',rebuild)
    predictor=k.KumoPredictor.__new__(k.KumoPredictor)
    predictor.__dict__.update(state=state,state_sha=k.state_hash(state),checkpoint_sha=k.WEIGHT_SHA,
        x=k.features(x),y=k.labels(y),device='cpu',estimators=2,seed=31,_engine=model,last_forward_contexts=[])
    expected=predictor.predict_proba(z)
    path=tmp_path/'predictor.joblib';joblib.dump(predictor,path);restored=joblib.load(path)
    assert restored._engine is None and restored.device=='cpu'
    k.compare(expected,restored.predict_proba(z),atol=0.)
    assert calls==['cpu'] and all(t.device.type=='cpu' for t in restored.state.values())
    restored._engine=None
    key=next(name for name,value in restored.state.items() if value.is_floating_point())
    restored.state[key].view(-1)[0]+=1
    with pytest.raises(RuntimeError,match='state changed'):restored.engine()


def test_bridge_rejects_reduced_support_and_float64_at_transfer_boundary():
    x,y,_=k.synthetic(12,3,9);a=k.features(x);b=k.labels(y)
    with pytest.raises(RuntimeError):k.CpuRecipeDeviceBridge('cpu',11).on_context_preprocessing_end(None,a,b,None,None)
    with pytest.raises(RuntimeError):k.CpuRecipeDeviceBridge('cpu',12).on_context_preprocessing_end(None,a,b,None,None)
    doubles=k.sdm.TableTensor(columns={'numerical':['value']},numerical=torch.ones(12,1,dtype=torch.float64))
    with pytest.raises(RuntimeError):k.CpuRecipeDeviceBridge('cpu',12).on_context_preprocessing_end(None,doubles,b,None,None)
    # Above raw mixed table still has categorical features; the bridge accepts
    # only the final numerical FP32 recipe representation, never raw tables.


def test_comparison_rejects_nonfinite_or_threshold_changes():
    with pytest.raises(AssertionError):k.compare(np.array([.499999]),np.array([.500001]))
    with pytest.raises(AssertionError):k.compare(np.array([np.nan]),np.array([.5]))
