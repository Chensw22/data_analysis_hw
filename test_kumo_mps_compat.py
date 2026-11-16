"""CPU and mocked-device tests only; no checkpoint download or MPS execution."""
from types import SimpleNamespace

import pytest
import torch

import kumo_resource_probe as probe  # Sets the pinned vendor import path.
import kumo_mps_compat as compat
from test_kumo_resource_probe import tiny_model


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def embedding():
    torch.manual_seed(19)
    return compat.KumoCellEmbedding(channels=16, group_size=3, num_frequencies=4,
                                    device='cpu').eval().requires_grad_(False)


def test_cpu_unchanged_idempotent_state_parameters_and_class():
    model = tiny_model()
    x, y, z = probe.synthetic(24, 7, 19)
    tables = probe.features(x), probe.labels(y), probe.features(z)
    before, _ = probe.probabilities(model, *tables, 'cpu', estimators=2)
    states = {k: v.clone() for k, v in model.state_dict().items()}
    params = [id(p) for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    method = compat.KumoCellEmbedding._forward
    matmul = torch.matmul
    assert compat.install(model) == compat.install(model) == 1
    after, _ = probe.probabilities(model, *tables, 'cpu', estimators=2)
    probe.compare(before, after, atol=0.)
    assert list(states) == list(model.state_dict())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, states[key], atol=0, rtol=0)
    assert params == [id(p) for p in model.parameters()]
    assert modes == [m.training for m in model.modules()]
    assert compat.KumoCellEmbedding._forward is method and torch.matmul is matmul


def test_mocked_mps_branch_keeps_bias_missingness_and_strided_destination(monkeypatch):
    cell = embedding()
    x = torch.randn(2, 11, 5)
    x[0, 3, 2] = float('nan')
    mask = torch.tensor([[True, False, True, False, False]]).expand(2, -1)
    with torch.inference_mode():
        expected = cell(x, mask, train_size=7)
        compat.install(cell)
        # Exercise the MPS-only allocation/copy branch with CPU arithmetic.
        monkeypatch.setattr(compat, '_needs_safe_out', lambda x, out: out is not None)
        buffer = torch.full((2, 11, 9, 16), 12345.)
        destination = buffer[..., 4:, :]
        actual = cell(x, mask, train_size=7, out=destination)
        assert actual is destination and not destination.is_contiguous()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert torch.all(buffer[..., :4, :] == 12345.)
        no_out = cell(x, mask, train_size=7)
        torch.testing.assert_close(no_out, expected, atol=0, rtol=0)


def test_predicate_and_failclosed_installation(monkeypatch):
    assert compat._needs_safe_out(SimpleNamespace(device=SimpleNamespace(type='mps')), object())
    assert not compat._needs_safe_out(SimpleNamespace(device=SimpleNamespace(type='mps')), None)
    assert not compat._needs_safe_out(torch.ones(1), torch.ones(1))
    with pytest.raises(RuntimeError, match='exactly one'):
        compat.install(torch.nn.Module())
    with pytest.raises(RuntimeError, match='exactly one'):
        compat.install(torch.nn.ModuleList([embedding(), embedding()]))
    cell = embedding()
    cell._forward = lambda *a, **k: None
    with pytest.raises(RuntimeError, match='unknown instance'):
        compat.install(cell)
    monkeypatch.setattr(compat, '_PINNED_CLASSES', (
        (compat.KumoCellEmbedding, 'sdm/models/kumo/tabular/cell_embedding.py', 'wrong'),))
    with pytest.raises(RuntimeError, match='exact pinned'):
        compat.install(embedding())
