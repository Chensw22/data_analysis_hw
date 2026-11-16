"""Instance-local workaround for PyTorch MPS strided matmul out= (#204931).

Only the pinned Kumo cell projection is affected. Its official implementation
still computes Fourier features, bias and missingness in their original order;
MPS copies the completed result into the caller's buffer afterwards.
"""
import hashlib
import inspect
from pathlib import Path
from types import MethodType

from sdm.models.kumo.tabular.cell_embedding import CellEmbedding as KumoCellEmbedding
from sdm.models.tabfm.cell_embedding import CellEmbedding as TabFMCellEmbedding

COMPAT_ID = 'kumo_cell_embedding_mps_allocate_then_copy_v1'
UPSTREAM_COMMIT = '909d1d49a68e4d6bf71a468f9f813869f62ba970'
_ROOT = Path(__file__).resolve().parent
_PINNED_CLASSES = (
    (KumoCellEmbedding, 'sdm/models/kumo/tabular/cell_embedding.py',
     '38f2021479bd85afc116f6245a6f2162d03fb530ac171d3d7da7def0a1d8c307'),
    (TabFMCellEmbedding, 'sdm/models/tabfm/cell_embedding.py',
     '42f7abc6a428a941f5df407b005a2c16b396bb6c2d022ed2db76fae2641afd75'),
)
_ORIGINAL_FORWARD = KumoCellEmbedding._forward
_ORIGINAL_PARENT_FORWARD = TabFMCellEmbedding._forward


def verify_upstream():
    for cls, relative, expected in _PINNED_CLASSES:
        path = _ROOT/'third_party/kumo'/relative
        if (Path(inspect.getfile(cls)).resolve() != path.resolve()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise RuntimeError('Kumo MPS compatibility requires the exact pinned upstream source.')
    if (KumoCellEmbedding._forward is not _ORIGINAL_FORWARD
            or TabFMCellEmbedding._forward is not _ORIGINAL_PARENT_FORWARD):
        raise RuntimeError('Upstream class methods have already been replaced.')


def _needs_safe_out(x, out):
    return x.device.type == 'mps' and out is not None


def _forward(self, x, freq, weight, bias, *, missing, out=None):
    if not _needs_safe_out(x, out):
        return _ORIGINAL_FORWARD(self, x, freq, weight, bias, missing=missing, out=out)
    # Call the Kumo override, including its nan_lin addition, before copying.
    # No global torch patch, precision change or alternate attention backend.
    result = _ORIGINAL_FORWARD(self, x, freq, weight, bias, missing=missing, out=None)
    out.copy_(result)
    return out


def install(model):
    """Bind exactly one official instance; repeated installation is a no-op."""
    verify_upstream()
    embeddings = [m for m in model.modules() if isinstance(m, KumoCellEmbedding)]
    if len(embeddings) != 1 or type(embeddings[0]) is not KumoCellEmbedding:
        raise RuntimeError('Expected exactly one unmodified Kumo CellEmbedding instance.')
    embedding = embeddings[0]
    existing = embedding.__dict__.get('_forward')
    if existing is not None:
        if (getattr(existing, '__func__', None) is not _forward
                or getattr(existing, '__self__', None) is not embedding):
            raise RuntimeError('CellEmbedding already has an unknown instance override.')
    else:
        embedding._forward = MethodType(_forward, embedding)
    return 1
