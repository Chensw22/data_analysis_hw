"""Instance-local post-projection CAST attention batching; official vendor untouched.

Only SDPA execution is decomposed. Projected/scaled Q, full K/V, first-head
expansion, explicit score_scale, output projection and cache transaction stay native.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
VENDOR=ROOT/'third_party/exaone'
COMMIT='d149d0f0441053f0cf6b2b45f4b2067a81619cf9'
ATTENTION_SHA='06eed04809a4a306a4af0f9a25e3f67d1175eb560eb3c9a0d23830e08f2a9b85'
MANIFEST_SHA='fcb86550748b20d25e027c954b13a4a03bdaa392cf4901c7342b40362ea14f17'


def verify_vendor():
    if hashlib.sha256((VENDOR/'SOURCE.json').read_bytes()).hexdigest()!=MANIFEST_SHA:
        raise RuntimeError('Pinned EXAONE source manifest changed.')
    manifest=json.loads((VENDOR/'SOURCE.json').read_text())
    if manifest['commit']!=COMMIT:raise RuntimeError('Wrong EXAONE source commit.')
    for name,digest in manifest['files_sha256'].items():
        if hashlib.sha256((VENDOR/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Unmodified official EXAONE source changed: '+name)
    if manifest['files_sha256']['src/exaonetabular/model/attention.py']!=ATTENTION_SHA:
        raise RuntimeError('Attention source guard changed.')
    return manifest


verify_vendor()
sys.path.insert(0,str(VENDOR/'src'))
import exaonetabular
if not Path(exaonetabular.__file__).resolve().is_relative_to((VENDOR/'src').resolve()):
    raise RuntimeError('Different EXAONE runtime already imported.')
import torch
from torch.nn import functional as F
from exaonetabular.model.attention import TensorAttention


def flat_batch_limit(context_length):
    return 1 if context_length>256 else 512


def chunked_sdpa(query,key,value,**options):
    if (query.ndim!=4 or key.ndim!=4 or value.shape!=key.shape or query.shape[:2]!=key.shape[:2]
            or query.shape[-1]!=key.shape[-1]):raise ValueError('Aligned flat batch/head Q/K/V required.')
    # No global backend monkeypatch: choose PyTorch's supported local device backend.
    # Each independent query sees the original complete K/V and softmax denominator.
    output=torch.empty(query.shape,dtype=query.dtype,device=query.device)
    batch=flat_batch_limit(key.shape[-2])
    for start in range(0,len(query),batch):
        for offset in range(0,query.shape[-2],256):
            output[start:start+batch,:,offset:offset+256]=F.scaled_dot_product_attention(
                query[start:start+batch,:,offset:offset+256],key[start:start+batch],value[start:start+batch],**options)
    return output


class ChunkedTensorAttention(TensorAttention):
    # The following native forward is copied verbatim except its SDPA dispatch block.
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        cache_context: bool = False,
        use_cached_context: bool = False,
        reuse_first_context_head: bool = False,
    ) -> torch.Tensor:
        for name, option in (
            ("cache_context", cache_context),
            ("use_cached_context", use_cached_context),
            ("reuse_first_context_head", reuse_first_context_head),
        ):
            if not isinstance(option, bool):
                raise TypeError(f"{name} must be bool")
        if context is not None and not isinstance(context, torch.Tensor):
            raise TypeError("context must be a torch.Tensor or None")
        if cache_context and use_cached_context:
            raise ValueError("cache_context and use_cached_context are mutually exclusive")
        if use_cached_context and context is not None:
            raise ValueError("cached context cannot be combined with explicit context")
        if reuse_first_context_head and context is None and not use_cached_context:
            raise ValueError("head reuse requires explicit or cached context")

        query = self._validate_tensor("query", query)
        query_batch = query.shape[:-2]
        self_projected_key: torch.Tensor | None = None
        self_projected_value: torch.Tensor | None = None
        if context is None and not use_cached_context and not torch.is_grad_enabled():
            projected = F.linear(
                query, self._inference_projection_weight("qkv")
            )
            projected_query, self_projected_key, self_projected_value = projected.split(
                self.embedding_dim, dim=-1
            )
            query_heads = self._as_heads(projected_query)
        else:
            query_heads = self._as_heads(F.linear(query, self.query_weight))
        query_heads.mul_(
            self.head_scale.reshape(
                *((1,) * (query_heads.ndim - 3)), self.head_count, 1, 1
            )
        )

        if use_cached_context:
            if not self.has_cached_context:
                raise RuntimeError("no cached context is available")
            key_heads = self._cached_key
            value_heads = self._cached_value
            assert key_heads is not None and value_heads is not None
            if key_heads.shape[:-3] != query_batch or key_heads.dtype != query.dtype:
                raise RuntimeError("cached context is incompatible with query")
            if key_heads.device != query.device:
                if key_heads.device.type != "cpu":
                    raise RuntimeError("cached context is on an unsupported device")
                key_heads = key_heads.to(
                    device=query.device, non_blocking=key_heads.is_pinned()
                )
                value_heads = value_heads.to(
                    device=query.device, non_blocking=value_heads.is_pinned()
                )
        else:
            selected = query if context is None else self._validate_tensor("context", context)
            if selected.shape[:-2] != query_batch:
                raise ValueError("query and context batch dimensions must match")
            if self_projected_key is not None and self_projected_value is not None:
                key_heads = self._as_heads(self_projected_key)
                value_heads = self._as_heads(self_projected_value)
            else:
                key_heads, value_heads = self._project_context(
                    selected,
                    first_head_only=reuse_first_context_head and not cache_context,
                )
            key_heads.mul_(math.log(key_heads.shape[-2]))

        # Cache the already context-length-scaled K and unmodified V. The same
        # context length is intrinsic to a retained projection, so later calls
        # can reuse the exact scaled bits instead of allocating another K copy.
        # Head reuse remains a per-call view and does not change cache selection.
        cache_key = key_heads if cache_context else None
        cache_value = value_heads if cache_context else None

        if reuse_first_context_head:
            key_heads = key_heads[..., :1, :, :].expand(*key_heads.shape[:-3], self.head_count, *key_heads.shape[-2:])
            value_heads = value_heads[..., :1, :, :].expand(*value_heads.shape[:-3], self.head_count, *value_heads.shape[-2:])

        context_length = key_heads.shape[-2]
        prefix = query_heads.shape[:-3]
        query_length = query_heads.shape[-2]
        flat_query = query_heads.reshape(
            -1, self.head_count, query_length, self.head_width
        )
        flat_key = key_heads.reshape(
            -1, self.head_count, context_length, self.head_width
        )
        flat_value = value_heads.reshape(
            -1, self.head_count, context_length, self.head_width
        )
        attention_options = {"dropout_p": 0.0}
        if self.score_scale is not None:
            attention_options["scale"] = self.score_scale
        attended = chunked_sdpa(flat_query, flat_key, flat_value, **attention_options)
        attended = attended.reshape(
            *prefix, self.head_count, query_length, self.head_width
        )
        del flat_query, flat_key, flat_value, query_heads
        if not cache_context:
            del key_heads, value_heads
        joined = attended.transpose(-3, -2).flatten(-2)
        output = F.linear(joined, self.output_weight)

        if cache_context:
            # Assign only after all computations succeed, replacing the pair together.
            assert cache_key is not None and cache_value is not None
            self._cached_key, self._cached_value = cache_key.detach(), cache_value.detach()
        return output


def install_local(model):
    verify_vendor();names=[]
    for name,module in model.named_modules():
        if type(module) in [TensorAttention,ChunkedTensorAttention]:
            module.__class__=ChunkedTensorAttention;names.append(name)
    if not names:raise RuntimeError('No CAST attention modules found.')
    return dict(modules=names,source_sha256=ATTENTION_SHA,instance_local=True,
        q_chunk=256,flat_batch_long_k=1,flat_batch_short_k=512,long_k_boundary=256,
        full_key_value_sequence=True,original_projection_ssmax_output_cache=True,
        backend='native_default_device_SDPA_in_locked_PyTorch')
