"""TabR-S core adapted from Yandex's MIT-licensed TabR implementation.

Upstream: yandex-research/tabular-dl-tabr, commit
17baa9082506f8e7a0f8d11bb1e08212926a1507, bin/tabr.py.
See LICENSE and upstream_tabr.py.txt in this directory. This adaptation preserves
the default simple architecture, label embedding, key differences, residual value
aggregation and candidate gradients. It replaces Faiss with exact torch top-k,
accepts already concatenated numeric/one-hot input, and excludes self by row ID.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TabRModel(nn.Module):
    def __init__(self, n_features, d_main=265, context_size=96,
                 context_dropout=0.38920071545944357,
                 dropout0=0.38852797479169876, dropout1=0.0):
        super().__init__()
        self.context_size = int(context_size)
        self.linear = nn.Linear(n_features, d_main)
        # Default TabR-S has no encoder residual blocks or mixer normalization.
        self.label_encoder = nn.Embedding(2, d_main)
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(nn.Linear(d_main, 2*d_main), nn.ReLU(),
                               nn.Dropout(dropout0), nn.Linear(2*d_main, d_main, bias=False))
        self.dropout = nn.Dropout(context_dropout)
        self.predictor = nn.Sequential(nn.LayerNorm(d_main), nn.Linear(d_main, 2*d_main),
            nn.ReLU(), nn.Dropout(dropout0), nn.Linear(2*d_main, d_main), nn.Dropout(dropout1))
        self.head = nn.Sequential(nn.LayerNorm(d_main), nn.ReLU(), nn.Linear(d_main, 1))
        nn.init.uniform_(self.label_encoder.weight, -1., 1.)

    def encode(self, x):
        representation = self.linear(x)
        return representation, self.K(representation)

    @torch.no_grad()
    def retrieve(self, query_keys, library_keys, query_ids, library_ids):
        # Only discrete neighbor selection is detached. Selected keys and values
        # are gathered from the original differentiable tensors in forward().
        distance = (query_keys.square().sum(1, keepdim=True)
                    - 2*query_keys@library_keys.T + library_keys.square().sum(1)[None])
        n_available = len(library_keys)
        if query_ids is not None:
            self_mask = query_ids[:, None].eq(library_ids[None, :])
            distance.masked_fill_(self_mask, torch.inf)
            n_available = int((~self_mask).sum(1).min().item())
        k = min(self.context_size, n_available)
        if k < 1:
            raise ValueError('Retrieval requires at least one non-self training neighbor.')
        return distance.topk(k, dim=1, largest=False, sorted=True).indices

    def forward(self, query, library, library_y, library_ids, query_ids=None):
        if library_y.dtype != torch.long:
            raise TypeError('Class labels for retrieval must be integer embedding indices.')
        if len(library) != len(library_y) or len(library) != len(library_ids):
            raise ValueError('Retrieval features, labels and row IDs must align.')
        x, keys = self.encode(query)
        _, candidate_keys = self.encode(library)
        indices = self.retrieve(keys, candidate_keys, query_ids, library_ids)
        context_keys = candidate_keys[indices]
        similarities = (-keys.square().sum(-1, keepdim=True)
            + 2*(keys[:, None, :]*context_keys).sum(-1) - context_keys.square().sum(-1))
        weights = self.dropout(F.softmax(similarities, dim=-1))
        values = self.label_encoder(library_y[indices]) + self.T(keys[:, None]-context_keys)
        x = x + (weights[:, :, None]*values).sum(1)
        x = x + self.predictor(x)
        return self.head(x).squeeze(-1)
