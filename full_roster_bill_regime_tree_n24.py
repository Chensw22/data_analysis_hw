"""Fit a hierarchical mixture of spending regressions with registration-dependent weights."""
import json
from math import ceil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_zero_truncated_bills_n24 import inputs

MODES = ('pooled', 'constant_weights', 'registration_weights')
OUT = ROOT/'artifacts/nov24_bill_regime_tree'


def spending_features(bill):
    amounts = np.nan_to_num(bill)
    values = np.log1p(amounts)
    total = amounts.sum(1)
    shares = amounts / np.maximum(total[:, None], 1)
    return np.column_stack([values, amounts > 0, np.isnan(bill), shares, np.log1p(total),
                            np.maximum(values - 4, 0), np.maximum(values - 6, 0)])


class Regimes(nn.Module):
    def __init__(self, context_dim, bill_dim, mode):
        super().__init__()
        self.mode = mode
        self.context = nn.Linear(context_dim, 1)
        leaves = 1 if mode == 'pooled' else 8
        self.leaves = nn.Linear(bill_dim, leaves)
        if mode == 'registration_weights':
            self.node = nn.Linear(context_dim, leaves-1)
        elif mode == 'constant_weights':
            self.node = nn.Parameter(torch.zeros(leaves-1))

    def weights(self, context):
        if self.mode == 'pooled':
            return torch.ones_like(context[:, :1])
        logits = self.node(context) if self.mode == 'registration_weights' else self.node.expand(len(context), -1)
        splits = torch.sigmoid(logits)
        weights = torch.ones_like(splits[:, :1])
        for width in (1, 2, 4):
            right = splits[:, width-1:2*width-1]
            weights = torch.stack([weights*(1-right), weights*right], dim=2).flatten(1)
        return weights

    def forward(self, context, bills):
        conditional = torch.sigmoid(self.context(context) + self.leaves(bills))
        return (self.weights(context)*conditional).sum(1)


def standardize(x, t):
    return torch.tensor(((x-x[t].mean(0))/x[t].std(0).clip(.2)).clip(-10, 10), dtype=torch.float32)


def fit(context, bills, y, t, q, seed):
    c, b = [standardize(x, t) for x in (context, bills)]
    truth = torch.tensor(y, dtype=torch.float32)
    predictions = []
    for mode in MODES:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = Regimes(c.shape[1], b.shape[1], mode)
        opt = torch.optim.AdamW(model.parameters(), lr=.007, weight_decay=.03)
        snapshots = []
        for epoch in range(42):
            for rows in np.array_split(rng.permutation(t), ceil(len(t)/512)):
                p = model(c[rows], b[rows]).clamp(1e-6, 1-1e-6)
                opt.zero_grad()
                F.binary_cross_entropy(p, truth[rows]).backward()
                opt.step()
            if epoch in (29, 35, 41):
                with torch.inference_mode():
                    weights = model.weights(c[q])
                    torch.testing.assert_close(weights.sum(1), torch.ones(len(q)))
                    snapshots.append(model(c[q], b[q]).numpy())
        predictions.append(np.mean(snapshots, axis=0))
    return np.column_stack(predictions)


def protocol(c, b, y, group, rows, fold, tag):
    OUT.mkdir(exist_ok=True)
    p = np.empty((len(rows), len(MODES)))
    for f in np.unique(fold):
        mask = fold == f
        q = rows[mask]
        t = rows[~mask & ~np.isin(group[rows], group[q])]
        assert not np.isin(group[t], group[q]).any()
        path = OUT/f'{tag}_{f}.npz'
        if not path.exists():
            scores = fit(c, b, y, t, q, 590)
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, t=t, q=q, truth=y[t], p=scores)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for key, expected in [('t', t), ('q', q), ('truth', y[t])]:
                np.testing.assert_array_equal(z[key], expected)
            p[mask] = z['p']
        print(tag, int(f), flush=True)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return p


def run():
    _, old_y, old_group, rows, oldfold, _, champion, _, _, _, _, old, _ = aligned()
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    reference = latest_reference(y, group, fold, names, half, full)
    gate, _ = gate_probabilities(names, half, full, y, group, fold)
    np.testing.assert_array_equal(gate >= .5, reference)
    c, bill, _ = inputs(raw)
    b = spending_features(bill)
    a = protocol(c, b, y, group, rows, oldfold, 'historical')
    d = protocol(c, b, y, group, np.arange(len(y)), fold, 'full')
    unseen = ~np.isin(group, old_group)
    common = ((full >= .5) != y[:, None]).all(1)
    hard = common[rows] & (champion != old_y)
    results = {}
    for j, mode in enumerate(MODES):
        results[mode] = {}
        for label, weight in [('direct', 1.), ('quarter', .25)]:
            pa = (1-weight)*old[:, 0] + weight*a[:, j] >= .5
            if weight < 1:
                pa = np.where((old[:, 0] >= .5) == champion, pa, champion)
            pb = (1-weight)*gate + weight*d[:, j] >= .5
            record = dict(historical=stats(pa, champion, old_y, old_group, oldfold),
                          full=stats(pb, reference, y, group, fold),
                          unseen=stats(pb[unseen], reference[unseen], y[unseen], group[unseen], fold[unseen]),
                          common_correct=int((common & (pb == y)).sum()),
                          repeated_correct=int((hard & (pa == old_y) & (pb[rows] == old_y)).sum()))
            results[mode][label] = record
            print(mode, label, [record[k]['net'] for k in ('historical', 'full', 'unseen')],
                  record['common_correct'], record['repeated_correct'], flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/bill_regime_tree.json'
    path.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    torch.set_num_threads(1)
    run()
