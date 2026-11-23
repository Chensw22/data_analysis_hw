"""Learn passenger similarity separately from the labels used to vote on each query."""
import json

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn import functional as F

from broad_tree_field_dividends_n23 import hard_cohorts
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, SEED, data
from matched_consensus_errors_n23 import roster_fields


def embedding(x, group, label, train, focus, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = nn.Sequential(nn.Linear(x.shape[1], 64), nn.SiLU(), nn.Linear(64, 12))
    optimizer = torch.optim.AdamW(net.parameters(), lr=.002)
    for _ in range(24):
        for batch in np.array_split(rng.permutation(train), 16):
            z = F.normalize(net(x[batch]))
            affinity = 12 * z @ z.T
            valid = group[batch, None] != group[None, batch]
            same = label[batch, None] == label[None, batch]
            assert (valid & same).any(1).all()
            log_hit = (affinity.masked_fill(~(valid & same), -torch.inf).logsumexp(1)
                       - affinity.masked_fill(~valid, -torch.inf).logsumexp(1))
            weight = (1-log_hit.exp()).square() if focus else 1.
            loss = -(weight*log_hit).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        return F.normalize(net(x))


def vote(z, support, query, y):
    labels = y[support].float()
    return torch.cat([((12 * z[rows] @ z[support].T).softmax(1) @ labels)
                       for rows in np.array_split(query, 5)]).numpy()


def predict_neighbors(inputs, groups, y, t, q, f):
    assert not np.isin(groups[t], groups[q]).any()
    g = torch.tensor(groups.astype(int))
    labels = torch.tensor(y)
    predictions = {name: np.zeros(len(q)) for name in ['fixed', 'learned', 'focus']}
    halves = GroupKFold(2, shuffle=True, random_state=SEED+int(f)).split(t, groups=groups[t])
    for half, (a, b) in enumerate(halves):
        fit, library = t[a], t[b]
        assert not np.isin(groups[fit], groups[library]).any()
        x = (inputs-inputs[fit].mean(0))/inputs[fit].std(0).clamp_min(.5)
        for name in predictions:
            z = (F.normalize(x) if name == 'fixed' else
                 embedding(x, g, labels, fit, name == 'focus', SEED+2*int(f)+half))
            predictions[name] += .5*vote(z, library, q, labels)
    return predictions


def roster_tensor(groups):
    own, peer = roster_fields(np.arange(len(groups)), groups.astype(int))
    return torch.tensor(np.column_stack([own, peer]), dtype=torch.float32)


def train(y, groups, fold):
    inputs = roster_tensor(groups)
    predictions = {name: np.zeros(len(y)) for name in ['fixed', 'learned', 'focus']}
    for f in np.unique(fold):
        t, q = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        for name, p in predict_neighbors(inputs, groups, y, t, q, f).items():
            predictions[name][q] = p
        print('completed outer fold', int(f), flush=True)
    return predictions


def run():
    raw, y, groups, fold = data()
    columns = []
    for name in (*FAMILIES, 'mitra96'):
        with np.load(OUT / f'{name}.npz') as bank:
            np.testing.assert_array_equal(bank['passenger'], raw.PassengerId)
            np.testing.assert_array_equal(bank['fold'], fold)
            np.testing.assert_array_equal(bank['y'], y)
            columns.append(bank['p'])
    scores = np.column_stack(columns)
    mitra = scores[:, -1]
    base = mitra >= .5
    common = ((scores >= .5) != y[:, None]).all(1)
    _, inverse, count = np.unique(groups, return_inverse=True, return_counts=True)
    strong_negative = (count[inverse] == 1) & (scores <= .25).all(1)
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as bank:
        dev = np.sort(bank['development'])
    hard = np.zeros(len(y), bool)
    hard[dev] = hard_cohorts(dev, y[dev])[2]
    probs = train(y, groups, fold)
    def summarize(pred):
        fixed = (base != y) & (pred == y)
        harmed = (base == y) & (pred != y)
        delta = fixed.astype(int)-harmed
        return dict(accuracy=float((pred == y).mean()), fixed=int(fixed.sum()), harmed=int(harmed.sum()),
                    shared_fixed=int((common & fixed).sum()), historical_fixed=int((hard & fixed).sum()),
                    high_conf_solo_negative_fixed=int((fixed & strong_negative).sum()),
                    high_conf_solo_negative_harmed=int((harmed & strong_negative).sum()),
                    fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)])
    results = {}
    for name, p in probs.items():
        assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
        results[name] = {'standalone': summarize(p >= .5),
                         'mean_with_mitra': summarize(p+mitra >= 1)}
    dest = ROOT / 'artifacts/nov23_full_roster_metric_neighbors'
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest / 'oof.npz', passenger=raw.PassengerId.to_numpy(str), y=y, fold=fold, **probs)
    report = ROOT / 'reports/nov23_matched_learning_curve/metric_neighbors.json'
    report.write_text(json.dumps(results, indent=2)+'\n')
    print(report.read_text())


if __name__ == '__main__':
    torch.set_num_threads(2)
    run()
