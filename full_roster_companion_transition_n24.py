"""Fit two-sided changes to passenger probabilities from observed companion fields."""
import json

import numpy as np
import torch
from scipy.special import expit, logit
from torch import nn
from torch.nn import functional as F

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_maxout_n24 import fields


def booking_index(groups):
    _, codes = np.unique(groups, return_inverse=True)
    orders = np.split(np.argsort(codes, kind='stable'), np.cumsum(np.bincount(codes))[:-1])
    empty = len(groups)
    peers = np.full((empty, max(map(len, orders))-1), empty, dtype=int)
    for order in orders:
        for person in order:
            peers[person, :len(order)-1] = order[order != person]
    return codes, orders, peers


def swapped_index(q, codes, orders, peers, seed):
    substitute = peers[q].copy()
    rng = np.random.default_rng(seed)
    present = np.unique(codes[q])
    sizes = np.array([len(orders[i]) for i in present])
    replacement = np.arange(len(codes))
    replaced = np.zeros(len(codes), dtype=bool)
    for size in np.unique(sizes):
        candidates = rng.permutation(present[sizes == size])
        if size == 1 or len(candidates) < 2:
            continue
        for src, dest in zip(candidates, np.roll(candidates, 1), strict=True):
            replacement[orders[src]] = orders[dest]
            replaced[orders[src]] = True
    substitute[:] = peers[replacement[q]]
    valid = substitute < len(codes)
    np.testing.assert_array_equal(valid, peers[q] < len(codes))
    row, col = np.where(valid & replaced[q, None])
    assert (codes[q[row]] != codes[substitute[row, col]]).all()
    return substitute, replaced[q]


class CompanionModel(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(width, 40), nn.SiLU(), nn.Linear(40, 20), nn.SiLU())
        self.own = nn.Linear(20, 1)
        self.pair = nn.Sequential(nn.Linear(60, 24), nn.SiLU(), nn.Linear(24, 2))
        nn.init.constant_(self.pair[-1].bias, -3.)

    def predict(self, x, people, neighbors, solo_only=False):
        focal = self.embed(x[people])
        start = torch.sigmoid(self.own(focal)).flatten()
        if solo_only:
            return start, start
        other = self.embed(x[neighbors])
        left = focal[:, None, :].expand_as(other)
        rate = F.softplus(self.pair(torch.cat([left, other, left*other], dim=-1)))
        rate = (rate*(neighbors != len(x)-1).unsqueeze(-1)).sum(1)
        total = rate.sum(1)
        moved = -torch.expm1(-total)
        positive = rate[:, 0]/total.clamp_min(1e-9)
        return start + moved*(positive-start), start


def fit(x, y, t, q, peers, alternate, seed):
    train_x = ((x-x[t].mean(0))/x[t].std(0).clip(.2)).clip(-10, 10).astype('float32')
    x = torch.from_numpy(np.vstack([train_x, np.zeros_like(train_x[:1])]))
    truth = torch.tensor(y, dtype=torch.float32)
    neighbors = torch.as_tensor(peers)
    other = torch.as_tensor(alternate)
    columns = []
    for solo in (True, False):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = CompanionModel(x.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.025)
        snapshot = []
        for epoch in range(34):
            for batch in np.array_split(rng.permutation(t), int(np.ceil(len(t)/512))):
                p, _ = model.predict(x, batch, neighbors[batch], solo)
                opt.zero_grad()
                F.binary_cross_entropy(p.clamp(1e-6,1-1e-6), truth[batch]).backward()
                opt.step()
            if epoch in (23, 28, 33):
                with torch.inference_mode():
                    real, personal = model.predict(x, q, neighbors[q], solo)
                    if solo:
                        snapshot.append(real.numpy()[:,None])
                    else:
                        swapped, _ = model.predict(x, q, other)
                        snapshot.append(torch.stack([real, personal, swapped], dim=1).numpy())
        columns.append(np.mean(snapshot, axis=0))
    result = np.column_stack(columns)
    singles = peers[q, 0] == len(y)
    np.testing.assert_array_equal(result[singles, 1], result[singles, 2])
    np.testing.assert_array_equal(result[singles, 1], result[singles, 3])
    return result


def protocol(x, y, group, rows, folds, booking, tag):
    codes, orders, peers = booking
    scores = np.empty((len(rows), 4))
    changed = np.empty(len(rows), dtype=bool)
    out = ROOT/'artifacts/nov24_companion_transition'
    out.mkdir(exist_ok=True)
    for f in np.unique(folds):
        held = folds == f
        q = rows[held]
        t = rows[~held & ~np.isin(group[rows], group[q])]
        assert not np.isin(group[t], group[q]).any()
        alternate, changed[held] = swapped_index(q, codes, orders, peers, 509+int(f))
        path = out/f'{tag}_{f}.npz'
        if not path.exists():
            result = fit(x, y, t, q, peers, alternate, 632)
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, train=t, query=q, train_labels=y[t], alternate=alternate, scores=result)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for k, v in [('train', t), ('query', q), ('train_labels', y[t]), ('alternate', alternate)]:
                np.testing.assert_array_equal(z[k], v)
            scores[held] = z['scores']
        print(tag, int(f), flush=True)
    assert np.isfinite(scores).all()
    return scores, changed


def decisions(p, base, anchor):
    out = {name: p[:, j] >= .5 for j, name in enumerate(('personal', 'real', 'peer_removed', 'swapped'))}
    for j, name in [(0, 'personal'), (1, 'real'), (3, 'swapped')]:
        out[name+'_quarter'] = np.where((base >= .5) == anchor, .75*base+.25*p[:, j] >= .5, anchor)
    odds = logit(p.clip(1e-6, 1-1e-6))
    for j, name in [(2, 'removed'), (3, 'swapped')]:
        correction = odds[:,1]-odds[:,j]
        out['peer_delta_'+name] = np.where((base >= .5) == anchor,
            expit(logit(base.clip(1e-6,1-1e-6))+.25*correction) >= .5, anchor)
    return out


def run():
    _, oy, og, rows, ofold, _, champion, _, _, _, _, old, _ = aligned()
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    ref = latest_reference(y, group, fold, names, half, full)
    gate, _ = gate_probabilities(names, half, full, y, group, fold)
    np.testing.assert_array_equal(gate >= .5, ref)
    x, booking = fields(raw), booking_index(group)
    a, _ = protocol(x, y, group, rows, ofold, booking, 'historical')
    b, swapped = protocol(x, y, group, np.arange(len(y)), fold, booking, 'full')
    original, current = decisions(a, old[:,0], champion), decisions(b, gate, ref)
    multi = booking[2][:,0] < len(y)
    unseen = ~np.isin(group, og)
    common = ((full >= .5) != y[:,None]).all(1)
    hard = common[rows] & (champion != oy)
    cabin = raw.Cabin.str.fullmatch(r'[^/]+/\d+/[PS]', na=False)
    orders_per_cabin = raw.loc[cabin].assign(booking=group[cabin]).groupby('Cabin').booking.nunique()
    cross_order_cabin = raw.Cabin.isin(orders_per_cabin[orders_per_cabin > 1].index).to_numpy()
    reach = {}
    for name, mask in [('all', common), ('false_negative', common & y), ('false_positive', common & ~y)]:
        reach[name] = dict(total=int(mask.sum()), solo=int((mask & ~multi).sum()),
            booking_peer=int((mask & multi).sum()), other_order_same_cabin=int((mask & cross_order_cabin).sum()),
            either=int((mask & (multi | cross_order_cabin)).sum()))
    real_right = (b[:,1] >= .5) == y
    substituted_right = (b[:,3] >= .5) == y
    report = dict(multimember_rows=int(multi.sum()), swapped_rows=int(swapped.sum()), reach=reach,
        correct_with_real_only=int((swapped & real_right & ~substituted_right).sum()),
        correct_with_substitution_only=int((swapped & substituted_right & ~real_right).sum()), decisions={})
    for name, p in current.items():
        summary = {}
        for mask, key in [(np.ones(len(y),bool), 'full'), (multi, 'group'), (~multi, 'solo'), (unseen, 'unseen')]:
            summary[key] = stats(p[mask], ref[mask], y[mask], group[mask], fold[mask])
        summary['historical'] = stats(original[name], champion, oy, og, ofold)
        summary['common_group'] = int((common & multi & (p == y)).sum())
        summary['common_solo'] = int((common & ~multi & (p == y)).sum())
        summary['persistent_both'] = int((hard & (original[name] == oy) & (p[rows] == oy)).sum())
        report['decisions'][name] = summary
        print(name, [summary[k]['net'] for k in ('historical', 'full', 'unseen')], summary['common_group'], flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/companion_transition.json'
    path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    torch.set_num_threads(1)
    run()
