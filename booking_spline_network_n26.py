"""Train a two-layer network of learned linear splines and matched controls.

.venv-tabpfn/bin/python booking_spline_network_n26.py train
DYLD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs .venv/bin/python booking_spline_network_n26.py report
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT, data

OUT = ROOT / 'artifacts/nov26_booking_spline_network'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/booking_spline_network.json'
FEES = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
CONFIG = dict(seeds=[431, 731], width=24, epochs=48, batch=384, lr=.005, weight_decay=.006,
              second_difference_penalty=.001, hidden_knots=9, positive_quantiles=9)
METHODS = ['additive', 'silu', 'learned_spline']


def basis(raw, groups, t):
    cabin = raw.Cabin.str.split('/', expand=True)
    fields = [raw.HomePlanet, raw.Destination, raw.CryoSleep, raw.VIP, cabin[0], cabin[2]]
    block = []
    for col in fields:
        strings = col.astype('string').fillna('__missing__')
        levels = strings.iloc[t].unique()
        block.extend(strings.eq(value).to_numpy(bool) for value in levels)
        block.append(~strings.isin(levels).to_numpy())
    numerical = {'Age': raw.Age, 'cabin_number': pd.to_numeric(cabin[1]),
                 'order_size': raw.groupby(groups).PassengerId.transform('size')}
    bills = raw[FEES]
    numerical.update({name: bills[name] for name in FEES})
    numerical['total_observed_bill'] = bills.sum(axis=1, min_count=1)
    numerical['active_services'] = bills.gt(0).sum(axis=1)
    for name, column in numerical.items():
        x = column.to_numpy(float)
        obs = np.isfinite(x)
        block.append(~obs)
        if name in FEES or name == 'total_observed_bill':
            block.append(x == 0)
            obs &= x > 0
            x = np.log1p(x)
        knots = np.unique(np.quantile(x[t][obs[t]], np.linspace(0, 1, CONFIG['positive_quantiles'])))
        rank = np.interp(np.nan_to_num(x, nan=knots[0]), knots, np.arange(len(knots)))
        block.extend(np.maximum(1 - abs(rank-j), 0)*obs for j in range(len(knots)))
    return np.ascontiguousarray(np.column_stack(block), dtype=np.float32)


def neural_model(d, mode):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(d, CONFIG['width'])
            self.readout = nn.Linear(CONFIG['width'], 1)
            if mode == 'learned_spline':
                self.heights = nn.Parameter(torch.zeros(CONFIG['width'], CONFIG['hidden_knots']))
                self.register_buffer('knots', torch.linspace(-3., 3., CONFIG['hidden_knots']))

        def forward(self, x):
            # Each coefficient of a first-layer numeric hat learns its own edge height.
            hidden = self.first(x)
            if mode == 'silu':
                return self.readout(F.silu(hidden)).squeeze(1)
            step = self.knots[1]-self.knots[0]
            hat = (1-((hidden.clamp(-3, 3).unsqueeze(-1)-self.knots)/step).abs()).clamp_min(0)
            return self.readout(hidden).squeeze(1) + (hat*self.heights).sum((1, 2))

        def penalty(self):
            return self.heights.diff(n=2).square().mean() if mode == 'learned_spline' else 0

    return Net()


def fit_neural(x, y, t, q, seed, mode):
    import torch
    from torch.nn import functional as F
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed+79)
    train, query = torch.from_numpy(x[t]), torch.from_numpy(x[q])
    target = torch.from_numpy(y[t].astype(np.float32))
    net = neural_model(x.shape[1], mode)
    opt = torch.optim.AdamW(net.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    batches = (len(t)+CONFIG['batch']-1)//CONFIG['batch']
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CONFIG['epochs']*batches, eta_min=.0004)
    for _ in range(CONFIG['epochs']):
        for batch in torch.randperm(len(t), generator=gen).split(CONFIG['batch']):
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(net(train[batch]), target[batch])
            (loss+CONFIG['second_difference_penalty']*net.penalty()).backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        return torch.sigmoid(net(query)).numpy()


def splits(raw, g, fold):
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        h = np.sort(z['development'])
    new = np.flatnonzero(~np.isin(g, g[h]))
    assert len(np.unique(h)) == 6954 and len(new) == 1044
    for k in range(8):
        yield f'fold{k}', np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    yield 'history_to_new', h, new


def train():
    import torch
    torch.set_num_threads(1)
    raw, y, g, fold = data()
    OUT.mkdir(exist_ok=True)
    for tag, t, q in splits(raw, g, fold):
        assert not np.isin(g[t], g[q]).any()
        x = basis(raw, g, t)
        packet = {'additive': LogisticRegression(C=1., max_iter=1000).fit(x[t], y[t]).predict_proba(x[q])[:, 1]}
        for mode in METHODS[1:]:
            packet[mode] = np.stack([fit_neural(x, y, t, q, seed, mode) for seed in CONFIG['seeds']])
        np.savez_compressed(OUT/(tag+'.npz'), t=t, q=q, passenger=raw.PassengerId.iloc[q].to_numpy(str),
                            train_y=y[t], features=x.shape[1], torch_version=str(torch.__version__), **packet)
        print(tag, {j:round(float(((p.reshape(-1,len(q)).mean(0)>=.5)==y[q]).mean()),5)
                    for j,p in packet.items()}, flush=True)


def report():
    from hashlib import sha256
    from sklearn.metrics import log_loss
    from distinct_challenger_error_reach_n26 import population
    from tabnet_attention_transfer_n26 import collect
    raw, y, g, fold = data()
    scores = {}
    files = {}
    for tag, t, q in splits(raw, g, fold):
        path = OUT/(tag+'.npz')
        with np.load(path) as z:
            for key, expected in [('t', t), ('q', q), ('passenger', raw.PassengerId.iloc[q]), ('train_y', y[t])]:
                np.testing.assert_array_equal(z[key], expected)
            assert not np.isin(g[t], g[q]).any()
            desc = files[tag] = dict(training_rows=len(t), training_bookings=len(np.unique(g[t])),
                  query_rows=len(q), features=int(z['features']), sha256=sha256(path.read_bytes()).hexdigest())
            desc['torch_version'] = str(z['torch_version'])
            group = 'history_to_new' if tag == 'history_to_new' else 'eight_fold'
            packet = scores.setdefault(group, {})
            for method in METHODS:
                p = z[method].reshape(-1, len(q))
                full = packet.setdefault(method, np.full((p.shape[0],len(y)), np.nan))
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and np.isnan(full[:,q]).all()
                full[:,q] = p
    groups = {'eight_fold': np.arange(len(y)), 'history_to_new': q}
    metrics = {}
    for tag, ids in groups.items():
        result = metrics[tag] = {}
        for method, p in scores[tag].items():
            avg = p[:,ids].mean(0)
            win = (avg>=.5)==y[ids]
            result[method] = dict(rows=len(ids), correct=int(win.sum()), accuracy=float(win.mean()),
                log_loss=float(log_loss(y[ids],avg)), seed_accuracy=((p[:,ids]>=.5)==y[ids]).mean(1).tolist(),
                fold_accuracy=[float(win[fold[ids]==k].mean()) for k in range(8)])
    network, truth, groups_check, fold_check, _ = collect()
    for a,b in [(y,truth),(g,groups_check),(fold,fold_check)]:
        np.testing.assert_array_equal(a,b)
    extended = population()
    np.testing.assert_array_equal(y, extended[1])
    np.testing.assert_array_equal(fold, extended[3])
    scores_by_parent = {}
    for name, source in [('full','eight_fold'), ('historical_to_new','history_to_new')]:
        segments = network[name]
        ids, same, parent = (np.concatenate([s[j] for s in segments]) for j in range(3))
        queued_edit = same & np.concatenate([s[3]['actual_and_scores'] for s in segments])
        queued = parent ^ queued_edit
        assert np.isin(ids,groups[source]).all()
        pred = {m:a[:,ids] for m,a in scores[source].items()}
        votes = {m:(p.mean(0)>=.5) for m,p in pred.items()}
        scope = {'known_all':np.ones(len(ids),bool),'ten_unanimous':same,'untouched_by_queued_tabnet':same&~queued_edit}
        if name == 'full':
            np.testing.assert_array_equal(extended[5][ids], same)
            scope['no_archived_challenger_countervote'] = same & (extended[6][ids]==0)
        res = scores_by_parent[name] = {}
        for title, mask in scope.items():
            ref = queued if title == 'known_all' else parent
            error = ref!=y[ids]
            alt = votes['learned_spline']!=ref
            support = np.where(ref[None,:], 1-pred['learned_spline'], pred['learned_spline'])
            edits = {m:v!=ref for m,v in votes.items()}
            edits.update(spline_alone_vs_two_controls=alt & (votes['silu']==ref) & (votes['additive']==ref),
                         spline_and_silu=alt & (votes['silu']!=ref),
                         spline_opposed_both_seeds=(support>.5).all(0),
                         spline_opposed_both_seeds_at_80pct=(support>=.8).all(0))
            if name=='full' and title=='ten_unanimous':
                assert mask.sum()==7891 and (error&mask).sum()==1181
            def counts(a):
                selected = mask&a
                gain = np.where(selected, np.where(error,1,-1),0)
                return dict(fixed=int((gain==1).sum()),harmed=int((gain==-1).sum()),net=int(gain.sum()),
                            fold_net=[int(gain[fold[ids]==k].sum()) for k in range(8)])
            res[title] = dict(rows=int(mask.sum()), errors=int((mask&error).sum()), changes={m:counts(a) for m,a in edits.items()})
    contents = dict(config=CONFIG, files=files, accuracy=metrics, scopes=scores_by_parent,
        train_inputs='Official training roster only. Home planet, destination, cryosleep, VIP, deck, side, cabin number, age, five bills, their observed sum and active count, booking size. Order size is the count on the unlabeled 8693-row training roster, including members outside the historical subset. No names or identifiers enter the design. Each outer training set determines its categories, positive numeric knots and first-layer design.',
        architecture='One positive-quantile hat basis per numeric input. Linear coefficients give each incoming numeric-hidden edge its own spline. Each hidden unit then has a learned nine-knot output spline plus a linear term. SiLU replaces this second layer with a fixed activation under identical inputs and width. The logistic control uses the same first-layer basis without hidden units. Both networks train on the same batches and fixed two seeds; probability mean is evaluated at 0.5.',
        earlier_implementations='broad_adaptive_manifest_splines_n23.py selects products of input basis terms; solo_conditional_splines_n25.py uses predefined interactions and linear logistic coefficients. Neither learns a spline of a hidden projection.',
        scope='Eight outer folds train from all other orders. Transfer trains once per recipe on exactly 6954 historic labels to predict 1044 passengers whose orders never appeared there. No tuning using their labels. In 1033/1044 cases the existing competing releases can be reconstructed. That queued comparator uses some full-data outer-fold base predictions. These protocols and passengers overlap; this is not an independent new dataset.',
        denominators='known_all compares each model to the queued TabNet version. ten_unanimous compares to all ten prior versions on common decisions. untouched_by_queued_tabnet retains only unanimous cases the double-confirmed TabNet did not flip. The zero-archived block is the previously enumerated bank and three TabNet recipes; its membership is not a model error oracle.',
        caches=str(OUT.relative_to(ROOT)))
    REPORT.write_text(json.dumps(contents, indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
    for name,r in scores_by_parent.items():
        for key,v in r.items():
            print(name,key,v['changes']['learned_spline'],flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'report'])
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.action == 'train':
            train()
        else:
            report()


if __name__ == '__main__':
    main()
