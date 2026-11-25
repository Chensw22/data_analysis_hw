"""Fit a two-state booking likelihood and measure both individual rescues and joint prediction."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/nov25_shared_booking_state'
DEST = ROOT / 'reports/nov23_matched_learning_curve/shared_booking_state.json'
MODES = ('independent', 'shared', 'shuffled')
FEES = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


def roster_data():
    raw = pd.read_csv(ROOT / 'data/raw/train.csv', usecols=lambda c: c != 'Name')
    groups = raw.PassengerId.str.split('_').str[0].to_numpy()
    age = raw.Age.fillna(28).to_numpy()
    cabin = raw.Cabin.str.split('/', expand=True)
    amounts = raw[FEES].fillna(0).to_numpy()
    positive = amounts > 0
    total = amounts.sum(axis=1, keepdims=True)
    categories = raw[['CryoSleep', 'VIP', 'HomePlanet', 'Destination']].astype('string').fillna('?')
    categories = categories.assign(deck=cabin[0].fillna('?'), side=cabin[2].fillna('?'))
    person = np.column_stack([pd.get_dummies(categories, dtype=float), age / 40, age < 13, age < 18,
                              raw.Age.isna(), pd.to_numeric(cabin[1]).fillna(500) / 900, raw.Cabin.isna(),
                              np.log1p(amounts) / 7, raw[FEES].isna(), positive, np.log1p(total) / 8,
                              amounts / total.clip(1), positive.sum(1) / 5])
    features = pd.DataFrame(person).groupby(groups)
    n = features[0].transform('size').to_numpy()
    x = np.column_stack([person, features.transform('mean'), np.log(n)])
    return raw, groups, x.astype('float32')


def partitions(raw):
    out = {}
    for tag, path, rk in [('historical','nov25_historical_nine/historical','rows'),
                          ('eight','nov25_production_anchor_cpu/oof','row')]:
        with np.load(ROOT / 'artifacts' / (path + '.npz')) as z:
            rows, folds = z[rk], z['fold']
            np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[rows])
            np.testing.assert_array_equal(z['y'], raw.Transported.iloc[rows])
        out[tag] = (rows, folds)
    return out


def fit(features, labels, train, query, groups, mode, seed):
    import torch
    from torch import nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(features.shape[1], 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 2))
    opt = torch.optim.AdamW(net.parameters(), lr=.004, weight_decay=.03)
    center, scale = features[train].mean(0), features[train].std(0).clip(.15)
    x = torch.as_tensor(((features - center) / scale).clip(-6, 6))
    _, codes = np.unique(groups[train], return_inverse=True)
    if mode == 'shuffled':
        codes = np.random.default_rng(seed).permutation(codes)
    elif mode == 'independent':
        codes = np.arange(len(train))
    idx = torch.as_tensor(codes)
    n_groups = int(codes.max() + 1)
    xt = x[train]
    sign = torch.tensor(np.where(labels[train], 1, -1), dtype=torch.float32)[:, None]

    def logits(x_batch):
        mean, loading = net(x_batch).unbind(1)
        return torch.stack([mean - loading, mean + loading], dim=1)

    for _ in range(100):
        per_person = F.logsigmoid(sign * logits(xt))
        joint = xt.new_zeros((n_groups, 2)).index_add_(0, idx, per_person)
        loss = -(joint.logsumexp(1) - np.log(2)).sum() / len(train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return logits(x[query]).numpy()


def train_models():
    import torch
    torch.set_num_threads(2)
    raw, groups, features = roster_data()
    y = raw.Transported.to_numpy()
    OUT.mkdir(exist_ok=True)
    for tag, (rows, fold) in partitions(raw).items():
        for f in np.unique(fold):
            q = rows[fold == f]
            t = rows[(fold != f) & ~np.isin(groups[rows], groups[q])]
            assert not np.intersect1d(groups[t], groups[q]).size
            path = OUT / f'{tag}_{f}.npz'
            if not path.exists():
                states = np.stack([[fit(features, y, t, q, groups, mode, seed) for seed in (913, 1519)]
                                   for mode in MODES])
                assert np.isfinite(states).all() and states.shape == (len(MODES), 2, len(q), 2)
                with path.with_suffix('.tmp').open('wb') as handle:
                    np.savez_compressed(handle, t=t, q=q, ids=raw.PassengerId.iloc[q].to_numpy(str),
                                        labels=y[t], states=states)
                path.with_suffix('.tmp').replace(path)
            print(tag, int(f), flush=True)


def joint_loss(states, truth, group):
    # Integrate each fitted network's shared state before mixing the two seeds.
    _, codes = np.unique(group, return_inverse=True)
    logp = -np.logaddexp(0., -states * np.where(truth, 1, -1)[None, :, None])
    joint = np.zeros((len(states), codes.max() + 1, 2))
    for k in range(len(states)):
        np.add.at(joint[k], codes, logp[k])
    collapsed = np.logaddexp(joint[:, :, 0], joint[:, :, 1]) - np.log(2)
    return -float((np.logaddexp.reduce(collapsed, axis=0) - np.log(len(states))).sum())


def evaluate():
    from full_roster_conditional_pool_n23 import data
    from full_roster_six_failure_archive_n24 import prediction_sets
    raw, y, groups, f8 = data()
    labeled = raw.assign(Transported=y)
    splits = partitions(labeled)
    rows_h, fh = splits['historical']
    _, panels = prediction_sets(raw, y, groups, f8, rows_h, fh)
    output = {}
    for tag, (rows, fold) in splits.items():
        if tag == 'historical':
            with np.load(ROOT / 'artifacts/nov25_historical_nine/historical.npz') as z:
                inc, prev = z['nine'], z['v131']
            with np.load(ROOT / 'artifacts/nov25_mitra_fp32_head/oof.npz') as z:
                np.testing.assert_array_equal(z['row'], rows)
                native = z['native']
            eligible = np.ones(len(rows), bool)
            panel = panels[1]
        else:
            with np.load(ROOT / 'artifacts/nov25_production_anchor_cpu/oof.npz') as z:
                inc, native, eligible = z['certified_nine'], z['v96'], z['certified_v131_equals_v96']
                prev = native >= .5
            panel = panels[0]
        inc = inc.astype(bool)
        votes = np.column_stack(panel) >= .5
        agree = eligible & (votes == inc[:, None]).all(1)
        free = eligible & ((native >= .5) == prev) & (prev == inc)
        truth = y[rows]
        prob = np.empty((len(rows), len(MODES)))
        losses = np.zeros(len(MODES))
        has_training_companion = np.zeros(len(rows), bool)
        for f in np.unique(fold):
            use = fold == f
            if tag == 'historical':
                packet = json.loads((ROOT / 'artifacts/v108_exact_oof' / f'fold{f}_packet.json').read_text())
                np.testing.assert_array_equal(packet['query_ids'], rows[use])
                has_training_companion[use] = np.isin(groups[rows[use]], groups[packet['train_ids']])
            q = rows[use]
            t = rows[(fold != f) & ~np.isin(groups[rows], groups[q])]
            with np.load(OUT / f'{tag}_{f}.npz') as z:
                for key, value in [('t', t), ('q', q), ('labels', y[t]), ('ids', raw.PassengerId.iloc[q])]:
                    np.testing.assert_array_equal(z[key], value)
                states = z['states'].astype(float)
            for j in range(len(MODES)):
                prob[use, j] = expit(states[j]).mean(axis=(0, 2))
                losses[j] += joint_loss(states[j], truth[use], groups[q])
        counts = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()[rows]
        def effect(alt):
            d = np.where(free & (alt != inc), np.where(inc == truth, -1, 1), 0)
            return dict(fixed=int((d > 0).sum()), harmed=int((d < 0).sum()),
                        common_fixed=int((d[agree] > 0).sum()), common_harmed=int((d[agree] < 0).sum()),
                        solo_net=int(d[counts == 1].sum()), party_net=int(d[counts > 1].sum()),
                        unseen_common_fixed=int(((d > 0) & agree & ~has_training_companion).sum()),
                        unseen_common_harmed=int(((d < 0) & agree & ~has_training_companion).sum()),
                        unseen_order_fixed=int(((d > 0) & ~has_training_companion).sum()),
                        unseen_order_harmed=int(((d < 0) & ~has_training_companion).sum()),
                        fold_net=[int(d[fold == k].sum()) for k in np.unique(fold)])
        analysis = {}
        for j, name in enumerate(MODES):
            marginal_loss = log_loss(truth, prob[:, j])
            item = dict(raw_accuracy=float(((prob[:, j] >= .5) == truth).mean()),
                        marginal_logloss=marginal_loss,
                        group_logloss_per_person=marginal_loss if name == 'independent' else losses[j] / len(rows))
            for w in (1., .25):
                item[str(w)] = effect((w * prob[:, j] + (1 - w) * native) >= .5)
            if name != 'independent':
                raw_shift = logit(prob[:, j].clip(.001, .999)) - logit(prob[:, 0].clip(.001, .999))
                item['odds_shift'] = effect(logit(native.clip(.001, .999)) + raw_shift >= 0)
            analysis[name] = item
        output[tag] = dict(rows=len(rows), baseline_verified=int(eligible.sum()), common_errors=int((agree & (inc != truth)).sum()),
                           common_correct=int((agree & (inc == truth)).sum()), certified_party_rows=int((eligible & (counts > 1)).sum()),
                           native_training_companion=int(has_training_companion.sum()), models=analysis)
        print(tag, output[tag], flush=True)
    DEST.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='store_true')
    if parser.parse_args().report:
        evaluate()
    else:
        train_models()
