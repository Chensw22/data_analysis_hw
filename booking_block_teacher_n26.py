"""Detect swapped manifest blocks and distil three genuinely nested teachers into a shared encoder."""
import argparse
import copy
import subprocess
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from full_roster_booking_validation_n23 import data, ROOT

OUT = ROOT / 'artifacts/nov26_booking_block_teacher'
FEES = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
VARIANTS = ['scratch', 'swap_labels', 'swap_mean', 'swap_individual']
TEACHERS = ['extra_trees', 'hist_boost', 'additive_logistic']


def design(raw, group):
    cabin = raw.Cabin.str.split('/', expand=True)

    def category(s, levels):
        return np.column_stack([s.eq(v).fillna(False) for v in levels] + [~s.isin(levels)])

    def smooth(series, scale, knots):
        x = series.fillna(0).to_numpy(float) / scale
        return np.column_stack([series.isna(), series.eq(0), x, *[np.maximum(x-k, 0) for k in knots]])

    route = np.column_stack([category(raw.HomePlanet, ['Earth','Europa','Mars']),
                             category(raw.Destination, ['TRAPPIST-1e','55 Cancri e','PSO J318.5-22']),
                             category(raw.VIP, [True,False])])
    berth = np.column_stack([category(cabin[0], list('ABCDEFGT')), category(cabin[2], ['P','S']),
                             smooth(pd.to_numeric(cabin[1]), 1800., [.2,.4,.6,.8])])
    order_size = raw.groupby(group).PassengerId.transform('size').to_numpy()
    age = np.column_stack([smooth(raw.Age, 80., [.15,.25,.4,.55,.7]), np.log(order_size)])
    bill = np.column_stack([smooth(np.log1p(raw[name]), 9., [.35,.55,.7,.85]) for name in FEES])
    blocks = [route, berth, age, category(raw.CryoSleep, [True,False]), bill]
    return (np.column_stack(blocks).astype(np.float32),
            torch.tensor(np.repeat(np.arange(len(blocks)), [b.shape[1] for b in blocks])))


def split_teachers(x, y, g, t, q, seed):
    inner = GroupKFold(3, shuffle=True, random_state=seed)
    soft = np.full((len(t),3), np.nan)
    outer_single = np.full((3,len(q),3), np.nan)
    inner_fold = np.full(len(t),-1)
    for k,(fit,apply) in enumerate(inner.split(t, groups=g[t])):
        assert (inner_fold[apply]==-1).all()
        inner_fold[apply] = k
        assert not np.isin(g[t[fit]], np.r_[g[t[apply]],g[q]]).any()
        models = [ExtraTreesClassifier(n_estimators=96, min_samples_leaf=5, max_features=.8,
                                       random_state=seed, n_jobs=1),
                  HistGradientBoostingClassifier(max_iter=145, learning_rate=.065, max_leaf_nodes=16,
                                                 l2_regularization=2., early_stopping=False, random_state=seed),
                  LogisticRegression(C=2., max_iter=400)]
        for j, model in enumerate(models):
            model.fit(x[t[fit]],y[t[fit]])
            soft[apply,j] = model.predict_proba(x[t[apply]])[:,1]
            outer_single[k,:,j] = model.predict_proba(x[q])[:,1]
    assert (inner_fold>=0).all() and np.isfinite(soft).all() and np.isfinite(outer_single).all()
    return soft, outer_single, inner_fold


def frozen_teacher_mean(outer_single):
    divided = outer_single/3
    # The logistic teacher emits float32. The original scoring loop divided by three
    # before accumulating into float64, so replay that rounding exactly.
    divided[:,:,2] = outer_single[:,:,2].astype(np.float32)/3
    return divided.sum(0)


def teacher_panel(x, y, g, t, q, seed):
    soft, outer_single, _ = split_teachers(x,y,g,t,q,seed)
    return torch.tensor(soft, dtype=torch.float32), frozen_teacher_mean(outer_single)


def cache_teachers():
    raw,y,g,fold = data()
    x,_ = design(raw,g)
    dest = OUT/'nested'
    dest.mkdir(parents=True,exist_ok=True)
    for k in range(8):
        t,q = np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        soft,outer_single,inner_fold = split_teachers(x,y,g,t,q,1384+k)
        with np.load(OUT/f'fold{k}.npz') as frozen:
            for key,expected in [('t',t),('q',q),('train_y',y[t])]:
                np.testing.assert_array_equal(frozen[key],expected)
            reference = np.column_stack([frozen[name] for name in TEACHERS])
        # Frozen accumulation divided each slice by three before summing.
        np.testing.assert_array_equal(frozen_teacher_mean(outer_single),reference)
        unrounded = outer_single.mean(0)
        np.testing.assert_allclose(unrounded,reference,atol=5e-8,rtol=0)
        np.testing.assert_array_equal(unrounded>=.5,reference>=.5)
        partial = dest/f'fold{k}.partial.npz'
        np.savez_compressed(partial,t=t,q=q,train_y=y[t],soft=soft,outer_single=outer_single,
                            inner_fold=inner_fold,teacher_names=np.array(TEACHERS))
        partial.replace(dest/f'fold{k}.npz')
        print('verified nested teacher cache',k,flush=True)


class Network(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(width,96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(.1),
                                     nn.Linear(96,64), nn.LayerNorm(64), nn.SiLU())
        self.detect = nn.Linear(64,5)
        self.outcome = nn.Linear(64,1)
        self.distil = nn.Linear(64,3)


def train_swap(model, x, groups, field, seed):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=.0013, weight_decay=.01)
    for _ in range(18):
        for start in np.array_split(rng.permutation(len(x)), int(np.ceil(len(x)/256))):
            donor = rng.integers(len(x), size=(len(start),5))
            same = groups[donor] == groups[start,None]
            while same.any():
                donor[same] = rng.integers(len(x), size=same.sum())
                same = groups[donor] == groups[start,None]
            swapped = torch.tensor(rng.random((len(start),5)) < .35)
            new = x[start].clone()
            detected = swapped.clone()
            for j in range(5):
                cols = field == j
                original = new[:,cols]
                other = x[donor[:,j]][:,cols]
                detected[:,j] &= (other != original).any(1)
                new[:,cols] = torch.where(swapped[:,j,None], other, original)
            loss = F.binary_cross_entropy_with_logits(model.detect(model.encoder(new)), detected.float())
            opt.zero_grad()
            loss.backward()
            opt.step()


def supervised(model, x, y, soft, query, seed):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=.0011, weight_decay=.01)
    snapshots = []
    for epoch in range(28):
        model.train()
        for b in torch.randperm(len(x), generator=rng).split(256):
            hidden = model.encoder(x[b])
            loss = F.binary_cross_entropy_with_logits(model.outcome(hidden).ravel(), y[b])
            if soft is not None:
                loss += .75 * F.binary_cross_entropy_with_logits(model.distil(hidden), soft[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch in (17,22,27):
            model.eval()
            with torch.no_grad():
                snapshots.append(model.outcome(model.encoder(query)).ravel().sigmoid().numpy())
    return np.mean(snapshots, axis=0)


def run():
    raw,y,g,fold = data()
    x,field = design(raw,g)
    features = torch.tensor(x)
    target = torch.tensor(y, dtype=torch.float32)
    OUT.mkdir(exist_ok=True)
    result = {key: np.full(len(y),np.nan) for key in VARIANTS+TEACHERS}
    for k in range(8):
        t,q = np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.isin(g[t],g[q]).any()
        cache = OUT / f'fold{k}.npz'
        if cache.exists():
            with np.load(cache) as z:
                for name,value in [('t',t),('q',q),('train_y',y[t])]:
                    np.testing.assert_array_equal(z[name],value)
                for name in result:
                    result[name][q] = z[name]
            continue
        seed = 1384+k
        soft,teacher = teacher_panel(x,y,g,t,q,seed)
        for j,name in enumerate(TEACHERS):
            result[name][q] = teacher[:,j]
        torch.manual_seed(seed)
        scratch = Network(x.shape[1])
        swapped = copy.deepcopy(scratch)
        train_swap(swapped, features[t], g[t], field, seed)
        mean = soft.mean(1,keepdim=True).expand_as(soft)
        for name,net,teacher_target in zip(VARIANTS,[scratch,swapped,swapped,swapped],
                                           [None,None,mean,soft],strict=True):
            result[name][q] = supervised(copy.deepcopy(net),features[t],target[t],teacher_target,features[q],seed+97)
        np.savez_compressed(cache,t=t,q=q,train_y=y[t],**{name:values[q] for name,values in result.items()})
        print('outer fold',k+1,'/',8,{name:round(np.mean((v[q]>=.5)==y[q]),4) for name,v in result.items()},flush=True)
    assert all(np.isfinite(v).all() for v in result.values())
    # The archive imports tree libraries in the project environment; neural training uses .venv-neural.
    subprocess.run([str(ROOT/'.venv/bin/python'), str(ROOT/'evaluate_booking_block_teacher_n26.py')], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teachers-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        if args.teachers_only:
            cache_teachers()
        else:
            run()
