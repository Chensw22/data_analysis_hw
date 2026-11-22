"""Validate cross-deck residual transfer at aligned relative cabin positions."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART, ROOT, SEEDS
from conditional_view_density_n23 import fit_score
from exaone_validation_data import CATEGORICAL, compact_features
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write

WIDTHS = [.025, .075]


@dataclass
class Decks:
    deck: np.ndarray
    side: np.ndarray
    group: np.ndarray
    solo: np.ndarray
    coords: dict


def layout(raw, row):
    berth = raw.Cabin.str.extract(r'^([A-GT])/(\d+)/(P|S)$')
    deck = berth[0].map(dict(zip('ABCDEFGT', range(8)))).fillna(-1).to_numpy(int)
    side = berth[2].map({'P': 0, 'S': 1}).fillna(-1).to_numpy(int)
    berth_num = pd.to_numeric(berth[1]).astype(float)
    # The extent of each deck and side is observable from unlabeled manifest fields.
    extent = berth_num.groupby([berth[0], berth[2]]).transform('max').clip(lower=1)
    pos = (berth_num/extent).to_numpy()
    group = raw.PassengerId.str.split('_').str[0].astype(int).to_numpy()
    _, inv, size = np.unique(group, return_inverse=True, return_counts=True)
    chronological = (group-group.min())/(group.max()-group.min())
    phase = np.random.default_rng(902).uniform(.15, .85, size=8)
    rotated = (pos + phase[deck]) % 1.
    coords = {'cabin': pos[row], 'rotated': rotated[row], 'booking_order': chronological[row]}
    return Decks(deck[row], side[row], group[row], size[inv][row] == 1, coords)


def triangular_sums(train_x, train_r, train_w, query_x, width):
    # Evaluate linear triangular kernels via left/right range sums, returning
    # donor mass and residual total without forming a passenger-by-passenger array.
    order = np.argsort(train_x)
    x, r, w = train_x[order], train_r[order], train_w[order]
    prefix = np.vstack((np.zeros(4), np.cumsum(np.column_stack((w, w*x, w*r, w*r*x)), axis=0)))
    left = np.searchsorted(x, query_x-width)
    middle = np.searchsorted(x, query_x)
    right = np.searchsorted(x, query_x+width)
    lo, hi = prefix[middle]-prefix[left], prefix[right]-prefix[middle]
    def reduce_mass(i):
        return lo[:, i] + hi[:, i] + (lo[:, i+1]-hi[:, i+1]+query_x*(hi[:, i]-lo[:, i]))/width
    return np.column_stack((np.maximum(reduce_mass(0), 0), reduce_mass(2)))


def field(layout, coordinate, train, query, residual):
    assert not np.intersect1d(layout.group[train], layout.group[query]).size
    _, inv, n = np.unique(layout.group[train], return_inverse=True, return_counts=True)
    weights = 1/n[inv]
    output = np.zeros((len(query), 4))
    mass = np.zeros_like(output)
    for deck in range(8):
        for side in (0, 1):
            local = (layout.deck[query] == deck) & (layout.side[query] == side)
            if not local.any():
                continue
            valid = (layout.side[train] == side) & (layout.deck[train] >= 0)
            separation = np.abs(layout.deck[train]-deck)
            # Both channels exclude own deck; remote donors cannot also be adjacent donors.
            for category, band in enumerate([separation == 1, separation > 1]):
                donors = valid & band
                if donors.any():
                    for i, width in enumerate(WIDTHS):
                        sums = triangular_sums(coordinate[train[donors]], residual[donors], weights[donors],
                                               coordinate[query[local]], width)
                        col = 2*category+i
                        mass[local, col] = sums[:, 0]
                        output[local, col] = sums[:, 1]/(4+sums[:, 0])
    return output, mass


def split(train, group, seed, n=3):
    splitter = GroupKFold(n_splits=n, shuffle=True, random_state=seed)
    for a, b in splitter.split(train, groups=group[train]):
        yield train[a], train[b]


def transfer(x, y, categories, layout, t, q, seed):
    assert not np.intersect1d(layout.group[t], layout.group[q]).size
    direct = fit_score(x[t], y[t], x[q], categories)
    held = np.full(len(y), np.nan)
    for a, b in split(t, layout.group, seed, n=2):
        held[b] = expit(fit_score(x[a], y[a], x[b], categories))
    residual = y[t]-held[t]
    assert np.isfinite(residual).all()
    fields, masses = {}, {}
    for name, coord in layout.coords.items():
        fields[name], masses[name] = field(layout, coord, t, q, residual)
    return direct, fields, masses


def fit_offsets(x, offset, y):
    def objective(w):
        score = offset+x@w
        return np.sum(np.logaddexp(0, score)-y*score) + np.dot(w, w), x.T@(expit(score)-y)+2*w
    fitted = minimize(objective, np.zeros(x.shape[1]), method='L-BFGS-B', jac=True, bounds=[(-4, 4)]*x.shape[1])
    if not fitted.success:
        raise RuntimeError(fitted.message)
    return fitted.x


def run():
    panel = pd.read_csv(ART/'nov23_global_error_inventory.csv')
    row = panel.row.to_numpy()
    y, bh, hh = read_panel()
    np.testing.assert_array_equal(panel.target, y)
    wrong = ~bh.any(0)
    hard = wrong & ~hh.any(axis=(0, 1))
    assert [len(y), wrong.sum(), hard.sum()] == [6954, 1070, 466]
    train = pd.read_csv(ROOT/'data/raw/train.csv', usecols=lambda c: c not in ['Name', 'Transported'])
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=['PassengerId', 'Cabin'])
    boat = layout(pd.concat([train[['PassengerId','Cabin']], test], ignore_index=True), row)
    raw = compact_features(train.iloc[row]).reset_index(drop=True)
    for col in CATEGORICAL:
        raw[col] = pd.Categorical(raw[col]).codes
    x = raw.to_numpy(float)
    categories = np.isin(raw.columns, CATEGORICAL)
    slices = dict(all=np.ones(len(y), bool), solo=boat.solo, solo_positive=boat.solo & y,
                  wrong1070=wrong, hard466=hard, solo_wrong=boat.solo & wrong, solo_hard=boat.solo & hard,
                  solo_missed=boat.solo & wrong & y, solo_hard_missed=boat.solo & hard & y)
    out = ART/'nov23_across_deck_field'
    out.mkdir(exist_ok=True)
    results, all_hits = {}, {}
    for seed in SEEDS:
        with np.load(ART/f'nov23_group_kernel/{seed}.npz') as z:
            fold, base = z['fold'], z['base']
            verify(z,row,y,fold)
        predictors = {'base': base.copy(), 'direct': np.full(len(y), np.nan)}
        for name in boat.coords:
            for anchor in ('direct','base'):
                predictors[name+'_'+anchor] = np.full(len(y), np.nan)
        support, coefs = np.zeros((len(y), 4)), []
        for k in np.unique(fold):
            t, q = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            inner_offset = np.full(len(y), np.nan)
            inner = {key: np.zeros((len(y), 4)) for key in boat.coords}
            for j,(a,b) in enumerate(split(t,boat.group,490+int(k))):
                inner_offset[b], fields, _ = transfer(x,y,categories,boat,a,b,789+j)
                for key, vals in fields.items():
                    inner[key][b] = vals
            direct, outer, mass = transfer(x,y,categories,boat,t,q,689+int(k))
            predictors['direct'][q] = expit(direct)
            support[q] = mass['cabin']
            coeff = {}
            for key in inner:
                w = fit_offsets(inner[key][t], inner_offset[t], y[t])
                coeff[key] = w.tolist()
                adjustment = outer[key] @ w
                predictors[key+'_direct'][q] = expit(direct+adjustment)
                predictors[key+'_base'][q] = expit(logit(base[q])+adjustment)
            coefs.append(coeff)
        assert all(np.isfinite(p).all() for p in predictors.values())
        hits = {k: (p>=.5)==y for k,p in predictors.items()}
        scores = {}
        for name, correct in hits.items():
            anchor = 'direct' if name.endswith('_direct') else 'base'
            ref = hits[anchor]
            scores[name] = {s: dict(correct=int(correct[sel].sum()), fixed=int((~ref & correct & sel).sum()),
                                   harmed=int((ref & ~correct & sel).sum())) for s,sel in slices.items()}
            all_hits.setdefault(name, []).append(correct)
        results[str(seed)] = dict(scores=scores, coefficients=coefs, support={s:int(((support.max(1)>1e-6)&sel).sum())
                                                                             for s,sel in slices.items()})
        np.savez_compressed(out/f'{seed}.npz', row=row,y=y,fold=fold,support=support,solo=boat.solo,
                            p=predictors['cabin_base'],**predictors)
        print(seed, {k: v['all']['correct'] for k,v in scores.items()}, flush=True)
    stable = {k: {s:int(np.all(hit,axis=0)[sel].sum()) for s,sel in slices.items()} for k,hit in all_hits.items()}
    have = boat.deck >= 0
    correlation = float(np.corrcoef(boat.coords['cabin'][have],boat.coords['booking_order'][have])[0,1])
    write(ROOT/'reports/nov23_across_deck_field/result.json', dict(sizes={s:int(sel.sum()) for s,sel in slices.items()},
          cabin_order_correlation=correlation, results=results, stable=stable))


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
