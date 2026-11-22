"""Group-held-out classification by category-specific cross-view density ratios."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART, ROOT, SEEDS
from exaone_validation_data import CATEGORICAL, compact_features
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write

BILLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


def fit_score(train, target, query, categorical, weight=None):
    model = HistGradientBoostingClassifier(
        max_iter=105, max_leaf_nodes=13, min_samples_leaf=25, l2_regularization=2,
        learning_rate=.07, early_stopping=False, categorical_features=categorical, random_state=711,
    )
    model.fit(train, target, sample_weight=weight)
    return model.decision_function(query)


def unrelated_order(groups, rng):
    # Within random booking blocks, a cyclic rotation by at least the largest
    # booking's size pairs every person with a different booking, with no repeats.
    values, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    rank = rng.permutation(len(values))[inv]
    order = np.lexsort((rng.random(len(groups)), rank))
    shift = rng.integers(counts.max(), len(order) - counts.max() + 1)
    donors = np.empty_like(order)
    donors[order] = np.roll(order, shift)
    assert (groups != groups[donors]).all()
    return donors


def pair_codes(a, b):
    # Real and permuted records have exactly identical per-column marginals.
    # Explicit joint levels make pure interactions learnable even when a tree
    # cannot find a positive-gain first split on any one variable.
    aa = a.copy()
    for col, thresholds in [(3, [6, 13, 18, 25, 40, 60]), (6, [100, 300, 600, 900, 1300, 1700])]:
        aa[:, col] = np.where(np.isnan(a[:, col]), 7, np.digitize(a[:, col], thresholds))
    aa = aa.astype('int16')
    complete = np.nan_to_num(b)
    cuts = np.log1p([.01, 50, 250, 700, 1500])
    bills = np.where(np.isnan(b), 6, np.digitize(b, cuts))
    totals = np.digitize(np.expm1(complete).sum(axis=1), [0.01, 150, 600, 1500, 3000])
    dominant = np.where(complete.max(axis=1) == 0, 5, complete.argmax(axis=1))
    bb = np.column_stack((bills, totals, dominant, (complete > 0).sum(axis=1))).astype('int16')
    pairs = (aa[:, :, None] * 8 + bb[:, None, :]).reshape(len(a), -1)
    return np.column_stack((aa, bb, pairs))


def fit_joint_codes(train, target, query, weight=None):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    design = encoder.fit_transform(train)
    model = LogisticRegression(C=.15, max_iter=250, solver='lbfgs', tol=1e-5)
    model.fit(design, target, sample_weight=weight)
    return model.decision_function(encoder.transform(query))


def ratio(a, b, groups, query_codes, rng):
    positive = pair_codes(a, b)
    noise = [pair_codes(a, b[unrelated_order(groups, rng)]) for _ in range(2)]
    train = np.vstack((positive, *noise))
    n = len(a)
    return fit_joint_codes(train, np.repeat([1, 0, 0], n), query_codes,
                           weight=np.repeat([1., .5, .5], n))


def raw_scores(a, b, y, groups, categorical_a, train, query, seed, direct=False):
    rng = np.random.default_rng(seed)
    full = np.column_stack((a, b))
    categorical = np.r_[categorical_a, np.zeros(b.shape[1], dtype=bool)]
    prior = logit(y[train].mean())
    sa = fit_score(a[train], y[train], a[query], categorical_a)
    sb = fit_score(b[train], y[train], b[query], None)
    query_codes = pair_codes(a[query], b[query])
    heads = []
    for split in (y[train], rng.permutation(y[train])):
        contrasts = []
        for label in (False, True):
            rows = train[split == label]
            contrasts.append(ratio(a[rows], b[rows], groups[rows], query_codes, rng))
        heads.append(contrasts[1] - contrasts[0])
    result = np.column_stack((sa + sb - prior, *heads))
    direct_query = fit_score(full[train], y[train], full[query], categorical) if direct else None
    direct_codes = fit_joint_codes(pair_codes(a[train], b[train]), y[train], query_codes) if direct else None
    return result, direct_query, direct_codes


def calibrate(z, y, columns):
    # L2 towards the density-identity coefficients. Limit the noisy estimate's
    # weight to [0, 2], including the independent model as a boundary case.
    x = np.column_stack((np.ones(len(z)), z[:, columns]))
    target = np.r_[0., np.ones(len(columns))]
    penalty = np.r_[.001, .002, np.full(len(columns) - 1, .002)]

    def objective(w):
        s = x @ w
        loss = np.mean(np.logaddexp(0, s) - y * s) + .5 * np.dot(penalty, (w - target) ** 2)
        gradient = (x.T @ (expit(s) - y)) / len(y) + penalty * (w - target)
        return loss, gradient

    result = minimize(objective, target, jac=True, bounds=[(-2, 2), *[(0, 2)] * len(columns)],
                      method='L-BFGS-B')
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def summarize(p, y, base, wrong, hard, fold):
    ok = (p >= .5) == y
    ref = (base >= .5) == y
    return dict(accuracy=float(ok.mean()), log_loss=float(log_loss(y, p)), fixed=int((~ref & ok).sum()),
                harmed=int((ref & ~ok).sum()), hard1070=int(ok[wrong].sum()), hard466=int(ok[hard].sum()),
                fold_net=[int(ok[fold == k].sum() - ref[fold == k].sum()) for k in np.unique(fold)])


def run():
    panel = pd.read_csv(ART / 'nov23_global_error_inventory.csv')
    row = panel.row.to_numpy()
    y, bh, hh = read_panel()
    np.testing.assert_array_equal(panel.target, y)
    wrong = ~bh.any(0)
    hard = wrong & ~hh.any(axis=(0, 1))
    assert [len(y), wrong.sum(), hard.sum()] == [6954, 1070, 466]
    frame = pd.read_csv(ROOT / 'data/raw/train.csv', usecols=lambda c: c not in ['Transported', 'Name']).iloc[row]
    groups = frame.PassengerId.str.split('_').str[0].to_numpy()
    x = compact_features(frame).reset_index(drop=True)
    a_fields = [col for col in x if col not in BILLS]
    # Only dictionary names, without labels or frequencies, define the nominal codes.
    for col in CATEGORICAL:
        x[col] = pd.Categorical(x[col]).codes
    a = x[a_fields].to_numpy(float)
    b = np.log1p(x[BILLS].to_numpy(float))
    assert a_fields[3] == 'Age' and a_fields[6] == 'CabinNumber'
    categorical = np.isin(a_fields, CATEGORICAL)
    variants = {'independent': [0], 'conditional': [0, 1], 'sham': [0, 2]}
    out = ART / 'nov23_conditional_view_density'
    out.mkdir(exist_ok=True)
    summary, hits = {}, {}
    for seed in SEEDS:
        with np.load(ART / f'nov23_group_kernel/{seed}.npz') as z:
            fold, base = z['fold'], z['base']
            verify(z, row, y, fold)
        p = {name: np.full(len(y), np.nan) for name in [*variants, 'raw_independent', 'raw_conditional', 'direct', 'direct_pair_codes']}
        weights = []
        for k in np.unique(fold):
            t, v = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            assert not np.intersect1d(groups[t], groups[v]).size
            inner_scores = np.full((len(t), 3), np.nan)
            splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=457 + int(k))
            for j, (u, w) in enumerate(splitter.split(t, y[t], groups[t])):
                assert not np.intersect1d(groups[t[u]], groups[t[w]]).size
                inner_scores[w], _, _ = raw_scores(a, b, y, groups, categorical, t[u], t[w], 791 + j)
            assert np.isfinite(inner_scores).all()
            outer, direct, direct_codes = raw_scores(a, b, y, groups, categorical, t, v, 698, direct=True)
            p['direct'][v] = expit(direct)
            p['direct_pair_codes'][v] = expit(direct_codes)
            p['raw_independent'][v] = expit(outer[:, 0])
            p['raw_conditional'][v] = expit(outer[:, 0] + outer[:, 1])
            fitted = {}
            for name, cols in variants.items():
                w = calibrate(inner_scores, y[t], cols)
                p[name][v] = expit(w[0] + outer[:, cols] @ w[1:])
                fitted[name] = w.tolist()
            weights.append(fitted)
            print(seed, int(k), 'weights', fitted['conditional'], flush=True)
        assert all(np.isfinite(v).all() for v in p.values())
        np.savez_compressed(out / f'{seed}.npz', row=row, y=y, fold=fold, base=base, p=p['conditional'], **p)
        all_p = {'base': base, **p, **{f'{n}_blend': .75*base + .25*p[n] for n in [*variants, 'direct', 'direct_pair_codes']}}
        results = {name: summarize(val, y, base, wrong, hard, fold) for name, val in all_p.items()}
        for name, val in all_p.items():
            hits.setdefault(name, []).append((val >= .5) == y)
        summary[str(seed)] = dict(weights=weights, scores=results)
        print(seed, [(n, round(results[n]['accuracy'], 4), results[n]['hard466']) for n in all_p], flush=True)
    stable = {n: {'hard1070': int(np.all(v, axis=0)[wrong].sum()), 'hard466': int(np.all(v, axis=0)[hard].sum())}
              for n, v in hits.items()}
    write(ROOT / 'reports/nov23_conditional_view_density/result.json', dict(results=summary, stable=stable))


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
