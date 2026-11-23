"""Learn individual passenger coordinates from unlabeled pairs of travelling companions."""
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.special import logit
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_boosting_path_n23 import calibrated
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize
from class_analogue_experts_n23 import design
from exaone_validation_data import compact_features
from released_correction_audit_n23 import development

REPORT = ROOT / 'reports/nov23_booking_coordinates'
OUT = ART / 'nov23_booking_coordinates'


def load_roster():
    read = lambda part: pd.read_csv(ROOT / f'data/raw/{part}.csv',
                                    usecols=lambda col: col not in ('Transported', 'Name'))
    a, b = read('train'), read('test')
    roster = pd.concat([a, b], ignore_index=True)
    g = roster.PassengerId.str.split('_').str[0].astype(int).to_numpy()
    size = pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
    roster_size = len(a)
    audit = dict(test_rows=len(b), test_group_overlap=int(np.isin(g[roster_size:], g[:roster_size]).sum()),
                 test_known_cabin_overlap=int(b.Cabin.isin(a.Cabin.dropna()).sum()),
                 test_solo=int((size[roster_size:] == 1).sum()))
    return compact_features(roster), g, size, audit


def edge_table(groups, ids):
    table = pd.DataFrame({'g': groups[ids], 'i': ids})
    pairs = table.merge(table, on='g', suffixes=('1', '2')).query('i1 != i2')
    weights = 1 / pairs.groupby('g').g.transform('size').to_numpy()
    return pairs.i1.to_numpy(), pairs.i2.to_numpy(), weights / weights.sum()


def exchange_partner_vectors(frame, ids, size, groups, seed):
    # Reassign passenger profiles to fixed group slots within comparable original strata.
    # Rare strata can contribute unchanged slots; report this instead of dropping those people.
    bins = pd.DataFrame(dict(planet=frame.HomePlanet.iloc[ids].to_numpy(),
                             deck=frame.Deck.iloc[ids].to_numpy(), size=size[ids].clip(max=4)))
    mapping = np.arange(len(frame))
    rng = np.random.default_rng(seed)
    for cell in bins.groupby(list(bins)).indices.values():
        members = ids[cell]
        mapping[members] = rng.permutation(members)
    return mapping


def coordinates(x, edges):
    i, j, w = edges
    mean = w @ x[i]
    a, b = x[i] - mean, x[j] - mean
    marginal = (a.T * w) @ a
    paired = (a.T * w) @ b
    # Shared and complementary directions both carry pair identity.
    vals, vec = eigh((paired + paired.T) / 2, marginal + .04 * np.eye(x.shape[1]))
    best = np.argsort(-abs(vals))[:12]
    return (x - mean) @ vec[:, best] * np.sqrt(abs(vals[best])), vals[best]


def predict_neighbors(z, train, query, y):
    results = []
    for batch in np.array_split(query, max(1, int(np.ceil(len(query) / 160)))):
        dist = cdist(z[batch], z[train], 'sqeuclidean')
        # Use a continuous kernel, including ties instead of an arbitrary subset of them.
        scale = np.partition(dist, 47, axis=1)[:, 47:48].clip(1e-5)
        weights = np.exp(-6 * (dist - dist.min(1, keepdims=True)) / scale)
        results.extend((weights @ y + .5) / (weights.sum(1) + 1.))
    return np.array(results)


def pair_auc(z, edges, shuffled):
    i, j, w = edges
    if not len(i):
        return None
    def score(a, b):
        return -np.square(z[a] - z[b]).sum(1)
    return float(roc_auc_score(np.repeat([1, 0], len(i)), np.r_[score(i, j), score(shuffled[i], shuffled[j])],
                               sample_weight=np.tile(w, 2)))


def run():
    row, y, masks = historical()
    labels, release, _ = development()
    np.testing.assert_array_equal(y, labels)
    frame, groups, size, audit = load_roster()
    report, history, references = {}, {}, []
    pay = strata(frame.iloc[row])
    masks = {**masks, 'paid_transported': masks['repeat'] & (pay == 2) & y,
             'zero_stayed': masks['repeat'] & (pay == 0) & ~y,
             'solo_repeat': masks['repeat'] & (size[row] == 1)}
    OUT.mkdir(exist_ok=True)
    for seed in [*map(str, SEEDS), 'v131']:
        with np.load(ART / f'nov23_broad_partition_evidence/{seed}.npz') as cache:
            np.testing.assert_array_equal(cache['row'], row)
            np.testing.assert_array_equal(cache['y'], y)
            fold, base, ref = cache['fold'], cache['base'], cache['ref']
        np.testing.assert_array_equal(ref, release[131] if seed == 'v131' else base >= .5)
        raw = {name: np.zeros(len(y)) for name in ['individual', 'real', 'shuffled']}
        checks = []
        for f in np.unique(fold):
            q = np.flatnonzero(fold == f)
            excluded = np.isin(groups, groups[row[q]])
            t = np.flatnonzero((fold != f) & ~excluded[row])
            pool = np.flatnonzero(~excluded)
            verify = np.flatnonzero(excluded)
            train = row[t]
            x, _ = design(frame, pool)
            train_edges = edge_table(groups, pool)
            validation_edges = edge_table(groups, verify)
            rng_map = exchange_partner_vectors(frame, pool, size, groups, 45 + int(f))
            query_map = exchange_partner_vectors(frame, verify, size, groups, 87 + int(f))
            i, j, w = train_edges
            real, vals = coordinates(x, train_edges)
            fake, _ = coordinates(x, (rng_map[i], rng_map[j], w))
            _, vec = eigh(np.cov(x[pool].T))
            ordinary = x @ vec[:, -12:]
            variants = dict(individual=ordinary, real=real, shuffled=fake)
            assert not np.intersect1d(groups[train], groups[row[q]]).size
            checks.append(dict(axes=vals.tolist(), training_groups=int(np.unique(groups[i]).size),
                               pseudo_pair_same_booking=float(w @ (groups[rng_map[i]] == groups[rng_map[j]])),
                               pair_auc={k:pair_auc(v, validation_edges, query_map) for k, v in variants.items()}))
            for k, z in variants.items():
                raw[k][q] = predict_neighbors(z, train, row[q], y[t])
        predictions = {k: p >= .5 for k, p in raw.items()}
        free = ref == (base >= .5)
        for k in raw:
            vote = ref.copy()
            features = np.column_stack([logit(base.clip(.002,.998)), logit(raw['individual'].clip(.002,.998))])
            if k != 'individual':
                features = np.column_stack([features, logit(raw[k].clip(.002, .998))])
            for f in np.unique(fold):
                ix = np.flatnonzero((fold == f) & free)
                for a, b in GroupKFold(3, shuffle=True, random_state=925).split(ix, groups=groups[row[ix]]):
                    t, q = ix[a], ix[b]
                    vote[q] = calibrated(features[t], features[q], y, t) >= .5
            predictions['combined_' + k] = vote
        anchor = ref == y
        if seed != 'v131':
            references.append(anchor)
        metrics = {}
        for k, vote in predictions.items():
            hit = vote == y
            metrics[k] = dict(acc=float(hit.mean()), fixed=int((hit & ~anchor).sum()), harmed=int((anchor & ~hit).sum()),
                              solo_acc=float(hit[size[row] == 1].mean()), multi_acc=float(hit[size[row] > 1].mean()))
            if seed != 'v131':
                history.setdefault(k, []).append(hit)
        report[seed] = dict(models=metrics, verification=checks)
        np.savez_compressed(OUT / f'{seed}.npz', row=row, y=y, base=base, fold=fold, ref=ref, **raw)
        print(seed, metrics['combined_real'], flush=True)
    write(REPORT / 'results.json', dict(coverage=audit, plans=report, grouped={
        k:summarize(np.stack(h), np.stack(references), masks) for k, h in history.items()}))


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
