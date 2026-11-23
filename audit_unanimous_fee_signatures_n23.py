"""Check tie probabilities and profile the unanimous solo payers by observable bills."""
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import rankdata

from audit_stable_rescue_placebo_n23 import family_scores
from broad_geographic_transfer_n23 import BILLS, load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from fee_identity_assignment_n23 import observed_assignment


def verify_likelihood():
    amounts = np.array([[23, 0, 0, -1, -1], [1, 2, 3, 4, 5], [-1]*5, [0]*5, [2, 2, 3, 3, 4]])
    rng = np.random.default_rng(527)
    weights = np.stack([rng.normal(size=(len(set(a)), 5))[np.unique(a, return_inverse=True)[1]] for a in amounts])
    actual = observed_assignment(weights, amounts)
    perms = np.array(list(permutations(range(5))))
    expected = []
    for a, w in zip(amounts, weights):
        all_weights = w[np.arange(5), perms].sum(1)
        equivalent = (a[perms] == a).all(1)
        expected.append(logsumexp(all_weights[equivalent]) - logsumexp(all_weights))
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    return float(np.max(np.abs(actual-expected)))


def analyse_cells(labels, codes, scores, rng):
    cells = list(pd.Series(np.arange(len(codes))).groupby(codes).indices.values())
    pos = np.array([labels[idx].sum() for idx in cells], int)
    neg = np.array([len(idx)-p for idx, p in zip(cells, pos)])
    wins = pos > neg
    pairs = int(pos @ neg)
    ranks = np.zeros((len(labels), len(scores)))
    mixed_positive = 0
    for idx, p, n in zip(cells, pos, neg):
        if p and n:
            mixed_positive += int(p)
        for j, values in enumerate(scores.values()):
            ranks[idx, j] = rankdata(values[idx]) - (p+1)/2
    obs = labels.astype(float) @ ranks / pairs
    null = []
    for _ in range(20):
        shuffled = np.tile(labels, (100, 1))
        for idx, p, n in zip(cells, pos, neg):
            if p and n:
                order = np.argsort(rng.random((len(shuffled), len(idx))), axis=1)
                shuffled[:, idx] = labels[idx[order]]
        null.extend(shuffled @ ranks / pairs)
    null = np.asarray(null)
    return dict(cells=len(cells), strict_majority_cells=int(wins.sum()),
                hindsight_fixed=int(pos[wins].sum()), hindsight_harmed=int(neg[wins].sum()),
                hindsight_net=int((pos[wins]-neg[wins]).sum()), positives_in_mixed_cells=mixed_positive,
                matched_pairs=pairs, conditional_auc={k:dict(observed=float(obs[j]),
                null95=np.quantile(null[:, j], [.025, .975]).tolist()) for j,k in enumerate(scores)})


def audit():
    row, y, frame, groups, _ = load()
    positive = frame[BILLS].gt(0).to_numpy()
    eligible = (frame.BookingSize.to_numpy() == 1) & positive.any(1)
    prob = {k: [] for k in ('reference', 'anonymous', 'assigned')}
    for s in map(str, SEEDS):
        fold, base, ref, families = family_scores(s, row, y, frame)
        alternatives = np.column_stack(list(families.values()))
        assert alternatives.shape[1] == 48
        eligible &= ~ref & (alternatives < .5).all(1)
        prob['reference'].append(base)
        with np.load(ART/f'nov23_fee_identity_assignment/{s}.npz') as receipt:
            for key, value in [('row', row), ('y', y), ('fold', fold), ('ref', ref), ('base', base)]:
                np.testing.assert_array_equal(receipt[key], value)
            for k in ('anonymous','assigned'):
                prob[k].append(receipt[k])
    labels = y[eligible]
    assert len(labels) == len(set(groups[eligible])) == 922 and labels.sum() == 95
    scores = {k: np.mean(v, axis=0)[eligible] for k, v in prob.items()}
    bits = 2**np.arange(len(BILLS))
    signature = positive[eligible] @ bits
    raw = frame.loc[eligible, BILLS]
    missing = raw.isna().to_numpy() @ bits
    total = raw.sum(axis=1).to_numpy()
    cuts = np.unique(np.quantile(total, [.2, .4, .6, .8]))
    scale = np.searchsorted(cuts, total, side='left')
    by_signature = {}
    for sign in np.unique(signature):
        selected = labels[signature == sign]
        names = [name for i, name in enumerate(BILLS) if sign & bits[i]]
        by_signature['+'.join(names)] = dict(transported=int(selected.sum()), not_transported=int((~selected).sum()))
    partitions = dict(positive_services=signature, services_and_scale=signature*5+scale,
                      services_scale_missing=(signature*5+scale)*32+missing)
    rng = np.random.default_rng(531)
    result = dict(passengers=len(labels), target=int(labels.sum()), total_fee_quintile_boundaries=cuts.tolist(),
                  by_positive_services=by_signature, assignment_maximum_log_error=verify_likelihood(),
                  draws=2000, seed=531, analyses={k:analyse_cells(labels, v, scores, rng) for k,v in partitions.items()})
    write(ROOT/'reports/nov23_unanimous_fee_signatures/signatures.json', result)
    print(result)


if __name__ == '__main__':
    audit()
