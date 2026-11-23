"""Learn registration contrasts between opposite-label passengers with matched fee signatures."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS
from broad_opposing_passengers_n23 import evaluate
from exaone_validation_data import CATEGORICAL
from fee_identity_assignment_n23 import REPORT


def context(frame):
    cats = frame[CATEGORICAL].copy()
    cats['deck_side'] = frame.Deck + '/' + frame.Side
    cats['origin_deck'] = frame.HomePlanet + '/' + frame.Deck
    cats['origin_sleep'] = frame.HomePlanet + '/' + frame.CryoSleep
    age = frame.Age.fillna(28).to_numpy()
    cabin = frame.CabinNumber.fillna(600).to_numpy()/1000.
    return np.column_stack([pd.get_dummies(cats, dtype=float), frame.Age.isna(), frame.CabinNumber.isna(),
                            age/40, np.maximum(14-age, 0)/10, np.maximum(age-40, 0)/20,
                            cabin, np.maximum(cabin-.6, 0), np.log(frame.BookingSize)])


def codes(frame, train):
    bills = frame[BILLS]
    signature = bills.gt(0).to_numpy() @ (1 << np.arange(len(BILLS)))
    total = bills.sum(axis=1).to_numpy()
    paid = total[train][total[train] > 0]
    band = np.searchsorted(np.quantile(paid, [.2, .4, .6, .8]), total)
    return signature, signature*5 + band


def fit_rank(x, cell, y, booking, train):
    # Draw up to 512 pairs per cell, exclude within-booking pairs and weight
    # by inverse sampling probability divided by cell size.
    rng = np.random.default_rng(785)
    differences, weights = [], []
    for key in np.unique(cell[train]):
        members = train[cell[train] == key]
        pos, neg = members[y[members]], members[~y[members]]
        if not len(pos) or not len(neg):
            continue
        pair_count = len(pos)*len(neg)
        sampled = rng.choice(pair_count, size=min(512, pair_count), replace=False)
        a, b = pos[sampled//len(neg)], neg[sampled%len(neg)]
        valid = booking[a] != booking[b]
        if np.any(valid):
            differences.append(x[a[valid]]-x[b[valid]])
            weights.extend([pair_count/(len(sampled)*len(members))]*int(valid.sum()))
    diff = np.vstack(differences)
    ds = np.vstack([diff, -diff])
    target = np.repeat([True, False], len(diff))
    model = LogisticRegression(C=.5, fit_intercept=False, max_iter=400)
    model.fit(ds, target, sample_weight=.5*np.tile(weights, 2))
    return model.decision_function(x), len(diff)


def calibrate(score, code, prior, y, train):
    n = int(code.max())+1
    intercept = prior.copy()
    observed = np.bincount(code[train], weights=y[train], minlength=n)
    for _ in range(25):
        p = expit(score[train]+intercept[code[train]])
        residual = observed-np.bincount(code[train], weights=p, minlength=n)-2*(intercept-prior)
        precision = 2+np.bincount(code[train], weights=p*(1-p), minlength=n)
        step = (residual/precision).clip(-1, 1)
        intercept += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return intercept


def fit_fold(frame, groups, y, train, query):
    signature, cell = codes(frame, train)
    x = context(frame)
    ranked, n_pairs = fit_rank(x, cell, y, groups, train)
    result = {}
    for key, score in [('fees_only', np.zeros(len(y))), ('context_rank', ranked)]:
        prior = np.full(int(signature.max())+1, logit(y[train].mean())-score[train].mean())
        parent = calibrate(score, signature, prior, y, train)
        local = calibrate(score, cell, parent[np.arange(int(cell.max())+1)//5], y, train)
        result[key] = expit(score[query]+local[cell[query]])
    return result, dict(training_pairs=n_pairs)


if __name__ == '__main__':
    with threadpool_limits(1):
        evaluate('within_fee_context_rank', fit_fold, REPORT)
