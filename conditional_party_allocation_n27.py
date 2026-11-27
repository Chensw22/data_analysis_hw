"""Learn which companions were transported conditional on a training booking's exact total."""
import json
from collections import defaultdict
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT
from manifest_activity_transitions_n27 import design
from repeated_archive_misses_n26 import ART, cohort, count
from unanimous_error_reach_n27 import inputs


@lru_cache(maxsize=None)
def states(n):
    return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float)


def conditional_weights(x, y, bookings):
    strata = defaultdict(list)
    for rows in bookings:
        positive = int(y[rows].sum())
        if 0 < positive < len(rows):
            strata[len(rows), positive].append(rows)
    strata = {k: np.stack(v) for k, v in strata.items()}
    # All possible assignments with the correct count compete within their own order.
    def objective(w):
        residuals = np.zeros(len(x))
        score = x @ w
        loss = 1.5 * (w @ w)
        for (size, positive), rows in strata.items():
            candidates = states(size)
            candidates = candidates[candidates.sum(axis=1) == positive]
            logits = score[rows] @ candidates.T
            loss += (logsumexp(logits, axis=1) - (score[rows] * y[rows]).sum(axis=1)).sum()
            residuals[rows] = softmax(logits, axis=1) @ candidates - y[rows]
        return loss, x.T @ residuals + 3*w
    fitted = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-11})
    if not fitted.success:
        raise RuntimeError(fitted.message)
    return fitted.x, sum(len(v) for v in strata.values())


def transfer(prior, evidence, bookings):
    result = prior.copy()
    for rows in bookings:
        size = len(rows)
        if size == 1:
            continue
        options = states(size)
        counts = options.sum(axis=1)
        p = prior[rows]
        incumbent_joint = np.exp(options @ np.log(p) + (1-options) @ np.log1p(-p))
        mass = np.bincount(counts.astype(int), weights=incumbent_joint)
        allocation = options @ evidence[rows]
        updated = np.zeros(size)
        for k, weight in enumerate(mass):
            use = counts == k
            updated += weight * (softmax(allocation[use]) @ options[use])
        np.testing.assert_allclose(updated.sum(), p.sum(), atol=1e-8)
        result[rows] = updated
    return result


def passenger_matrix(frame):
    reg, act = design(frame)
    planet = frame.HomePlanet.fillna('?')
    return np.column_stack([reg, act, *[act * planet.eq(p).to_numpy()[:,None] for p in ['Earth','Europa']],
                            frame.PassengerId.str.split('_').str[1].astype(int).to_numpy()/8])


def run():
    raw, frame, y, g, fold, first, changed, _, opinions, _ = inputs()
    n = len(y)
    parent = first ^ changed
    _, _, known, fresh = cohort(raw, y, fold)
    common = known & ((opinions[:n] >= 0.5) == parent[:n, None]).all(axis=1)
    groups = frame.PassengerId.str.split('_').str[0]
    orders = list(frame.groupby(groups, sort=False).indices.values())
    multi = groups.duplicated(keep=False).to_numpy()
    scopes = dict(all=known, multi=known & multi[:n], fresh=known & fresh, shared=common,
                  shared_multi=common & multi[:n])
    wrong = parent[:n] != y
    references = np.median(opinions[:, 1:4], axis=1)
    prior_logit = np.where(parent, 1., -1.) * np.maximum(1e-6, abs(logit(references.clip(.002,.998))))
    prior = expit(prior_logit)
    np.testing.assert_array_equal(prior >= .5, parent)
    x = passenger_matrix(frame)
    seq = np.zeros(n, int)
    for k, ids in enumerate(np.array_split(np.unique(g), 8)):
        seq[np.isin(g, ids)] = k
    complete = [rows for rows in orders if rows[0] < n and known[rows].all()]
    def bounds(rows):
        hits = misses = paired = total = shared_possible = 0
        for ix in rows:
            fp = parent[ix] & ~y[ix]
            fn = ~parent[ix] & y[ix]
            paired += 2 * min(fp.sum(), fn.sum())
            shared_possible += min((fp & common[ix]).sum(), fn.sum()) + min((fn & common[ix]).sum(), fp.sum())
            if fp.any() or fn.any():
                hits += (fp | fn).sum()
                misses += 1
            total += len(ix)
        return dict(passengers=int(total), errors=int(hits), orders_with_errors=int(misses),
                    max_net_correct_preserving_hard_counts=int(paired),
                    max_shared_correct_without_harm=int(shared_possible))
    report = dict(complete_known_multi=bounds([rows for rows in complete if len(rows)>1]))
    saved = {}
    for scheme, f in [('random', fold), ('ordered', seq)]:
        scores = np.zeros((len(frame), 2))
        mix_counts = []
        for k in range(9):
            train = np.flatnonzero(f != k)
            query = np.flatnonzero(f == k) if k < 8 else np.arange(n,len(frame))
            assert not np.isin(groups.iloc[train], groups.iloc[query]).any()
            train_set = np.zeros(len(frame), bool)
            train_set[train] = True
            bank = [r for r in orders if train_set[r[0]]]
            assert all(train_set[r].all() for r in bank)
            population = LogisticRegression(C=1/3, fit_intercept=False, max_iter=500, tol=1e-7)
            population.fit(x[train], y[train])
            conditional, mixed = conditional_weights(x, y, bank)
            scores[query,0] = population.decision_function(x[query])
            scores[query,1] = x[query] @ conditional
            mix_counts.append(mixed)
        same_input = transfer(prior, scores[:,0], orders)
        conditioned = transfer(prior, scores[:,1], orders)
        # Only use the learned change of allocation odds compared to the same representation.
        increment = transfer(prior, prior_logit + scores[:,1] - scores[:,0], orders)
        actions = {key:(p >= .5) != parent for key,p in
                   [('ordinary',same_input), ('conditioned',conditioned), ('increment',increment)]}
        actions['supported_increment'] = actions['increment'] & actions['conditioned']
        saved[scheme+'_scores'] = scores
        result = {}
        for key, flag in actions.items():
            assert not flag[~multi].any()
            delta = flag[:n] * np.where(wrong, 1, -1)
            saved[scheme+'_'+key] = flag
            result[key] = dict(scopes={k:count(flag[:n], v, wrong) for k,v in scopes.items()},
                               fold_net=[int(delta[known & (f==j)].sum()) for j in range(8)],
                               test=int(flag[n:].sum()))
        report[scheme] = dict(mixed_train_orders=mix_counts, rules=result)
    np.testing.assert_allclose(saved['random_scores'][n:], saved['ordered_scores'][n:])
    report['repeat'] = {key:{k:count(saved['random_'+key][:n] & saved['ordered_'+key][:n], v, wrong)
                              for k,v in scopes.items()} for key in actions}
    out=ART/'nov27_conditional_party_allocation'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'aligned.npz', passenger=frame.PassengerId.to_numpy(str), y=y, fold=fold, **saved)
    (ROOT/'reports/nov23_matched_learning_curve/conditional_party_allocation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['complete_known_multi'],report['random']['rules']['supported_increment'],report['ordered']['rules']['supported_increment'],report['repeat'],flush=True)


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
