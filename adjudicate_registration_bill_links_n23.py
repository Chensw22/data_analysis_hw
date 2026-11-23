"""Learn when bill/registration alternatives win, exclusively on other held-out bookings."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from audit_stable_rescue_placebo_n23 import panel
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write


def adjudicate(y, booking, fold, base, ref, independent, linked, direct):
    direction = 2*ref.astype(float)-1
    scores = logit(np.column_stack([base, independent, linked, direct]).clip(.001, .999))
    relative = scores * direction[:, None]
    features = np.column_stack([relative[:, [0, 1, 3]], relative[:, 2]-relative[:, 1], direction])
    dispute = ((linked >= .5) != ref) | ((independent >= .5) != ref)
    eligible = dispute & (ref == (base >= .5))
    flip = np.zeros(len(y), bool)
    visited = np.zeros(len(y), bool)
    for outer in np.unique(fold):
        ids = np.flatnonzero((fold == outer) & eligible)
        for train, query in GroupKFold(3, shuffle=True, random_state=270).split(ids, groups=booking[ids]):
            t, q = ids[train], ids[query]
            assert not np.intersect1d(booking[t], booking[q]).size
            learner = LogisticRegression(C=.25, max_iter=300).fit(features[t], y[t] != ref[t])
            flip[q] = learner.predict(features[q])
            visited[q] = True
    np.testing.assert_array_equal(visited, eligible)
    return ref ^ flip


def run():
    row, y, frame, groups, _ = load()
    eligible, direction, _, _, old, _, _ = panel(row, y, frame)
    hard = eligible & (direction != y)
    right = eligible & (direction == y)
    prior_dissent = np.any(list(old.values()), axis=0)
    assert (hard.sum(), (hard & prior_dissent).sum(), (right & prior_dissent).sum()) == (372, 87, 604)
    plans, history = {}, {}
    for seed in [*map(str, SEEDS), 'v131']:
        with np.load(ART/f'nov23_registration_bill_links/{seed}.npz') as cache:
            np.testing.assert_array_equal(cache['row'], row)
            np.testing.assert_array_equal(cache['y'], y)
            base, ref = cache['base'], cache['ref']
            alternatives = {k: cache[k] for k in ['independent', 'linked', 'direct']}
            routed = adjudicate(y, groups, cache['fold'], base, ref, **alternatives)
        predictions = {k: p >= .5 for k, p in alternatives.items()}
        predictions['routed'] = routed
        plans[seed] = {}
        for key, vote in predictions.items():
            hit = y == vote
            plans[seed][key] = paired(hit, y == ref)
            if seed != 'v131':
                history.setdefault(key, []).append(hit)
    summary = {}
    for key, outcomes in history.items():
        hits = np.stack(outcomes)
        stable_fix = hard & hits.all(0)
        stable_harm = right & ~hits.any(0)
        summary[key] = dict(hard_ever=int((hard & hits.any(0)).sum()),
                            hard_stable=int(stable_fix.sum()), new_hard=int((stable_fix & ~prior_dissent).sum()),
                            stable_harms=int(stable_harm.sum()), new_harms=int((stable_harm & ~prior_dissent).sum()))
    write(ROOT/'reports/nov23_registration_bill_links/adjudication.json', dict(plans=plans, stable=summary))
    print(summary)
    print({s: p['routed'] for s, p in plans.items()})


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
