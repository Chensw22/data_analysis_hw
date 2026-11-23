"""Learn when a dissenting small-data model is useful on other held-out bookings."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_support_trajectories_n23 import LEVEL, SUBSETS
from broad_tree_field_dividends_n23 import hard_cohorts
from support_doubling_consistency_n23 import REPORT, get_bank


def features(bank, base, ref):
    # Positive means support for overturning the reference.
    aligned = logit(bank.clip(.001, .999)) * np.where(ref, -1., 1.)
    quarters = aligned[LEVEL == 1]
    # Select the strongest challenger without looking at the target. Follow that
    # specific quarter into every larger fit that still includes its training data.
    strongest = quarters.argmax(0)
    path = []
    for size in (2, 3):
        continued = np.stack([aligned[[i for i,s in enumerate(SUBSETS) if len(s)==size and j in s]].mean(0)
                              for j in range(4)])
        path.append(continued[strongest, np.arange(len(base))])
    risk = np.where(ref, 1-base, base)
    anchor = np.c_[ref, logit(risk.clip(.001, .999))]
    signals = np.column_stack([anchor, np.sort(quarters, axis=0).T, *path, aligned[-1]])
    dissent = ((bank[LEVEL == 1] >= .5) != ref[None, :]).any(0)
    return {'risk':anchor, 'challenger_path':signals}, dissent & (ref==(base>=.5))


def run():
    row, y, _, groups, _ = load()
    _, _, hard, right = hard_cohorts(row, y)
    plans, history = {}, {}
    for seed in [*map(str, SEEDS), 'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            for k,v in [('row', row), ('y', y)]:
                np.testing.assert_array_equal(z[k], v)
            base, ref, fold = z['base'], z['ref'], z['fold']
        views, eligible = features(get_bank(seed, row, y, groups, fold), base, ref)
        plans[seed] = {}
        for name, x in views.items():
            flip = np.zeros(len(y), bool)
            seen = np.zeros(len(y), bool)
            for f in np.unique(fold):
                ids = np.flatnonzero((fold==f) & eligible)
                for a,b in GroupKFold(3, shuffle=True, random_state=962).split(ids, groups=groups[ids]):
                    t,q = ids[a], ids[b]
                    assert not np.intersect1d(groups[t], groups[q]).size
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=400))
                    model.fit(x[t], (ref != y)[t])
                    flip[q] = model.predict(x[q])
                    seen[q] = True
            np.testing.assert_array_equal(seen, eligible)
            hit = (ref ^ flip) == y
            plans[seed][name] = dict(**paired(hit, ref == y), hard_fixed=int((hard & hit).sum()))
            if seed != 'v131':
                history.setdefault(name, []).append(hit)
    stable = {k: dict(hard_ever=int((hard & np.any(h, axis=0)).sum()), hard_stable=int((hard & np.all(h, axis=0)).sum()),
                      stable_harms=int((right & ~np.any(h, axis=0)).sum())) for k,h in history.items()}
    write(REPORT/'gate.json', dict(plans=plans, stable=stable))
    print(plans, stable)


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
