"""Test whether withholding reliable fields reveals outcomes missed by eight model votes."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write


def learn_switch(scores, alternatives, y, ref, groups, fold, use_alternatives):
    sign = 1-2*ref.astype(float)
    aligned = logit(scores.clip(.003, .997))*sign[:, None]
    alt = logit(alternatives.clip(.003, .997))*sign[:, None]
    # Opposite predictions are needed for this intervention, for both the control and the treatment.
    # Prefer discrete votes here so p=.5 has the same meaning as in the archived classifiers.
    offers = ((alternatives >= .5) != ref[:, None]).any(1)
    unanimous = ((scores >= .5) == ref[:, None]).all(1)
    inputs = np.c_[aligned[:, 0], aligned.mean(1), aligned.max(1), aligned.std(1), ref]
    if use_alternatives:
        inputs = np.c_[inputs, alt, alt.max(1), alt.mean(1)]
    vote = ref.copy()
    for f in np.unique(fold):
        ix = np.flatnonzero(fold == f)
        for a, b in GroupKFold(3, shuffle=True, random_state=925).split(ix, groups=groups[ix]):
            assert not np.intersect1d(groups[ix[a]], groups[ix[b]]).size
            for subset in [offers & unanimous, offers & ~unanimous]:
                t, q = ix[a][subset[ix[a]]], ix[b][subset[ix[b]]]
                if len(q) and np.unique(ref[t] != y[t]).size == 2:
                    model = LogisticRegression(C=.2, max_iter=500).fit(inputs[t], ref[t] != y[t])
                    vote[q] ^= model.predict(inputs[q])
    return vote


def run():
    row, y, _, groups, _ = load()
    histories, wrong, right, metrics = {}, [], [], {}
    for seed in map(str, SEEDS):
        fold, ref, opinions, _ = read_opinions(seed, row, y)
        with np.load(ART/f'nov23_broad_geographic_transfer/{seed}.npz') as z:
            for k, v in [('row', row), ('y', y), ('fold', fold)]:
                np.testing.assert_array_equal(z[k], v)
            views = np.c_[z['full'], z['no_bills'], z['no_cabin']]
        votes = {name: views[:, i] >= .5 for i, name in enumerate(['full', 'no_bills', 'no_cabin'])}
        for k, use in [('offers_only', False), ('view_scores', True)]:
            votes[k] = learn_switch(opinions, views, y, ref, groups, fold, use)
        votes['reference'] = ref
        h = {k: v == y for k, v in votes.items()}
        consensus = ((opinions >= .5) == ref[:, None]).all(1)
        wrong.append(consensus & (ref != y))
        right.append(consensus & (ref == y))
        metrics[seed] = {k: paired(v, ref == y) for k, v in h.items()}
        for k, v in h.items():
            histories.setdefault(k, []).append(v)
    h = {k: np.stack(v) for k, v in histories.items()}
    wrong, right = np.stack(wrong).all(0), np.stack(right).all(0)
    assert wrong.sum() == 332 and right.sum() == 4144
    masks = {'unanimous_wrong': wrong, 'unanimous_right': right}
    miss = (~h['reference']).sum(0)
    masks.update({f'reference_wrong_{i}': miss == i for i in range(4)})
    stats = {}
    for key, hit in h.items():
        stats[key] = dict(groups={k: dict(rows=int(m.sum()), stable_right=int(hit[:, m].all(0).sum()),
                                          stable_wrong=int((~hit[:, m]).all(0).sum())) for k, m in masks.items()},
                          gain95_pp=interval((hit.astype(float)-h['reference']).mean(0), groups))
    coverage = {}
    for k, mask in masks.items():
        control = h['full'][:, mask].all(0)
        novel = {v: h[v][:, mask].all(0) & ~control for v in ['no_bills', 'no_cabin']}
        coverage[k] = dict(full_right=int(control.sum()), new_no_bills=int(novel['no_bills'].sum()),
                           new_no_cabin=int(novel['no_cabin'].sum()),
                           new_either=int((novel['no_cabin'] | novel['no_bills']).sum()))
    write(ROOT/'reports/nov23_pairwise_adjudication/withheld.json',
          dict(plans=metrics, grouped=stats, extra_coverage=coverage))
    print(coverage['unanimous_wrong'], stats['view_scores'], flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
