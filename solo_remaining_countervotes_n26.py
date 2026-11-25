"""Test a held-order selector on solo passengers missed by both new repair methods."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from distinct_challenger_error_reach_n26 import population
from full_roster_booking_validation_n23 import ROOT
from tabnet_attention_transfer_n26 import RESULT, collect
from target_mix_expert_routing_n26 import ticker


def learned_scores(counts, flags, reference, labels, t, q):
    fractions = np.column_stack(list(counts.values()))
    x = np.column_stack([reference, fractions, flags, fractions * reference[:, None]])
    linear = LogisticRegression(C=.5, max_iter=400).fit(x[t], labels[t]).predict_proba(x[q])[:, 1]
    # Back off an exact vote pattern to the observed error rate for this decision direction.
    code = flags.astype(int) @ (2 ** np.arange(flags.shape[1]))
    state = 2 * code + reference
    global_mean = (labels[t].sum() + 1) / (len(t) + 2)
    count = np.bincount(reference[t].astype(int), minlength=2)
    successes = np.bincount(reference[t].astype(int), weights=labels[t], minlength=2)
    base_rate = (successes + 8 * global_mean) / (count + 8)
    nstates = 2 ** (1 + flags.shape[1])
    support = np.bincount(state[t], minlength=nstates)
    wins = np.bincount(state[t], weights=labels[t], minlength=nstates)
    lookup = (wins[state[q]] + 8 * base_rate[reference[q].astype(int)]) / (support[state[q]] + 8)
    return linear, lookup


def context():
    raw, y, g, fold, parent, same, reach, blocks, flags, wrong, solo, _ = population()
    with np.load(ROOT / 'artifacts/nov26_replicated_solo_training_errors/scores.npz') as z:
        np.testing.assert_array_equal(z['passenger'], raw.PassengerId)
        np.testing.assert_array_equal(z['fold'], fold)
        stable = ((z['real'] >= .5) != parent[:, None]).all(1)
    with np.load(ROOT / 'artifacts/nov26_nested_environment_residual/scores.npz') as z:
        np.testing.assert_array_equal(z['ids'], raw.PassengerId)
        np.testing.assert_array_equal(z['fold'], fold)
        region = (z['departure_maximin'].mean(0) >= .5) != parent
    runs = collect()[0]
    masks = {}
    for tag, items in runs.items():
        ids, common, now = (np.concatenate([b[j] for b in items]) for j in range(3))
        use = common & ~np.concatenate([b[3]['actual_and_scores'] for b in items])
        scope = np.zeros(len(y), bool)
        scope[ids] = use & (parent[ids] == now)
        masks[tag] = scope & same & solo & ~stable & ~region & (reach > 0)
    return raw, y, g, fold, parent, blocks, flags, wrong, masks


def run():
    _, y, g, fold, parent, blocks, flags, wrong, masks = context()
    counts = {n: v.mean(axis=1) for n, v in blocks.items()}
    eligible = masks['full']
    assert (eligible & wrong).sum() == 524
    linear, lookup = np.full(len(y), np.nan), np.full(len(y), np.nan)
    top = np.zeros(len(y), bool)
    # Selected countervotes and their individual model predictions all exclude the focal booking.
    # Within the primary population, split each base validation fold once more before fitting.
    for k in range(8):
        rows = np.flatnonzero(eligible & (fold == k))
        for a,b in KFold(4,shuffle=True,random_state=9275+k).split(rows):
            t,q = rows[a],rows[b]
            assert not np.isin(g[t], g[q]).any()
            linear[q], lookup[q] = learned_scores(counts, flags, parent, wrong, t, q)
            top[q[np.argsort(-linear[q],kind='stable')[:int(np.ceil(len(q)*.1))]]] = True
    assert np.array_equal(np.isfinite(linear), eligible)
    report = {}
    for tag, scope in masks.items():
        mask = scope & eligible
        report[tag] = {'covered_rows': int(mask.sum()), 'errors': int((mask & wrong).sum())}
        actions = {name: b.any(1) for name,b in blocks.items()}
        actions.update(linear=linear>.5, pattern=lookup>.5, both=(linear>.5)&(lookup>.5),
                       linear_top_decile=top)
        for name, action in actions.items():
            result = ticker(mask, wrong, action)
            delta = mask * action * np.where(wrong,1,-1)
            result['net_by_fold'] = [int(delta[fold == k].sum()) for k in range(8)]
            report[tag][name] = result
        print(tag, report[tag]['covered_rows'], {n:(report[tag][n]['fixed'],report[tag][n]['harmed'])
              for n in ['linear','pattern','both','linear_top_decile']},flush=True)
    report['scope'] = 'All learning uses other solo bookings within the same primary base validation fold. Alternate panels only report intersecting rows with matching reference decisions and reuse the same primary predictions.'
    (RESULT.parent / 'solo_remaining_countervotes.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
