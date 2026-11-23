"""Diagnostic counterfactual: reveal validation peers' labels and withhold every focal target."""
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from ordered_booking_outcomes_n23 import (
    ART, ROOT, SEEDS, compact_features, contexts, directions, fit, forward, load, paired,
    read_opinions, write,
)


def conditional_mass(part, w, hold):
    index = part[0]
    result = np.full((len(hold), 2), np.nan)
    for j in range(index.shape[1]):
        present = index[:, j] >= 0
        focus = index[present, j]
        # Other parties are disjoint. Replacing every focal label at once
        # computes its two possible numerators independently in each party.
        forced = hold.copy()
        for label in (0, 1):
            forced[focus] = label
            result[focus, label] = forward(part, w, forced)[1][present]
    return result


def run():
    row, y, _, group, _ = load()
    raw = pd.read_csv(ROOT/'data/raw/train.csv', usecols=lambda c: c not in ['Name', 'Transported'])
    gid, pos = raw.PassengerId.str.split('_', expand=True).astype(int).to_numpy().T
    np.testing.assert_array_equal(group, gid[row])
    frame = compact_features(raw)
    frame['BookingSize'] = pd.Series(gid).map(pd.Series(gid).value_counts()).to_numpy()
    attrs = (frame.CryoSleep.to_numpy(), frame.Age.to_numpy(), raw.Cabin.fillna('').to_numpy())
    metrics, history, refs, shared = {}, {}, [], []
    no_observed_peer = pd.Series(group).map(pd.Series(group).value_counts()).to_numpy() == 1
    for seed in map(str, SEEDS):
        fold, ref, scores, _ = read_opinions(seed, row, y)
        refs.append(ref == y)
        shared.append(((scores >= .5) != y[:, None]).all(1))
        bank = {}
        for f in np.unique(fold):
            t, q = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            assert not np.intersect1d(group[t], group[q]).size
            observed, it, zt, iq, zq = contexts(frame, gid, row, y, t, q)
            assert np.all(observed[iq] == -1)
            hold = np.full(len(raw), -1, int)
            hold[row[q]] = y[q]
            training = directions(it, gid, pos, zt, *attrs, False)
            queries = [part for z in zq for part in directions(iq, gid, pos, z, *attrs, False)]
            for linked in [False, True]:
                name = 'ticket_chain' if linked else 'ticket_independent'
                w = fit(training, observed, linked)
                log_mass = np.stack([conditional_mass(part, w, hold)[row[q]] for part in queries])
                mix = logsumexp(log_mass, axis=0)
                bank.setdefault(name, np.full(len(y), np.nan))[q] = np.exp(mix[:, 1]-logsumexp(mix, axis=1))
        metrics[seed] = {}
        with np.load(ART/f'nov23_ordered_booking_outcomes/{seed}.npz') as z:
            for key, v in [('row', row), ('y', y), ('fold', fold), ('ref', ref)]:
                np.testing.assert_array_equal(z[key], v)
            for k, oracle in bank.items():
                before = z[k]
                assert np.isfinite(oracle).all()
                np.testing.assert_allclose(oracle[no_observed_peer], before[no_observed_peer], atol=1e-8)
                hit = (oracle >= .5) == y
                anchor = (before >= .5) == y
                direct = np.where(y, before, 1-before)
                revealed = np.where(y, oracle, 1-oracle)
                metrics[seed][k] = dict(vs_hidden=paired(hit, anchor), vs_reference=paired(hit, ref == y),
                                        mean_log_loss_hidden=float(-np.log(direct).mean()),
                                        mean_log_loss_revealed=float(-np.log(revealed).mean()),
                                        mean_prob_move_on_peer=float(np.abs(oracle-before)[~no_observed_peer].mean()))
                history.setdefault(k+'_revealed', []).append(hit)
                history.setdefault(k+'_hidden', []).append(anchor)
        print(seed, metrics[seed]['ticket_chain']['vs_hidden'], flush=True)
    allref = np.stack(refs)
    groups = {'no_peer_to_reveal': no_observed_peer, 'at_least_one_peer': ~no_observed_peer,
              'always_wrong': ~allref.any(0), 'sometimes_wrong': allref.any(0) & ~allref.all(0),
              'shared_by_eight': np.stack(shared).all(0), 'always_right': allref.all(0)}
    summary = {}
    for k, sequence in history.items():
        hit = np.stack(sequence)
        summary[k] = {s: dict(n=int(m.sum()), stable_hit=int(hit[:,m].all(0).sum())) for s,m in groups.items()}
    write(ROOT/'reports/nov23_ordered_booking_outcomes/peer_observation.json',
          dict(uses_query_peer_outcomes=True, deployable=False, plans=metrics, groups=summary))


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
