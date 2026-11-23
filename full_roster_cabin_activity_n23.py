"""Use neighboring booked cabins' unlabeled activity as context, including solo travelers."""
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.preprocessing import StandardScaler

from broad_booking_policy_replay_n23 import interval
from full_roster_adaptive_bill_prior_n23 import EVIDENCE, proposal_design
from full_roster_conditional_pool_n23 import BILLS, ROOT, banks, data, latest_reference
from full_roster_direct_reference_risk_n23 import choose_reference
from full_roster_unexpected_peers_n23 import correction, reference_score


def local_activity(all_raw, n):
    deck, number, side = all_raw.Cabin.str.split('/', expand=True).to_numpy().T
    number = pd.to_numeric(number, errors='coerce').astype(float)
    booking = all_raw.PassengerId.str.split('_').str[0].to_numpy()
    fees = all_raw[BILLS].fillna(0).to_numpy()
    activity = np.column_stack([np.log1p(fees), fees > 0, all_raw.CryoSleep.eq(True),
                                all_raw.Age.lt(13), all_raw.Age.fillna(28) / 30,
                                all_raw.HomePlanet.eq('Earth'), all_raw.HomePlanet.eq('Europa')])
    known = np.isfinite(number)
    slots = pd.DataFrame({'deck': deck[known], 'side': side[known], 'number': number[known],
                          'booking': booking[known], 'row': np.flatnonzero(known)})
    lanes = {}
    for lane, items in slots.groupby(['deck', 'side']):
        positions = []
        for (pos, group), room in items.groupby(['number', 'booking'], sort=True):
            ids = room.row.to_numpy()
            positions.append((pos, group, activity[ids].mean(axis=0)))
        lanes[lane] = (np.array([v[0] for v in positions]), np.array([v[1] for v in positions]),
                       np.stack([v[2] for v in positions]))
    width = activity.shape[1] + 2
    result = np.zeros((n, 4, width))
    available = np.zeros((n, 4), bool)
    for i in np.flatnonzero(known[:n]):
        for s, which in enumerate((side[i], 'S' if side[i] == 'P' else 'P')):
            if (deck[i], which) not in lanes:
                continue
            positions, groups, attrs = lanes[deck[i], which]
            pivot = positions.searchsorted(number[i])
            for j, step in enumerate((-1, 1)):
                start = pivot - 1 if step < 0 else pivot
                chosen = []
                while 0 <= start < len(positions) and len(chosen) < 4:
                    if groups[start] != booking[i]:
                        chosen.append(start)
                    start += step
                if chosen:
                    distances = np.abs(positions[chosen] - number[i])
                    result[i, 2*s+j] = np.r_[np.average(attrs[chosen], axis=0, weights=1/(1+distances)),
                                              np.log1p(distances.mean()), len(chosen)/4]
                    available[i, 2*s+j] = True
    return result, available


def run():
    raw, y, group, fold = data()
    test = pd.read_csv(ROOT / 'data/raw/test.csv', usecols=lambda k: k != 'Name')
    geo, avail = local_activity(pd.concat([raw, test], ignore_index=True), len(y))
    names, half, full = banks(raw, y, fold)
    ref = latest_reference(y, group, fold, names, half, full)
    alt = names.index('balanced_full')
    full_p = reference_score(full, ref, alt)
    proposal, design = proposal_design(half[:, :5], half[:, [names.index(k) for k in EVIDENCE]])
    legal = avail.any(1)
    earth = raw.HomePlanet.eq('Earth').to_numpy() & raw.CryoSleep.eq(True).to_numpy()
    _, ids, sizes = np.unique(group, return_inverse=True, return_counts=True)
    solo = sizes[ids] == 1
    deck = raw.Cabin.str.split('/').str[0].fillna('?').to_numpy()
    aggregate = geo.sum(1) / np.maximum(1, avail.sum(1))[:, None]
    shapes = {'pooled': aggregate, 'directional': geo.reshape(len(raw), -1),
              'sleep_interaction': np.column_stack([aggregate, aggregate * earth[:,None]])}
    shapes['exchanged'] = shapes['pooled']
    pred = {k: ref.copy() for k in shapes}
    for f in np.unique(fold):
        t = (fold != f) & (fold % 2 == f % 2)
        q = (fold == f) & legal
        tr_ref = np.zeros(len(y), bool)
        for h in np.unique(fold[t]):
            use = t & (fold != h)
            held = t & (fold == h) & legal
            assert not np.isin(group[use], group[held | q]).any()
            tr_ref[held] = choose_reference(half, design, proposal, y, use, held)
        t &= legal
        offset = reference_score(half[t], tr_ref[t], alt)
        for k, x in shapes.items():
            s = StandardScaler().fit(x[t])
            train = s.transform(x[t])
            test = s.transform(x[q])
            if k == 'exchanged':
                rng = np.random.default_rng(920 + int(f))
                idx = np.flatnonzero(q)
                z = np.zeros_like(test)
                for i, row in enumerate(idx):
                    pool = (group[idx] != group[row]) & (deck[idx] == deck[row]) & (earth[idx] == earth[row])
                    if pool.any():
                        z[i] = test[rng.choice(np.flatnonzero(pool))]
                test = z
            w = correction(train, y[t], offset)
            pred[k][q] = logit(full_p[q].clip(.001, .999)) + test @ w >= 0
    shared = ((full >= .5) != y[:, None]).all(1)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    report = dict(covered=int(legal.sum()), solo_covered=int((solo & legal).sum()), results={})
    for k, prediction in pred.items():
        delta = (prediction == y).astype(int) - (ref == y)
        report['results'][k] = dict(accuracy=float((prediction == y).mean()),
            fixed=int((delta == 1).sum()), harmed=int((delta == -1).sum()),
            solo_fixed=int((delta[solo] == 1).sum()), solo_harmed=int((delta[solo] == -1).sum()),
            earth_fixed=int((delta[earth] == 1).sum()), earth_harmed=int((delta[earth] == -1).sum()),
            common_fixed=int((delta[shared] == 1).sum()), additional_net=int(delta.sum()-delta[dev].sum()),
            fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)], ci_pp=interval(delta,group))
    (ROOT/'reports/nov23_matched_learning_curve/cabin_activity.json').write_text(json.dumps(report, indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    run()
