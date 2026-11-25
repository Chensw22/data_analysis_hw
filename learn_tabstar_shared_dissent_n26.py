"""Calibrate the value of TabSTAR disagreement exclusively on other held-out bookings from the same base fold."""
import json

import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import champions, fit
from full_roster_booking_validation_n23 import data
from original_tabstar_probe_n26 import ART, SHA
from target_mix_expert_routing_n26 import ROOT, batches, ticker


def packet(x, ids, parent, weak, strong):
    direction = np.where(parent, -1., 1.)
    scores = np.column_stack([logit(p[ids].clip(.002, .998)) * direction for p in [weak, strong]])
    return np.column_stack([x, scores, scores[:, 0] * scores[:, 1], (scores > 0).all(1)])


def run():
    raw, y, group, folds = data()
    nine, known, _, hist = champions()
    weak = np.full(len(y), np.nan)
    strong = weak.copy()
    for k in np.unique(folds):
        for prefix, field, dest in [('fold', 'readout', weak), ('tuned', 'finetuned', strong)]:
            with np.load(ART / f'{prefix}{k}.npz') as z:
                q, t = z['q'], z['t']
                np.testing.assert_array_equal(q, np.flatnonzero(folds == k))
                np.testing.assert_array_equal(z['train_y'], y[t])
                np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[q])
                assert z['checkpoint'] == SHA and not np.isin(group[q], group[t]).any()
                dest[q] = z[field]
    assert np.isfinite(strong).all() and np.isfinite(weak).all()
    bags = []
    for b in batches(True):
        if b.tag != 'full':
            continue
        t, q = b.train_ids, b.ids
        assert known[q].all() and known[t].all() and len(np.unique(folds[np.r_[q, t]])) == 1
        assert not np.isin(group[t], group[q]).any()
        old, parent = b.train_cell % 2 != 0, b.query_cell % 2 != 0
        train_common, common = old == nine[t], parent == nine[q]
        train_dissent = train_common & ((strong[t] >= .5) != old)
        dissent = common & ((strong[q] >= .5) != parent)
        x = packet(b.train_fields, t, old, weak, strong)
        z = packet(b.query_fields, q, parent, weak, strong)
        dedicated = fit(x, b.train_error, z, train_dissent.astype(float))
        general = fit(x, b.train_error, z, train_common.astype(float))
        blind = fit(b.train_fields, b.train_error, b.query_fields, train_common.astype(float))
        actions = dict(dedicated=dissent & (dedicated > .5), pooled=dissent & (general > .5),
                       old_fields=dissent & (blind > .5), both=dissent & (dedicated > .5) & (general > .5),
                       new_information=dissent & (dedicated > .5) & (general > .5) & (blind <= .5))
        bags.append((q, b.query_error, common, actions))
    ids, wrong, common = [np.concatenate([item[j] for item in bags]) for j in range(3)]
    assert len(ids) == len(np.unique(ids)) == 8532 and (common & wrong).sum() == 1232
    report = {}
    for n in bags[0][-1]:
        decision = np.concatenate([item[-1][n] for item in bags])
        masks = dict(common=common, newly_encountered_booking=common & ~np.isin(group[ids], group[hist]))
        report[n] = {k: ticker(m, wrong, decision) for k, m in masks.items()}
        report[n]['fold_net'] = [int(np.sum(np.where(wrong, 1, -1) * decision * (folds[ids] == k)))
                                 for k in np.unique(folds)]
        print(n, report[n], flush=True)
    path = ROOT / 'reports/nov23_matched_learning_curve/learn_tabstar_shared_dissent.json'
    path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
