"""Compare direct transport and error targets using all passenger orders as calibration examples."""
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from scipy.special import logit

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS
from broad_tree_field_dividends_n23 import hard_cohorts
from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, SEED, data
from full_roster_consensus_ranking_n23 import half_bank
from full_roster_metric_adjudication_n23 import load_scores
from tabicl_routed_context_n23 import DIGEST, WEIGHT, infer


def context(raw, groups):
    own = compact_features(raw)
    bills = raw[BILLS]
    counts = pd.Series(groups).map(pd.Series(groups).value_counts())
    own['BookingSize'] = counts
    # Other passengers in the query order contribute only their observed spending.
    values = np.log1p(bills.fillna(0))
    peer = (values.groupby(groups).transform('sum')-values).div((counts-1).clip(lower=1), axis=0)
    return own.join(peer.add_prefix('PeerLog'))


def design(scores, fields, target):
    anchor = scores[:, -1] >= .5
    p = np.where(anchor[:, None], 1-scores, scores) if target == 'error' else scores
    x = pd.DataFrame(logit(p.clip(.001, .999)), columns=[*FAMILIES, 'mitra96'])
    x['MitraDecision'] = anchor.astype(int)
    return x if target == 'scores_only' else x.join(fields)


def run():
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest() == DIGEST
    raw, y, groups, fold = data()
    half = half_bank(raw, y, fold)
    full = np.column_stack([load_scores(raw, y, fold, OUT / f'{k}.npz', ['p'])[:, 0] for k in (*FAMILIES, 'mitra96')])
    fields = context(raw, groups)
    base = full[:, -1] >= .5
    agree = ((full >= .5) == base[:, None]).all(1)
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    hard = np.zeros(len(y), bool)
    hard[dev] = hard_cohorts(dev, y[dev])[2]
    result, probabilities = {}, {}
    for target in ['scores_only', 'transport', 'error']:
        a, b = design(half, fields, target), design(full, fields, target)
        p = np.empty(len(y))
        for f in np.unique(fold):
            t = np.flatnonzero((fold % 2 == f % 2) & (fold != f))
            q = np.flatnonzero(fold == f)
            assert not np.isin(groups[t], groups[q]).any()
            x = pd.concat([a.iloc[t], b.iloc[q]], ignore_index=True)
            label = y[t] != (half[t, -1] >= .5) if target == 'error' else y[t]
            pred = infer(x, label, np.arange(len(t)), np.arange(len(t), len(x)), SEED+int(f))
            p[q] = np.where(base[q], 1-pred, pred) if target == 'error' else pred
        assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
        probabilities[target] = p
        result[target] = {}
        for name, pred in [('direct', p >= .5), ('with_mitra', (p+full[:, -1]) >= 1)]:
            fixed, harmed = (pred == y) & (base != y), (pred != y) & (base == y)
            delta = fixed.astype(int)-harmed
            result[target][name] = dict(accuracy=float((pred == y).mean()),
                fixed=int(fixed.sum()), harmed=int(harmed.sum()), unanimous_fixed=int((fixed & agree).sum()),
                unanimous_harmed=int((harmed & agree).sum()), historic_fixed=int((fixed & hard).sum()),
                historic_harmed=int((harmed & hard).sum()), additional_net=int(delta.sum()-delta[dev].sum()),
                fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)], ci_pp=interval(delta, groups))
        print(target, result[target], flush=True)
    cache = ROOT / 'artifacts/nov23_half_roster_disagreement/transformer_oof.npz'
    np.savez_compressed(cache, passenger=raw.PassengerId.to_numpy(str), fold=fold, y=y, **probabilities)
    report = ROOT / 'reports/nov23_matched_learning_curve/transformer_adjudication.json'
    report.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    torch.set_num_threads(2)
    run()
