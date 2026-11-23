"""Validate two independently trained budget experts against a larger-training reference."""
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_pretrained_spending_views_n23 import frames, predict
from matched_booking_learning_curve_n23 import portions


def evaluate(y, base, ref, cat, tab, fold, groups):
    result, saved = {}, {}
    for k, p in [('cat', cat), ('tab', tab), ('both', (cat+tab)/2)]:
        prob = .75*base+.25*p
        vote = np.where(ref==(base>=.5), prob>=.5, ref)
        gain = (vote==y).astype(float)-(ref==y)
        result[k] = dict(**paired(vote==y, ref==y), fold_net=[int(gain[fold==f].sum()) for f in np.unique(fold)],
                         gain_pp=float(100*gain.mean()), interval95_pp=interval(gain, groups))
        saved[k] = vote
    return result, saved


def run():
    row, y, _, groups, _ = load()
    raw = pd.read_csv(ROOT/'data/raw/train.csv', usecols=lambda c:c not in ('Name','Transported')).iloc[row].reset_index(drop=True)
    frame = frames(raw)[0]
    plans, history, base_hits = {}, {}, []
    for seed in [*map(str, SEEDS), 'v131']:
        fold, ref, opinions, _ = read_opinions(seed, row, y)
        with np.load(ART/f'nov23_pretrained_spending_views/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'], row)
            np.testing.assert_array_equal(z['fold'], fold)
            tab = z['budget']
        path = ART/('nov23_budget_quarter_replay/original.npz' if seed=='v131' else f'nov23_simplex_likelihood/{seed}.npz')
        with np.load(path) as z:
            np.testing.assert_array_equal(z['row'], row)
            np.testing.assert_array_equal(z['fold'], fold)
            cat = z['alternative' if seed=='v131' else 'budget_prior']
        plans[seed], saved = evaluate(y, opinions[:, 0], ref, cat, tab, fold, groups)
        if seed!='v131':
            base_hits.append(ref==y)
            for k, v in saved.items():
                history.setdefault(k, []).append(v==y)
    with np.load(ART/'nov23_budget_quarter_replay/matched.npz') as z:
        np.testing.assert_array_equal(row, z['row'])
        np.testing.assert_array_equal(y, z['y'])
        fold, base, cat = z['fold'], z['base'][1], z['alt'][1]
    out = ART/'nov23_budget_foundation_nearfull'
    out.mkdir(exist_ok=True)
    tscore = np.full(len(y), np.nan)
    for f, _, t, q in portions(groups):
        np.testing.assert_array_equal(q, np.flatnonzero(fold==f))
        assert not np.intersect1d(groups[t], groups[q]).size
        p = predict(frame, y, t, q, f)
        tscore[q] = p
        np.savez_compressed(out/f'{f}.npz', row=row[q], train=row[t], y=y[q], budget=p)
        print('near-full', f, flush=True)
    assert np.isfinite(tscore).all()
    plans['nearfull'], _ = evaluate(y, base, base>=.5, cat, tscore, fold, groups)
    base_hits = np.stack(base_hits)
    paired_ci = {}
    for k, v in history.items():
        gain = (np.stack(v).astype(float)-base_hits).mean(0)
        paired_ci[k] = dict(mean_gain_pp=float(100*gain.mean()), interval95_pp=interval(gain, groups))
    write(ROOT/'reports/nov23_bill_value/foundation_transfer.json', dict(plans=plans, grouped=paired_ci))
    print(plans['nearfull'], paired_ci, flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
