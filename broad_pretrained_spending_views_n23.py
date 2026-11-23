"""Test whether a pretrained tabular model benefits from the spending abstraction that transferred online."""
import hashlib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from exaone_validation_data import compact_features
from spend_simplex_likelihood_n23 import BILLS, design
from tabicl_routed_context_n23 import DIGEST, WEIGHT, TabICLClassifier


def frames(raw):
    compact = compact_features(raw)
    budget = design(compact)[0]
    # Preserve who spent most and least within each passenger, without exposing service amounts.
    ordered = budget.copy()
    bills = compact[BILLS].where(compact[BILLS] > 0)
    for name, values in bills.rank(axis=1, method='average', pct=True).items():
        ordered[name+'_relative_order'] = values.fillna(0)
    return budget, ordered


def predict(frame, y, t, q, f):
    model = TabICLClassifier(n_estimators=2, model_path=WEIGHT, allow_auto_download=False, device='cpu',
                             batch_size=1, n_jobs=2, random_state=73+int(f), kv_cache=False,
                             use_amp=False, use_fa3=False)
    model.fit(frame.iloc[t], y[t])
    np.testing.assert_array_equal(model.classes_, [False, True])
    return model.predict_proba(frame.iloc[q])[:, 1]


def run():
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest() == DIGEST
    row, y, _, groups, _ = load()
    raw = pd.read_csv(ROOT/'data/raw/train.csv', usecols=lambda c: c not in ['Name', 'Transported']).iloc[row].reset_index(drop=True)
    budget, order = frames(raw)
    out = ART/'nov23_pretrained_spending_views'
    out.mkdir(exist_ok=True)
    plans, history, mistakes = {}, {}, []
    for seed in [*map(str, SEEDS), 'v131']:
        fold, ref, scores, _ = read_opinions(seed, row, y)
        base = scores[:, 0]
        with np.load(ART/f'nov23_crossfamily_unlabelled_distillation/{seed}.npz') as z:
            for k, v in [('row', row), ('y', y), ('fold', fold), ('ref', ref)]:
                np.testing.assert_array_equal(z[k], v)
            all_amounts = z['teacher']
        if seed == 'v131':
            with np.load(ART/'nov23_budget_quarter_replay/original.npz') as z:
                np.testing.assert_array_equal(z['fold'], fold)
                tree = z['alternative']
        else:
            with np.load(ART/f'nov23_simplex_likelihood/{seed}.npz') as z:
                np.testing.assert_array_equal(z['fold'], fold)
                np.testing.assert_array_equal(z['row'], row)
                tree = z['budget_prior']
        cache = out/f'{seed}.npz'
        if cache.is_file():
            with np.load(cache) as z:
                for k, v in [('row', row), ('y', y), ('fold', fold), ('base', base)]:
                    np.testing.assert_array_equal(z[k], v)
                coarse, ordinal = z['budget'], z['ordinal']
        else:
            coarse, ordinal = np.full(len(y), np.nan), np.full(len(y), np.nan)
            for f in np.unique(fold):
                q = np.flatnonzero(fold == f)
                t = np.flatnonzero((fold != f) & ~np.isin(groups, groups[q]))
                assert not np.intersect1d(groups[t], groups[q]).size
                coarse[q] = predict(budget, y, t, q, f)
                ordinal[q] = predict(order, y, t, q, f)
                print(seed, int(f), 'fold completed', flush=True)
            np.savez_compressed(cache, row=row, y=y, fold=fold, base=base, budget=coarse, ordinal=ordinal)
        assert np.isfinite(coarse).all() and np.isfinite(ordinal).all()
        bank = dict(precise=all_amounts, tree=tree, budget=coarse, ordinal=ordinal,
                    tree_and_order=(tree+ordinal)/2)
        votes = {'reference': ref}
        for k, p in bank.items():
            votes[k] = p >= .5
            votes['quarter_'+k] = np.where((base >= .5) == ref, .75*base+.25*p >= .5, ref)
        hits = {k: v == y for k, v in votes.items()}
        plans[seed] = {k: dict(vs_reference=paired(h, ref == y),
                               vs_tree=paired(h, hits['quarter_tree'] if k.startswith('quarter_') else hits['tree']))
                       for k, h in hits.items()}
        if seed != 'v131':
            mistakes.append(((scores >= .5) != y[:, None]).all(1))
            for k, h in hits.items():
                history.setdefault(k, []).append(h)
        print(seed, plans[seed]['quarter_ordinal'], flush=True)
    h = {k: np.stack(v) for k, v in history.items()}
    miss = (~h['reference']).sum(0)
    common = np.stack(mistakes).all(0)
    assert common.sum() == 332
    grouped = {k: dict(gain95_pp=interval((v.astype(float)-h['reference']).mean(0), groups),
                       correct_common=int((v.all(0)&common).sum()),
                       correct_by_errors=[int((v.all(0)&(miss == i)).sum()) for i in range(4)])
               for k, v in h.items()}
    write(ROOT/'reports/nov23_bill_value/pretrained.json', dict(plans=plans, grouped=grouped))


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
