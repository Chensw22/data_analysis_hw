"""Give all labeled passengers a prediction made without labels from their order."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/nov23_full_roster_booking_validation'
SEED = 20261123
FAMILIES = ('tabicl', 'cat', 'v44_without_name', 'budget')


def data():
    raw = pd.read_csv(ROOT / 'data/raw/train.csv', usecols=lambda c: c != 'Name')
    y = raw.pop('Transported').to_numpy(bool)
    g = raw.PassengerId.str.split('_').str[0].to_numpy()
    fold = np.full(len(y), -1)
    for f, (t, q) in enumerate(GroupKFold(8, shuffle=True, random_state=SEED).split(raw, groups=g)):
        assert not np.isin(g[q], g[t]).any()
        fold[q] = f
    assert len(y) == 8693 and (fold >= 0).all()
    return raw, y, g, fold


def predictor(name, raw, y):
    if name == 'v44_without_name':
        import audit_top5_references as audit
        framework, _, fit, _, version = audit.initialize_reference('v44', OUT)
        matrices = {False: framework.build_features(raw, False)}
        assert not any('name' in k.lower() for k in matrices[False][0].columns)
        def predict(t, q, f):
            return fit(version, matrices, t, y[t].astype(int), q, SEED+f)[0]
    elif name == 'tabicl':
        import hashlib
        import torch
        from tabicl_routed_context_n23 import CAT, DIGEST, WEIGHT, infer
        torch.set_num_threads(2)
        assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest() == DIGEST
        x = raw.drop(columns=['Cabin', 'PassengerId']).copy()
        x[['Deck', 'CabinNumber', 'Side']] = raw.Cabin.str.split('/', expand=True)
        x.CabinNumber = pd.to_numeric(x.CabinNumber)
        x[CAT] = x[CAT].fillna('__MISSING__').astype(str)
        def predict(t, q, f):
            return infer(x, y, t, q, SEED+f)
    else:
        from exaone_validation_data import compact_features
        from expanded_training_roster_n23 import fitting
        from spend_simplex_likelihood_n23 import design, learn
        x = compact_features(raw)
        if name == 'budget':
            x = design(x)[0]
        def predict(t, q, f):
            return fitting(x, y[t], t, q) if name == 'cat' else learn(x, y, t, q)
    return predict


def train(name):
    OUT.mkdir(exist_ok=True)
    raw, y, _, fold = data()
    predict = predictor(name, raw, y)
    p = np.full(len(y), np.nan)
    for f in np.unique(fold):
        t, q = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        p[q] = predict(t, q, int(f))
        print(name, int(f), len(t), len(q), flush=True)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.savez_compressed(OUT / f'{name}.npz', passenger=raw.PassengerId.to_numpy(str), fold=fold, y=y, p=p)


def summarize():
    from scipy.special import logit
    from broad_booking_policy_replay_n23 import interval
    from broad_geographic_transfer_n23 import BILLS
    from broad_pairwise_adjudication_n23 import paired
    from broad_partition_evidence_n23 import write
    from broad_tree_field_dividends_n23 import hard_cohorts
    from expanded_score_frontier_n23 import source_scores
    from expanded_training_roster_n23 import eight_fold
    from learning_size_gate_n23 import fit_offset
    from matched_consensus_errors_n23 import calibrated, fitting, matched, roster_fields, tree_offset
    raw, y, g, fold = data()
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as z:
        dev = np.isin(np.arange(len(y)), z['development'])
    assert dev.sum() == 6954
    columns = []
    for name in FAMILIES:
        with np.load(OUT / f'{name}.npz') as z:
            for k, v in [('y', y), ('fold', fold), ('passenger', raw.PassengerId.to_numpy(str))]:
                np.testing.assert_array_equal(z[k], v)
            columns.append(z['p'])
    scores = np.column_stack(columns)
    base = (scores[:, 0] + scores[:, 2])/2
    ref = base >= .5
    hits = (scores >= .5) == y[:, None]
    paid = raw[BILLS].gt(0).any(axis=1).to_numpy()
    summary = {}
    for label, use in [('development', dev), ('additional', ~dev), ('all', np.ones(len(y), bool))]:
        wrong = use & ~hits.any(1)
        summary[label] = dict(rows=int(use.sum()), base_acc=float((ref[use] == y[use]).mean()),
                              both_strong_wrong=int((~hits[use, 0] & ~hits[use, 1]).sum()),
                              four_wrong=int(wrong.sum()), oracle_acc=float(hits[use].any(1).mean()),
                              paid_transported=int((wrong & paid & y).sum()),
                              no_positive_stayed=int((wrong & ~paid & ~y).sum()),
                              accuracies=dict(zip(FAMILIES, hits[use].mean(0).tolist())))
    previous_rows = np.flatnonzero(dev)
    old_fold, _, _ = eight_fold(previous_rows, y[dev])
    old_scores = np.column_stack([p for _, p in source_scores(previous_rows, y[dev], old_fold)])
    old_wrong = np.all((old_scores >= .5) != y[dev, None], axis=1)
    new_wrong = ~hits[dev].any(1)
    assert old_wrong.sum() == 813
    _, _, hard, _ = hard_cohorts(previous_rows, y[dev])
    summary['original_errors'] = dict(shared_in_both=int((old_wrong & new_wrong).sum()),
                                      now_at_least_one_correct=int((old_wrong & ~new_wrong).sum()),
                                      new_shared_errors=int((~old_wrong & new_wrong).sum()),
                                      hard_still_unanimous_wrong=int((hard & new_wrong).sum()),
                                      hard_reached_by_family=dict(zip(FAMILIES, hits[dev][hard].sum(0).tolist())))
    # Score the extra cohort using only development labels from other orders.
    # Keep the same outer-fold models for the calibration and evaluation predictions.
    corrected = ref.copy()
    cal_sizes = []
    vote = scores[:,0] >= .5
    eligible = ((scores >= .5) == vote[:,None]).all(1) & (vote != paid)
    prior = np.where(vote, 1-scores.mean(1), scores.mean(1))
    odds = logit(prior.clip(.001, .999))
    error = vote != y
    own, peers = roster_fields(np.arange(len(y)), g.astype(int))
    fields = np.column_stack([own, peers])
    flags = {k: np.zeros(len(y), bool) for k in ['intercept', 'matched', 'tree']}
    for f in np.unique(fold):
        q = np.flatnonzero((fold == f) & ~dev)
        t = np.flatnonzero((fold == f) & dev & ~np.isin(g, g[q]))
        assert not np.isin(g[t], g[q]).any() and dev[t].all()
        corrected[q] = fit_offset(logit(scores.clip(.001,.999)), logit(base.clip(.001,.999)), y, t, q) >= .5
        cal_sizes.append(len(t))
        for direction in [False, True]:
            scope = eligible & (vote == direction)
            train, query = t[scope[t]], q[scope[q]]
            pairs = matched(train, error, prior, g)
            flags['intercept'][query] = calibrated(odds, error, train, query) >= .5
            flags['matched'][query] = fitting(fields, odds, error, train, query, pairs) >= .5
            flags['tree'][query] = tree_offset(fields, odds, error, train, query) >= .5
    use = ~dev
    before, after = ref[use] == y[use], corrected[use] == y[use]
    summary['additional_calibration'] = dict(**paired(after, before), training_counts=cal_sizes,
        delta_interval_pp=interval(after.astype(int)-before.astype(int), g[use]),
        net_by_fold=[int((after[fold[use]==f]).sum()-(before[fold[use]==f]).sum()) for f in np.unique(fold)])
    consensus = np.where(eligible, vote, ref)
    assert np.sum(use & eligible & (consensus != y)) == (
        summary['additional']['paid_transported'] + summary['additional']['no_positive_stayed'])
    summary['additional_consensus'] = dict(rows=int((use & eligible).sum()), methods={})
    for k, flip in flags.items():
        assert not np.any(flip & (~eligible | dev))
        result = consensus ^ flip
        delta = (result[use] == y[use]).astype(int) - (ref[use] == y[use]).astype(int)
        summary['additional_consensus']['methods'][k] = dict(
            vs_consensus=paired(result[use] == y[use], consensus[use] == y[use]),
            vs_reference=paired(result[use] == y[use], ref[use] == y[use]),
            delta_interval_pp=interval(delta, g[use]),
            net_by_fold=[int(delta[fold[use]==f].sum()) for f in np.unique(fold)],
            by_paid={str(v): [int((flip & (paid==v) & error).sum()), int((flip & (paid==v) & ~error).sum())]
                     for v in [False, True]})
    write(ROOT / 'reports/nov23_matched_learning_curve/full_roster_validation.json', summary)
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('family', choices=(*FAMILIES, 'summarize'))
    family = parser.parse_args().family
    if family == 'summarize':
        summarize()
    else:
        train(family)
