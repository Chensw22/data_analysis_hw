"""Evaluate the v96 Mitra base on the full-roster grouped folds used by four other models."""
import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'FALSE'

import numpy as np

from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, data, predictor


def fit(roster, train, query, labels, counts):
    import torch

    base = counts.base
    train_y = base.binary_labels(labels, len(train))
    features = roster.plain.iloc[train]
    base.seed_everything(counts.SEED, 'mps')
    prep = counts.CountsPreprocessor(roster).fit(features, train_y, counts.SEED)
    support = prep.transform(features)
    assert support.shape == (len(train), 15) and prep.active_columns == counts.COLUMNS
    model, config = counts.training.fresh_model('mps')
    optimizer = base.optimizer_for(model)
    x = torch.as_tensor(support, device='mps')
    target = torch.as_tensor(train_y, device='mps')
    rng = np.random.default_rng(counts.SEED)
    for step in range(1, counts.STEPS+1):
        info = counts.training.episode_step(model, optimizer, x, target, train, rng, step)
        assert info['step'] == step and info['disjoint_ids']
    del optimizer, x, target
    base.release_memory()
    predictor = counts.CountsPredictor(prep, config, model.state_dict(), support, train_y, 'mps',
                                       {}, query_chunk=2318)
    del model
    base.release_memory()
    try:
        p = predictor.predict_proba(roster.plain.iloc[query])[:, 1]
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        return p
    finally:
        predictor.release()
        base.release_memory()


def train_folds():
    import torch
    import mitra_counts_fixed15_online_first as counts

    assert torch.backends.mps.is_available()
    counts.verify_reference()
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.65)
    raw, y, groups, fold = data()
    with np.load(OUT / 'cat.npz') as z:
        np.testing.assert_array_equal(z['passenger'], raw.PassengerId)
        np.testing.assert_array_equal(z['fold'], fold)
    roster = counts.load_roster()
    np.testing.assert_array_equal(roster.passenger_ids.iloc[:len(y)], raw.PassengerId)
    p = np.empty(len(y))
    for f in np.unique(fold):
        t, q = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        assert not np.isin(groups[q], groups[t]).any()
        path = OUT / f'mitra96_fold_{int(f)}.npz'
        if not path.exists():
            print('training fold', int(f), 'passengers', len(t), flush=True)
            pred = fit(roster, t, q, y[t], counts)
            pending = path.with_suffix('.tmp')
            with pending.open('wb') as stream:
                np.savez_compressed(stream, train=t, training_labels=y[t], query=q, p=pred, steps=counts.STEPS,
                                    checkpoint=counts.base.WEIGHT_SHA)
            pending.replace(path)
        with np.load(path) as z:
            for k, expected in [('train', t), ('training_labels', y[t]), ('query', q),
                                ('steps', 15), ('checkpoint', counts.base.WEIGHT_SHA)]:
                np.testing.assert_array_equal(z[k], expected)
            p[q] = z['p']
        print('completed fold', int(f), 'correct', int(((p[q]>=.5)==y[q]).sum()), 'of', len(q), flush=True)
    np.savez_compressed(OUT / 'mitra96.npz', passenger=raw.PassengerId.to_numpy(str), y=y, fold=fold, p=p)


def original_fit(family):
    import pandas as pd

    train, y, _, _ = data()
    test = pd.read_csv(ROOT / 'data/raw/test.csv', usecols=lambda c: c != 'Name')
    raw = pd.concat([train, test], ignore_index=True)
    predict = predictor(family, raw, y)
    with np.load(ROOT / 'artifacts/v96_oof_completion/oof.npz') as z:
        row, fold = z['dev_ids'], z['fold_id']
        np.testing.assert_array_equal(z['passenger_ids'], train.PassengerId.iloc[row].to_numpy(str))
    p = np.full(len(row), np.nan)
    for f in np.unique(fold):
        t, q = row[fold != f], row[fold == f]
        p[fold == f] = predict(t, q, int(f))
    full = predict(np.arange(len(y)), np.arange(len(y), len(raw)), 0)
    assert np.isfinite(p).all() and np.isfinite(full).all()
    np.savez_compressed(OUT / f'original_{family}.npz', row=row, y=y[row], fold=fold, p=p,
                        test_ids=test.PassengerId.to_numpy(str), test=full)


def original_replay(y, hard):
    import json
    import pandas as pd
    from broad_pairwise_adjudication_n23 import paired
    from kaggle_workflow import sha256
    from released_correction_audit_n23 import development
    from audit_flip_capacity_n23 import correct_count

    original_y, archived, _ = development()
    with np.load(ROOT / 'artifacts/v96_oof_completion/oof.npz') as z:
        row, original_fold, v96 = z['dev_ids'], z['fold_id'], z['prob']
        np.testing.assert_array_equal(y[row], original_y)
    test_id = pd.read_csv(ROOT / 'data/raw/test.csv', usecols=['PassengerId']).PassengerId
    views, tests = [], []
    for family in ['v44_without_name', 'tabicl']:
        with np.load(OUT / f'original_{family}.npz') as z:
            for k, expected in [('row', row), ('y', original_y), ('fold', original_fold), ('test_ids', test_id)]:
                np.testing.assert_array_equal(z[k], expected)
            views.append(z['p'])
            tests.append(z['test'])
    reference = archived[131]
    art = ROOT / 'artifacts'
    test_votes = {}
    for name in ['proposal_online_first_counts15', 'proposal_v131_party_consensus']:
        path = art / name / 'submission.csv'
        manifest = json.loads((ROOT / 'reports' / name / 'manifest.json').read_text())
        assert sha256(path) == manifest['sha256']
        table = pd.read_csv(path)
        np.testing.assert_array_equal(table.PassengerId, test_id)
        test_votes[name] = table.Transported.to_numpy()
    soft = pd.read_csv(art / 'proposal_online_first_counts15/test_probabilities.csv', float_precision='round_trip')
    np.testing.assert_array_equal(soft.PassengerId, test_id)
    np.testing.assert_array_equal(soft.probability >= .5, test_votes['proposal_online_first_counts15'])
    np.testing.assert_array_equal(soft.probability, soft.reload_probability)
    best = test_votes['proposal_v131_party_consensus']
    keep = best != (soft.probability.to_numpy() >= .5)
    scored = json.loads((ROOT/'reports/nov23_flip_capacity/primary_scores.json').read_text())['public_scores']
    historical = {}
    for path in (ROOT/'reports').glob('*/online_attempts/*.json'):
        receipt = json.loads(path.read_text())
        ref = str(receipt.get('submission_ref'))
        if ref not in scored or receipt.get('evaluation', {}).get('Status') != 'COMPLETE':
            continue
        csv = art/path.parents[1].name/'submission.csv'
        assert sha256(csv) == receipt['file_sha256']
        table = pd.read_csv(csv)
        np.testing.assert_array_equal(table.PassengerId, test_id)
        historical[ref] = path.parents[1].name, table.Transported.to_numpy(), correct_count(scored[ref], len(best))
    assert set(historical) == set(scored)
    result = {}
    for label, validation, testing in [('three_family_mean', [v96, *views], [soft.probability, *tests]),
                                       ('two_family_mean', views, tests)]:
        local = np.where(reference == (v96 >= .5), np.mean(validation, axis=0) >= .5, reference)
        query = np.where(keep, best, np.mean(testing, axis=0) >= .5)
        distances, upper = [], []
        for name, old, correct in historical.values():
            d = int((old != query).sum())
            distances.append(d)
            upper.append((correct+d, d, name))
        limit, distance, source = min(upper)
        delta = (local == original_y).astype(int) - (reference == original_y)
        result[label] = dict(**paired(local==original_y, reference==original_y),
            hard_correct=int((hard[row] & (local==original_y)).sum()),
            by_fold=[int(delta[original_fold==f].sum()) for f in np.unique(original_fold)],
            locked_test_rows=int(keep.sum()), changed_vs_v131=int((query!=best).sum()),
            changed_minimum_vs_submitted=min(distances), public_correct_upper_bound=limit,
            bounding_file=source, distance_to_bounding_file=distance)
    return result


def summarize():
    from broad_booking_policy_replay_n23 import interval
    from broad_geographic_transfer_n23 import BILLS
    from broad_pairwise_adjudication_n23 import paired
    from broad_partition_evidence_n23 import write
    from broad_tree_field_dividends_n23 import hard_cohorts

    raw, y, groups, fold = data()
    fields = []
    for name in [*FAMILIES, 'mitra96']:
        with np.load(OUT / f'{name}.npz') as z:
            for k, expected in [('passenger', raw.PassengerId), ('y', y), ('fold', fold)]:
                np.testing.assert_array_equal(z[k], expected)
            fields.append(z['p'])
    scores = np.column_stack(fields)
    paid = raw[BILLS].gt(0).any(axis=1).to_numpy()
    base = (scores[:, 0]+scores[:, 2])/2
    ref = base >= .5
    p = scores[:, -1]
    old_hits = (scores[:, :-1]>=.5) == y[:, None]
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as z:
        dev = np.isin(np.arange(len(y)), z['development'])
    _, _, hard_dev, _ = hard_cohorts(np.flatnonzero(dev), y[dev])
    hard = np.zeros(len(y), bool)
    hard[dev] = hard_dev
    results = {}
    for name, mask in [('all', np.ones(len(y), bool)), ('development', dev), ('additional', ~dev), ('hard', hard)]:
        miss = mask & ~old_hits.any(1)
        fixed = miss & ((p >= .5) == y)
        hit = mask & old_hits.all(1)
        harmed = hit & ((p >= .5) != y)
        confident = np.abs(p-.5) >= np.mean(np.abs(scores[:, :-1]-.5), axis=1)
        arms = {'mitra': p >= .5, 'three_family_mean': (2*base+p)/3 >= .5}
        mix = arms['three_family_mean']
        change = (mix == y).astype(int) - ((p >= .5) == y)
        results[name] = dict(rows=int(mask.sum()), four_all_wrong=int(miss.sum()), new_reachable=int(fixed.sum()),
                             four_all_correct=int(hit.sum()), mitra_opposes_correct=int(harmed.sum()),
                             new_paid_transported=int((fixed & paid & y).sum()),
                             new_no_positive_stayed=int((fixed & ~paid & ~y).sum()),
                             higher_confidence_fixed=int((fixed & confident).sum()),
                             higher_confidence_harmed=int((harmed & confident).sum()),
                             vs_mitra=paired(mix[mask]==y[mask], (p[mask]>=.5)==y[mask]),
                             net_vs_mitra_by_fold=[int(change[mask & (fold==f)].sum()) for f in np.unique(fold)],
                             delta_vs_mitra_interval_pp=interval(change[mask], groups[mask]),
                             models={k:paired(v[mask]==y[mask], ref[mask]==y[mask]) for k,v in arms.items()})
    results['original_replay'] = original_replay(y, hard)
    write(ROOT / 'reports/nov23_matched_learning_curve/full_roster_mitra.json', results)
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'summarize', 'v44_without_name', 'tabicl'])
    action = parser.parse_args().action
    if action == 'train':
        train_folds()
    elif action in ('v44_without_name', 'tabicl'):
        original_fit(action)
    else:
        summarize()
