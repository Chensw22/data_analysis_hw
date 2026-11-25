"""Predict transport jointly with observed spending using a frozen multiclass model.

Generate each split with .venv-mitra. Run --report with the main project environment.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/nov25_mitra_four_outcomes'
FEES = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
MODES = ('binary', 'spending_marginal', 'spending_conditional', 'shuffled_marginal')


def infer(model, support, target, query, activity, classes):
    import torch
    import mitra_candidate as mitra

    labels = torch.as_tensor(target, dtype=torch.int64, device='mps')
    chunks = []
    for start in range(0, len(query), 2318):
        batch = query[start:start + 2318]
        logits = mitra.model_forward(model, support, labels, batch)[:, :classes]
        prob = logits.softmax(1)
        if classes == 2:
            values = prob[:, 1, None]
        else:
            # Class index is 2 * observed_activity + transported.
            observed = torch.as_tensor(activity[start:start + len(batch)].astype(int), device='mps')
            by_activity = logits.reshape(-1, 2, 2)
            selected = by_activity[torch.arange(len(batch), device='mps'), observed]
            values = torch.stack([prob[:, [1, 3]].sum(1), selected.softmax(1)[:, 1]], dim=1)
        chunks.append(values.cpu().numpy())
    return np.vstack(chunks)


def generate(test):
    import torch
    import mitra_candidate as mitra
    from broad_error_transport_mitra_fp32_head_n25 import load, specs

    assert torch.backends.mps.is_available()
    torch.set_num_threads(3)
    torch.mps.set_per_process_memory_fraction(.65)
    entries, train, ids, _ = specs()
    roster = pd.concat([train, pd.read_csv(ROOT / 'data/raw/test.csv')], ignore_index=True)
    np.testing.assert_array_equal(roster.PassengerId, ids)
    active = roster[FEES].gt(0).any(axis=1).to_numpy()
    OUT.mkdir(exist_ok=True)
    for entry in entries:
        tag, _, _, t, q, old, _ = entry
        if (tag == 'test') != test:
            continue
        path = OUT / f'{tag}.npz'
        if not path.exists():
            saved, frame = load(entry, train, ids)
            support = torch.as_tensor(saved.support, device='mps')
            ask = torch.as_tensor(saved.preprocessor.transform(frame), device='mps')
            target = saved.labels.astype(int)
            np.testing.assert_array_equal(target, train.Transported.iloc[t].astype(int))
            del saved
            network, _ = mitra.load_initial_model(mitra.CHECKPOINT_DIR, 'mps')
            network.eval()
            mask = active[t].astype(int)
            null = mask.copy()
            rng = np.random.default_rng(625)
            for side in (0, 1):
                idx = np.flatnonzero(target == side)
                null[idx] = rng.permutation(mask[idx])
            np.testing.assert_array_equal(np.bincount(2 * mask + target), np.bincount(2 * null + target))
            with torch.inference_mode():
                base = infer(network, support, target, ask, active[q], 2)
                new = infer(network, support, 2 * mask + target, ask, active[q], 4)
                placebo = infer(network, support, 2 * null + target, ask, active[q], 4)
            predictions = np.column_stack([base, new, placebo[:, 0]])
            assert np.isfinite(predictions).all() and ((predictions >= 0) & (predictions <= 1)).all()
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, t=t, labels=target, q=q, passenger=ids.iloc[q].to_numpy(str),
                                    scores=predictions, native=old, checkpoint=mitra.WEIGHT_SHA)
            path.with_suffix('.tmp').replace(path)
            del network, support, ask
            mitra.release_memory()
        with np.load(path) as z:
            for key, value in [('t', t), ('q', q), ('passenger', ids.iloc[q]), ('checkpoint', mitra.WEIGHT_SHA)]:
                np.testing.assert_array_equal(z[key], value)
        print(tag, 'complete', flush=True)


def report():
    from full_roster_conditional_pool_n23 import data
    from full_roster_six_failure_archive_n24 import prediction_sets

    raw, y, groups, full_fold = data()
    with np.load(ROOT / 'artifacts/nov25_historical_nine/historical.npz') as z:
        rows, fold, nine, v131 = [z[key] for key in ('rows', 'fold', 'nine', 'v131')]
        np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[rows])
        np.testing.assert_array_equal(z['y'], y[rows])
    truth = y[rows]
    _, panels = prediction_sets(raw, y, groups, full_fold, rows, fold)
    agreement = ((np.column_stack(panels[1]) >= .5) == nine[:, None]).all(1)
    native = np.empty(len(rows))
    scores = np.empty((len(rows), len(MODES)))
    for k in np.unique(fold):
        use = fold == k
        with np.load(OUT / f'fold{k}.npz') as z:
            np.testing.assert_array_equal(z['q'], rows[use])
            np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[rows[use]])
            np.testing.assert_array_equal(z['labels'], y[z['t']])
            assert not np.intersect1d(z['q'], z['t']).size
            scores[use] = z['scores']
            native[use] = z['native']
    free = ((native >= .5) == v131) & (v131 == nine)
    result = dict(rows=len(rows), consensus_rows=int(agreement.sum()),
                  consensus_errors=int((agreement & (nine != truth)).sum()), modes={})
    for j, name in enumerate(MODES):
        raw_correct = (scores[:, j] >= .5) == truth
        delta_binary = raw_correct.astype(int) - ((scores[:, 0] >= .5) == truth)
        result['modes'][name] = dict(raw_accuracy=float(raw_correct.mean()),
                                     net_vs_frozen_binary_by_fold=[int(delta_binary[fold == k].sum()) for k in np.unique(fold)])
        for weight in (1., .25):
            mixed = (1 - weight) * native + weight * scores[:, j]
            predicted = np.where(free, mixed >= .5, nine)
            effect = (predicted == truth).astype(int) - (nine == truth)
            result['modes'][name][str(weight)] = dict(
                corrected=int((effect > 0).sum()), harmed=int((effect < 0).sum()),
                consensus_corrected=int((effect[agreement] > 0).sum()),
                consensus_harmed=int((effect[agreement] < 0).sum()),
                fold_net=[int(effect[fold == k].sum()) for k in np.unique(fold)])
    destination = ROOT / 'reports/nov23_matched_learning_curve/mitra_four_outcomes.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--test', action='store_true')
    group.add_argument('--report', action='store_true')
    arguments = parser.parse_args()
    if arguments.report:
        report()
    else:
        generate(arguments.test)
