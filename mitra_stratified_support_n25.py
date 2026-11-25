"""Retrieve a frozen binary predictor's training context by observed spending activity."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mitra_four_outcomes_n25 import FEES, infer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/nov25_mitra_stratified_support'
SCORES = ROOT / 'artifacts/nov25_mitra_four_outcomes'


def generate(test):
    import torch
    import mitra_candidate as mitra
    from broad_error_transport_mitra_fp32_head_n25 import load, specs

    entries, train, ids, _ = specs()
    roster = pd.concat([train, pd.read_csv(ROOT / 'data/raw/test.csv')], ignore_index=True)
    np.testing.assert_array_equal(roster.PassengerId, ids)
    active = roster[FEES].gt(0).any(axis=1).to_numpy()
    torch.set_num_threads(3)
    torch.mps.set_per_process_memory_fraction(.65)
    OUT.mkdir(exist_ok=True)
    for entry in entries:
        tag, _, _, t, q, _, _ = entry
        if (tag == 'test') != test:
            continue
        path = OUT / f'{tag}.npz'
        if path.exists():
            continue
        saved, frame = load(entry, train, ids)
        target = saved.labels.astype(int)
        features = saved.support
        asks = saved.preprocessor.transform(frame)
        del saved
        model, _ = mitra.load_initial_model(mitra.CHECKPOINT_DIR, 'mps')
        model.eval()
        null = np.random.default_rng(9879).permutation(active[t])
        scores = np.empty((len(q), 2))
        with torch.inference_mode():
            for side in (False, True):
                use = active[q] == side
                query = torch.as_tensor(asks[use], device='mps')
                for j, assignment in enumerate((active[t], null)):
                    donor = assignment == side
                    context = torch.as_tensor(features[donor], device='mps')
                    scores[use, j] = infer(model, context, target[donor], query, active[q][use], 2).ravel()
        assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
        with path.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, t=t, q=q, passenger=ids.iloc[q].to_numpy(str), active=active[q],
                                scores=scores, checkpoint=mitra.WEIGHT_SHA)
        path.with_suffix('.tmp').replace(path)
        del model, query, context
        mitra.release_memory()
        print(tag, 'complete', flush=True)


def report():
    from full_roster_conditional_pool_n23 import data
    from full_roster_six_failure_archive_n24 import prediction_sets

    raw, y, g, f = data()
    with np.load(ROOT / 'artifacts/nov25_historical_nine/historical.npz') as z:
        rows, fold, previous, nine = [z[k] for k in ('rows','fold','v131','nine')]
    _, panels = prediction_sets(raw, y, g, f, rows, fold)
    agreement = ((np.column_stack(panels[1]) >= .5) == nine[:, None]).all(1)
    truth = y[rows]
    scores = np.empty((len(rows), 2))
    full = np.empty(len(rows))
    native = np.empty(len(rows))
    for k in np.unique(fold):
        use = fold == k
        with np.load(OUT / f'fold{k}.npz') as z, np.load(SCORES / f'fold{k}.npz') as original:
            for key in ('q', 't', 'passenger', 'checkpoint'):
                np.testing.assert_array_equal(z[key], original[key])
            np.testing.assert_array_equal(z['q'], rows[use])
            np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[rows[use]])
            np.testing.assert_array_equal(z['active'], raw.iloc[rows[use]][FEES].gt(0).any(axis=1))
            scores[use] = z['scores']
            full[use] = original['scores'][:, 0]
            native[use] = original['native']
    eligible = ((native >= .5) == previous) & (previous == nine)
    binary = np.where(eligible, full >= .5, nine)
    results = {}
    for j, label in enumerate(('matched', 'shuffled')):
        for weight in (1., .5):
            p = weight * scores[:, j] + (1 - weight) * full
            pred = np.where(eligible, p >= .5, nine)
            net = (pred == truth).astype(int) - (nine == truth)
            diff = (pred == truth).astype(int) - (binary == truth)
            results[f'{label}_{weight}'] = dict(corrected=int((net > 0).sum()), harmed=int((net < 0).sum()),
                fold_net=[int(net[fold == k].sum()) for k in np.unique(fold)],
                net_vs_frozen_by_fold=[int(diff[fold == k].sum()) for k in np.unique(fold)],
                consensus_fixed=int((net[agreement] > 0).sum()), consensus_harmed=int((net[agreement] < 0).sum()))
    (ROOT / 'reports/nov23_matched_learning_curve/mitra_stratified_support.json').write_text(
        json.dumps(results, indent=2) + '\n')
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--test', action='store_true')
    mode.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.report:
        report()
    else:
        generate(args.test)
