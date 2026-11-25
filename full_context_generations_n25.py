"""Evaluate the 15-input frozen generations with near-deployment training size and grouped queries."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/nov25_full_context_generations'
DEST = ROOT / 'reports/nov23_matched_learning_curve/full_context_generations.json'


def infer(generation, prep, support_frame, query_frame, labels):
    import torch
    import mitra_candidate as mitra
    from mitra_generation_candidate import CHECKPOINTS, verify_checkpoint
    from mitra_four_outcomes_n25 import infer as infer_classes

    config = verify_checkpoint(generation)
    path = ROOT / CHECKPOINTS[generation]['directory'] / 'model.safetensors'
    state = mitra.load_file(str(path), device='cpu')
    net = mitra.make_model(config, state, device='mps').eval()
    del state
    support = torch.as_tensor(prep.transform(support_frame), device='mps')
    query = torch.as_tensor(prep.transform(query_frame), device='mps')
    with torch.inference_mode():
        p = infer_classes(net, support, labels, query, np.zeros(len(query_frame), bool), 2).ravel()
    del net, support, query
    mitra.release_memory()
    return p


def calculate():
    import torch
    import mitra_counts_fixed15_online_first as counts
    from mitra_generation_candidate import CHECKPOINTS
    from full_roster_booking_validation_n23 import data

    assert torch.backends.mps.is_available()
    torch.set_num_threads(3)
    torch.mps.set_per_process_memory_fraction(.65)
    roster = counts.load_roster()
    raw, y, groups, eight = data()
    np.testing.assert_array_equal(raw.PassengerId, roster.passenger_ids.iloc[:len(y)])
    OUT.mkdir(exist_ok=True)
    with np.load(ROOT / 'artifacts/nov25_historical_nine/historical.npz') as z:
        historical_rows, historical_folds = z['rows'], z['fold']
    for tag, folds in [('historical', historical_folds), ('eight', eight)]:
        for f in np.unique(folds):
            path = OUT / f'{tag}_{f}.npz'
            old_path = ROOT / 'artifacts/nov25_mitra_four_outcomes' / f'fold{f}.npz'
            if tag == 'historical':
                with np.load(old_path) as z:
                    t, q, v2 = z['t'], z['q'], z['scores'][:, 0]
                    np.testing.assert_array_equal(z['labels'], y[t])
                    np.testing.assert_array_equal(z['checkpoint'], CHECKPOINTS['v2']['sha256'])
                    np.testing.assert_array_equal(q, historical_rows[folds == f])
            else:
                t, q = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
                assert not np.intersect1d(groups[t], groups[q]).size
                with np.load(ROOT / 'artifacts/nov23_full_roster_booking_validation' / f'mitra96_fold_{f}.npz') as z:
                    for key, val in [('train', t), ('query', q), ('training_labels', y[t]), ('checkpoint', CHECKPOINTS['v2']['sha256'])]:
                        np.testing.assert_array_equal(z[key], val)
            if not path.exists():
                context, queries = roster.plain.iloc[t], roster.plain.iloc[q]
                prep = counts.CountsPreprocessor(roster).fit(context, y[t].astype(int), counts.SEED)
                assert prep.active_columns == counts.COLUMNS
                if tag == 'eight' or f == 0:
                    replay = infer('v2', prep, context, queries, y[t].astype(int))
                    if tag == 'historical':
                        np.testing.assert_array_equal(replay, v2)
                    v2 = replay
                v1 = infer('v1', prep, context, queries, y[t].astype(int))
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream, t=t, q=q, passenger=raw.PassengerId.iloc[q].to_numpy(str),
                                        labels=y[t], v1=v1, v2=v2,
                                        checkpoints=[CHECKPOINTS[k]['sha256'] for k in ('v1', 'v2')])
                path.with_suffix('.tmp').replace(path)
            print(tag, int(f), len(t), len(q), flush=True)


def report():
    from full_roster_conditional_pool_n23 import data
    from full_roster_six_failure_archive_n24 import prediction_sets

    raw, y, groups, eight = data()
    with np.load(ROOT / 'artifacts/nov25_historical_nine/historical.npz') as z:
        rows_h, fold_h, inc_h, prev_h = [z[k] for k in ('rows', 'fold', 'nine', 'v131')]
    _, panels = prediction_sets(raw, y, groups, eight, rows_h, fold_h)
    result = {}
    aligned = {}
    group_size = raw.groupby(groups).PassengerId.transform('size').to_numpy()
    for tag, rows, fold, panel in [('historical', rows_h, fold_h, panels[1]), ('eight', np.arange(len(y)), eight, panels[0])]:
        if tag == 'eight':
            with np.load(ROOT / 'artifacts/nov25_production_anchor_cpu/oof.npz') as z:
                np.testing.assert_array_equal(z['row'], rows)
                np.testing.assert_array_equal(z['fold'], fold)
                native, inc, eligible = z['v96'], z['certified_nine'].astype(bool), z['certified_v131_equals_v96']
                prev = native >= .5
        else:
            inc, prev, eligible = inc_h, prev_h, np.ones(len(rows), bool)
            with np.load(ROOT / 'artifacts/nov25_mitra_fp32_head/oof.npz') as z:
                np.testing.assert_array_equal(z['row'], rows)
                native = z['native']
        p = np.empty((len(rows), 2))
        context_overlap = np.zeros(len(rows), bool)
        for f in np.unique(fold):
            use = fold == f
            with np.load(OUT / f'{tag}_{f}.npz') as z:
                np.testing.assert_array_equal(z['q'], rows[use])
                np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[rows[use]])
                np.testing.assert_array_equal(z['labels'], y[z['t']])
                context_overlap[use] = np.isin(groups[rows[use]], groups[z['t']])
                p[use] = np.column_stack([z['v1'], z['v2']])
        assert ((p >= 0) & (p <= 1)).all()
        v1, v2 = p.T
        free = eligible & ((native >= .5) == prev) & (prev == inc)
        votes = np.column_stack(panel) >= .5
        agree = (votes == inc[:, None]).all(1)
        native_agree = (votes == (native[:, None] >= .5)).all(1)
        aligned[tag] = {}
        methods = {'v1': v1, 'v2': v2, 'generations_mean': p.mean(1),
                   'v1_and_tuned_mean': .5 * (v1 + native), 'three_model_mean': (v1 + v2 + native) / 3}
        cases = {'exact_baseline': eligible, 'orders_absent_from_training': eligible & ~context_overlap,
                 'all_archive_agree': eligible & agree}
        if tag == 'eight':
            cases['additional_orders'] = eligible & ~np.isin(groups[rows], groups[rows_h])
        entry = {}
        for name, candidate in methods.items():
            action = free & ((candidate >= .5) != inc)
            effect = np.where(action, np.where(inc == y[rows], -1, 1), 0)
            metrics = {}
            for key, mask in cases.items():
                metrics[key] = dict(rows=int(mask.sum()), original_errors=int((mask & (inc != y[rows])).sum()),
                                    fixed=int((effect[mask] > 0).sum()), harmed=int((effect[mask] < 0).sum()),
                                    fold_net=[int(effect[mask & (fold == f)].sum()) for f in np.unique(fold)])
            diff = (candidate >= .5) != (native >= .5)
            nd = np.where(diff, np.where((native >= .5) == y[rows], -1, 1), 0)
            native_masks = {'all_vs_native': np.ones(len(rows), bool), 'solo_vs_native': group_size[rows] == 1,
                            'party_vs_native': group_size[rows] > 1, 'unanimous_vs_native': native_agree,
                            'whole_order_unseen_vs_native': ~context_overlap}
            for key, mask in native_masks.items():
                metrics[key] = dict(rows=int(mask.sum()), fixed=int((nd[mask] > 0).sum()),
                                    harmed=int((nd[mask] < 0).sum()), fold_net=[int(nd[mask & (fold == f)].sum()) for f in np.unique(fold)])
            aligned[tag][name] = nd
            entry[name] = metrics
        if tag == 'historical':
            assert entry['v2']['exact_baseline']['fixed'] == 70 and entry['v2']['exact_baseline']['harmed'] == 58
        print(tag, [(k, v['exact_baseline']['fixed'], v['exact_baseline']['harmed']) for k, v in entry.items()], flush=True)
        result[tag] = entry
    result['same_passengers_vs_native'] = {}
    for name, old in aligned['historical'].items():
        new = aligned['eight'][name][rows_h]
        result['same_passengers_vs_native'][name] = dict(
            both_fixed=int(((old > 0) & (new > 0)).sum()), both_harmed=int(((old < 0) & (new < 0)).sum()),
            historical_fixed=int((old > 0).sum()), historical_harmed=int((old < 0).sum()),
            later_fixed=int((new > 0).sum()), later_harmed=int((new < 0).sum()))
    DEST.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='store_true')
    if parser.parse_args().report:
        report()
    else:
        calculate()
