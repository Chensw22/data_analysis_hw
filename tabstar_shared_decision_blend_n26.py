"""Test whether original TabSTAR adds actionable evidence to a frozen cross-architecture blend."""
import argparse
import json
import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import champions
from outside_panel_coverage_n26 import archives, projection
from original_tabstar_probe_n26 import ART, SHA
from target_mix_expert_routing_n26 import ROOT, ticker

OTHER = ['Mitra2_adapted', 'TabDPT', 'LimiX2']


def run(tuned):
    raw, y, groups, folds, views, _ = projection(include_full=True)
    outsiders, _ = archives(raw, y, groups, folds, True)
    nine, known, _, _ = champions()
    peers = np.column_stack([outsiders[n] for n in OTHER])
    evidence = logit(np.clip(peers, .002, .998))
    baseline = np.median(evidence, axis=1)
    neural = np.full(len(y), np.nan)
    for k in np.unique(folds):
        path = ART / (f'tuned{k}.npz' if tuned else f'fold{k}.npz')
        with np.load(path) as z:
            q, t = z['q'], z['t']
            np.testing.assert_array_equal(q, np.flatnonzero(folds == k))
            np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[q])
            np.testing.assert_array_equal(z['train_y'], y[t])
            assert z['checkpoint'] == SHA and not np.isin(groups[q], groups[t]).any()
            if tuned:
                np.testing.assert_array_equal(t, np.flatnonzero(folds != k))
            neural[q] = z['finetuned' if tuned else 'readout']
    assert np.isfinite(neural).all() and np.all((neural >= 0) & (neural <= 1))
    new = logit(np.clip(neural, .002, .998))
    # Give the median of three existing architectures two votes in log odds and TabSTAR one.
    blend = (2 * baseline + new) / 3
    meta = {}
    for tag, (valid, parent) in views.items():
        assert known[valid].all()
        common = valid & (parent == nine)
        dissent = (neural >= .5) != parent
        peer_flip = (baseline >= 0) != parent
        combined_flip = (blend >= 0) != parent
        rules = dict(three_architectures=peer_flip, blend=combined_flip,
                     newly_flipped=combined_flip & ~peer_flip,
                     new_model_veto=peer_flip & ~combined_flip,
                     three_with_tabstar_confirmation=peer_flip & dissent,
                     three_with_confident_confirmation=peer_flip & dissent & (np.abs(new) >= np.log(3)))
        result = {name: {maskname: ticker(mask, parent != y, rule) for maskname, mask in
                        [('all_paired', valid), ('common', common)]} for name, rule in rules.items()}
        # When applied as a veto, newly retained decisions help where the release was correct.
        reject = rules['new_model_veto'] & common
        result['increment_vs_three'] = dict(
            realigned_correct=int(np.sum(common & ((blend >= 0) == y) & ((baseline >= 0) != y))),
            realigned_wrong=int(np.sum(common & ((blend >= 0) != y) & ((baseline >= 0) == y))),
            veto_beneficial=int(np.sum(reject & (parent == y))), veto_costly=int(np.sum(reject & (parent != y))))
        meta[tag] = result
        print(tag, result['increment_vs_three'], result['blend']['common'], flush=True)
    name = 'finetuned' if tuned else 'readout'
    path = ROOT / f'reports/nov23_matched_learning_curve/tabstar_{name}_blend.json'
    path.write_text(json.dumps(dict(models=OTHER, variant=name, views=meta), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tuned', action='store_true')
    with threadpool_limits(1):
        run(parser.parse_args().tuned)
