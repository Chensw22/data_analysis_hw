"""Replace repeated support entries by real distinct people at identical label positions."""
import hashlib
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from broad_class_cost_path_n23 import REPORT
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_passenger_transport_n23 import manifest_coordinates
from broad_replicated_prior_n23 import contexts
from tabicl_routed_context_n23 import DIGEST, WEIGHT, TabICLClassifier, torch


def replace_repetitions(template, training, labels, coordinates, label, rng):
    positions = defaultdict(list)
    for i, passenger in enumerate(template):
        positions[passenger].append(i)
    extra_slots = [slot for passenger, slots in positions.items() if labels[passenger] == label
                   for slot in slots[1:]]
    assert len(extra_slots) == 1024
    candidates = rng.permutation(np.setdiff1d(training[labels[training] == label], list(positions)))
    distances = cdist(coordinates[template[extra_slots]], coordinates[candidates], 'sqeuclidean')
    rows, neighbors = linear_sum_assignment(distances)
    np.testing.assert_array_equal(rows, np.arange(len(extra_slots)))
    changed = {}
    for name, selected in [('similar', candidates[neighbors]), ('random', candidates[:len(extra_slots)])]:
        proposal = template.copy()
        proposal[extra_slots] = selected
        np.testing.assert_array_equal(labels[proposal], labels[template])
        assert np.unique(proposal).size == len(template) and set(positions).issubset(proposal)
        changed[name] = proposal
    return changed


def probability(frame, labels, support, query, seed, reset_index=False):
    model = TabICLClassifier(n_estimators=2, model_path=WEIGHT, allow_auto_download=False, device='cpu',
                             batch_size=1, n_jobs=2, random_state=seed, kv_cache=False,
                             use_amp=False, use_fa3=False)
    inputs = frame.iloc[support]
    model.fit(inputs.reset_index(drop=True) if reset_index else inputs, labels[support])
    np.testing.assert_array_equal(model.classes_, [False, True])
    p = model.predict_proba(frame.iloc[query])
    np.testing.assert_allclose(p.sum(1), 1., atol=1e-5)
    return p[:, 1]


def main():
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest() == DIGEST
    row, y, frame, groups, _ = load()
    coordinates = manifest_coordinates(frame)
    seed = SEEDS[0]
    with np.load(ART / f'nov23_replicated_prior/{seed}.npz') as data:
        np.testing.assert_array_equal(data['row'], row)
        np.testing.assert_array_equal(data['y'], y)
        fold = data['fold']
    plan = {}
    for f in np.unique(fold):
        q = np.flatnonzero(fold == f)
        t = np.flatnonzero((fold != f) & ~np.isin(groups, groups[q]))
        probs = defaultdict(list)
        index_delta = []
        for label, template in zip([False, True], contexts(t, y, 183 + int(f))[1:3]):
            replacements = replace_repetitions(template, t, y, coordinates, label,
                                               np.random.default_rng(895 + int(f) + 10 * label))
            for name, support in {'duplicate': template, **replacements}.items():
                probs[name].append(probability(frame, y, support, q, 43 + int(f)))
            reset = probability(frame, y, template, q, 43 + int(f), reset_index=True)
            index_delta.append(float(np.max(np.abs(reset - probs['duplicate'][-1]))))
        plan[str(f)] = dict(query=len(q), max_index_delta=index_delta, arms={})
        for name, (low, high) in probs.items():
            diff = high - low
            plan[str(f)]['arms'][name] = dict(lower=int((diff < 0).sum()), equal=int((diff == 0).sum()),
                                              mean_difference=float(diff.mean()), median_difference=float(np.median(diff)))
        print(f, plan[str(f)], flush=True)
    write(REPORT / 'distinct.json', dict(seed=int(seed), by_fold=plan))


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
