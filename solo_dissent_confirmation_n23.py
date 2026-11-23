"""Compare a solo decision confirmed by two different models against each model alone."""
import numpy as np
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_tree_field_dividends_n23 import hard_cohorts
from pending_family_coverage_n23 import DONOR, PAIRS, development_votes, read, rules
from solo_dissent_neighbors_n23 import REPORT


def run():
    row, y, frame, groups, _ = load()
    solo = frame.BookingSize.to_numpy() == 1
    _, _, hard, right = hard_cohorts(row, y)
    weight = read(PAIRS/'test_budget.json')['test']['change_150']['weight']
    history, plans = {}, {}
    for seed in [*map(str, SEEDS), 'v131']:
        ref, pending = development_votes(frame, groups, row, y, seed, weight)
        unchanged = ~rules(ref, pending)[0]['all']
        queued = pending[DONOR]
        with np.load(ART/f'nov23_solo_peer_teacher/{seed}.npz') as teacher, np.load(
                ART/f'nov23_solo_dissent_neighbors/{seed}.npz') as local:
            for key, value in [('row', row), ('y', y), ('ref', ref)]:
                np.testing.assert_array_equal(teacher[key], value)
            for key in ['row', 'y', 'fold', 'ref', 'base']:
                np.testing.assert_array_equal(teacher[key], local[key])
            scope = local['eligible']
            assert not np.any(scope & ~solo)
            peer = teacher['companion_teacher']
            prob_peer_error = np.where(ref, 1-peer, peer)
            neighbor = local['passenger']
        neighbor_flip = neighbor > .5
        peer_flip = prob_peer_error > .5
        actions = dict(neighbor=neighbor_flip, teacher=peer_flip, both=neighbor_flip & peer_flip,
                       either=neighbor_flip | peer_flip, average=.5*(neighbor+prob_peer_error) > .5)
        plans[seed] = {}
        for name, action in actions.items():
            flip = scope & action
            hit = (ref ^ flip) == y
            extension = queued ^ (unchanged & flip)
            assert not np.any(unchanged & (queued != ref))
            plans[seed][name] = dict(vs_reference=paired(hit, ref == y),
                                     adding_outside_vs_queued=paired(extension == y, queued == y),
                                     combined_vs_reference=paired(extension == y, ref == y))
            if seed != 'v131':
                history.setdefault(name, []).append(hit)
        print(seed, plans[seed]['both'], flush=True)
    stable = {}
    for name, h in history.items():
        hits = np.stack(h)
        stable[name] = dict(hard_ever=int((solo & hard & hits.any(0)).sum()),
                            hard_stable=int((solo & hard & hits.all(0)).sum()),
                            stable_harms=int((solo & right & ~hits.any(0)).sum()))
    write(REPORT/'confirmation.json', dict(plans=plans, stable=stable))


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
