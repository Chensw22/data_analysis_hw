"""Transfer only the extra evidence supplied by modeling services jointly."""
import numpy as np
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from audit_stable_rescue_placebo_n23 import family_scores
from broad_geographic_transfer_n23 import BILLS, load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_tree_field_dividends_n23 import hard_cohorts
from fee_identity_assignment_n23 import LABEL, REPORT
from joint_fee_permutation_n23 import LABEL as JOINT


def main():
    row, y, frame, _, _ = load()
    _, _, hard, right = hard_cohorts(row, y)
    plans, history, payer_sets = {}, {}, []
    for seed in [*map(str, SEEDS), 'v131']:
        fold, base, ref, families = family_scores(seed, row, y, frame)
        with np.load(ART/f'nov23_{LABEL}/{seed}.npz') as first, np.load(ART/f'nov23_{JOINT}/{seed}.npz') as joint:
            for key, value in [('row', row), ('y', y), ('ref', ref), ('fold', fold), ('base', base)]:
                for cache in [first, joint]:
                    np.testing.assert_array_equal(cache[key], value)
            np.testing.assert_array_equal(first['anonymous'], joint['anonymous'])
            assigned, anon, full = first['assigned'], first['anonymous'], joint['joint']
        odds = lambda p: logit(p.clip(1e-8, 1-1e-8))
        shift = odds(full)-odds(assigned)
        values = {'joint': full, 'quarter_joint': .75*base+.25*full}
        for name, delta in [('identity', odds(assigned)-odds(anon)), ('joint_minus_additive', shift)]:
            for weight in [1., .25]:
                values[f'{name}_{weight}'] = expit(odds(base)+weight*delta)
        hit0 = (ref == y)
        scope = (frame.BookingSize.to_numpy() == 1) & frame[BILLS].gt(0).any(axis=1).to_numpy()
        scope &= ~ref & (np.column_stack(list(families.values())) < .5).all(1)
        plans[seed] = {}
        for name, v in values.items():
            vote = v >= .5 if name == 'joint' else np.where(ref == (base >= .5), v >= .5, ref)
            hit = vote == y
            plans[seed][name] = dict(**paired(hit, hit0), unanimous_fixed=int((scope & hit & ~hit0).sum()),
                                     unanimous_harmed=int((scope & ~hit & hit0).sum()))
            if seed != 'v131':
                history.setdefault(name, []).append(hit)
        if seed != 'v131':
            payer_sets.append(scope)
    target = hard & y & np.logical_and.reduce(payer_sets)
    assert target.sum() == 86
    stable = {}
    for name, seq in history.items():
        hits = np.stack(seq)
        stable[name] = dict(hard_ever=int((hits.any(0) & hard).sum()), hard_stable=int((hits.all(0) & hard).sum()),
                            target_ever=int((hits.any(0) & target).sum()), target_stable=int((hits.all(0) & target).sum()),
                            stable_harms=int((~hits.any(0) & right).sum()))
    write(REPORT/'interaction_transfer.json', dict(plans=plans, stable=stable))
    print({name: [p[name]['fixed']-p[name]['harmed'] for p in plans.values()] for name in stable}, stable)


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
