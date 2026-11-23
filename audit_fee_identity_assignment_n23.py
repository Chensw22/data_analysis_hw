"""Score new fee pairings on held-out bookings, including previously unanimous solo payers."""
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from audit_stable_rescue_placebo_n23 import family_scores, panel
from broad_geographic_transfer_n23 import BILLS, load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_tree_field_dividends_n23 import hard_cohorts
from fee_identity_assignment_n23 import LABEL, REPORT
from pending_family_coverage_n23 import PAIRS, development_votes, read, rules


def main():
    row, y, frame, groups, _ = load()
    solo_paid = (frame.BookingSize.to_numpy() == 1) & frame[BILLS].gt(0).any(axis=1).to_numpy()
    _, _, hard, right = hard_cohorts(row, y)
    _, _, _, _, prior, _, _ = panel(row, y, frame)
    prior_seen = np.logical_or.reduce(list(prior.values()))
    weight = read(PAIRS/'test_budget.json')['test']['change_150']['weight']
    scopes, history, plans = [], {}, {}
    for seed in [*map(str, SEEDS), 'v131']:
        fold, base, ref, families = family_scores(seed, row, y, frame)
        scope = solo_paid & ~ref & (np.column_stack(list(families.values())) < .5).all(1)
        old_ref, previous = development_votes(frame, groups, row, y, seed, weight)
        np.testing.assert_array_equal(old_ref, ref)
        outside = ~rules(ref, previous)[0]['all']
        with np.load(ART/f'nov23_{LABEL}/{seed}.npz') as assigned, np.load(
                ART/f'nov23_registration_bill_links/{seed}.npz') as ordinary:
            for cache in [assigned, ordinary]:
                for key, value in [('row', row), ('fold', fold), ('y', y), ('ref', ref), ('base', base)]:
                    np.testing.assert_array_equal(cache[key], value)
            direct = ordinary['direct']
            alt, anon = assigned['assigned'], assigned['anonymous']
        with np.load(ART/f'nov23_within_fee_context_rank/{seed}.npz') as rank:
            for key, value in [('row', row), ('fold', fold), ('y', y), ('ref', ref), ('base', base)]:
                np.testing.assert_array_equal(rank[key], value)
            probs = dict(anonymous=anon, assigned=alt, direct=direct, both_mean=.5*(alt+direct),
                         fees_only=rank['fees_only'], context_rank=rank['context_rank'])
        metrics = {}
        for name, p in probs.items():
            vote = p >= .5
            blend = np.where(ref == (base >= .5), .75*base+.25*p >= .5, ref)
            metrics[name] = dict(vs_reference=paired(vote == y, ref == y), vs_anonymous=paired(vote == y, (anon>=.5) == y),
                                 quarter=paired(blend == y, ref == y), outside_quarter=paired(blend[outside] == y[outside], ref[outside] == y[outside]),
                                 unanimous_count=int(scope.sum()), unanimous_fixed=int((scope & vote & y).sum()),
                                 unanimous_harmed=int((scope & vote & ~y).sum()),
                                 unanimous_auc=float(roc_auc_score(y[scope], p[scope])), all_log_loss=float(log_loss(y,p)))
            if seed != 'v131':
                history.setdefault(name, []).append(vote == y)
                history.setdefault('quarter_'+name, []).append(blend == y)
        if seed != 'v131':
            scopes.append(scope)
        plans[seed] = metrics
    shared = np.logical_and.reduce(scopes)
    target = hard & shared & y
    assert (shared.sum(), (shared & y).sum(), target.sum()) == (922, 95, 86)
    stability = {}
    for name, sequence in history.items():
        hit = np.stack(sequence)
        stability[name] = dict(target_ever=int((target & hit.any(0)).sum()), target_stable=int((target & hit.all(0)).sum()),
                               stable_hard=int((hard & hit.all(0)).sum()), new_stable_hard=int((hard & hit.all(0) & ~prior_seen).sum()),
                               stable_harms=int((right & ~hit.any(0)).sum()))
    write(REPORT/'comparison.json', dict(plans=plans, stability=stability))
    print(stability)
    print({s: v['both_mean']['quarter'] for s,v in plans.items()})


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
