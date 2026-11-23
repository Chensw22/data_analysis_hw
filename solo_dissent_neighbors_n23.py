"""Estimate the local odds of an erroneous solo prediction using other comparable solo bookings."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from audit_stable_rescue_placebo_n23 import family_scores, panel
from broad_geographic_transfer_n23 import BILLS, load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ROOT, SEEDS, write
from broad_tree_field_dividends_n23 import hard_cohorts

REPORT = ROOT/'reports/nov23_solo_dissent_neighbors'


def match(risk, coords, errors):
    # Equal fractional inclusion of ties at the Kth distance keeps exact duplicates exchangeable.
    distances = cdist(coords, coords)
    np.fill_diagonal(distances, np.inf)
    k = min(32, len(risk)-1)
    assert k > 0
    radius = np.partition(distances, k-1, axis=1)[:, k-1, None]
    interior, border = distances < radius, distances == radius
    weights = interior + border * ((k-interior.sum(1))/border.sum(1))[:, None]
    np.testing.assert_allclose(weights.sum(1), k)
    assert not weights.diagonal().any()
    # Estimate a local shift in log odds, retaining each neighbor's original risk.
    # A zero-centered normal prior has precision two on the common log-odds shift.
    offsets = logit(risk.clip(.001, .999))
    shift = np.zeros(len(risk))
    for _ in range(12):
        fitted = expit(offsets[None, :] + shift[:, None])
        gradient = (weights * (errors[None, :] - fitted)).sum(1) - 2*shift
        curvature = (weights * fitted * (1-fitted)).sum(1) + 2
        shift += (gradient / curvature).clip(-1., 1.)
    assert np.max(np.abs(gradient)) < 1e-6
    return expit(offsets + shift)


def inputs(frame, ref, base, families):
    risk = np.where(ref, 1-base, base)
    fractions = np.column_stack([((p >= .5) != ref[:, None]).mean(1) for p in families.values()])
    dissent = fractions.max(1) > 0
    # Distances for risk and disagreement are scaled on fixed, interpretable ranges.
    confidence = 2*risk[:, None]
    votes = np.column_stack([confidence, fractions.mean(1), (fractions > .5).mean(1)])
    bills = frame[BILLS]
    positive = bills.gt(0).any(axis=1).to_numpy()
    sleep = pd.get_dummies(frame.CryoSleep, dtype=float).to_numpy()/np.sqrt(2)
    log_fees = np.log1p(bills.fillna(0).to_numpy()).clip(0, 9)/9.
    context = np.column_stack([votes, sleep, positive, bills.isna().mean(axis=1), log_fees/np.sqrt(len(BILLS))])
    return risk, dissent, {'confidence': confidence, 'dissent': votes, 'passenger': context}, positive


def main():
    row, y, frame, groups, _ = load()
    _, _, hard, right = hard_cohorts(row, y)
    solo = frame.BookingSize.to_numpy() == 1
    assert len(np.unique(groups[solo])) == solo.sum() and int((solo & hard).sum()) == 226
    plans, history, candidate_masks = {}, {}, []
    for seed in [*map(str, SEEDS), 'v131']:
        fold, base, ref, families = family_scores(seed, row, y, frame)
        risk, dissent, views, paid = inputs(frame, ref, base, families)
        error = ref != y
        eligible = solo & dissent & (ref == (base >= .5))
        if seed != 'v131':
            candidate_masks.append(eligible)
        scores = {'unchanged': risk.copy()}
        for name, coords in views.items():
            prob = risk.copy()
            visited = np.zeros(len(y), bool)
            for f in np.unique(fold):
                for direction in (False, True):
                    ids = np.flatnonzero(eligible & (fold == f) & (ref == direction))
                    prob[ids] = match(risk[ids], coords[ids], error[ids])
                    visited[ids] = True
            np.testing.assert_array_equal(visited, eligible)
            scores[name] = prob
        metrics = {}
        for name, prob in scores.items():
            flip = eligible & (prob > .5)
            hit = (~error) ^ flip
            metrics[name] = dict(**paired(hit, ~error), auc=float(roc_auc_score(error[eligible], prob[eligible])),
                                 brier=float(brier_score_loss(error[eligible], prob[eligible])),
                                 hard_fixes=int((hit & solo & hard).sum()))
            if seed != 'v131':
                history.setdefault(name, []).append(hit)
        strata = {}
        for key, mask in {'all': eligible, 'sleep_recorded': eligible & frame.CryoSleep.eq('True').to_numpy(),
                          'positive_fee': eligible & paid, 'high_confidence': eligible & (risk <= .2),
                          'below_65_confidence': eligible & (risk > .35)}.items():
            strata[key] = dict(n=int(mask.sum()), wrong=int((mask & error).sum()))
        plans[seed] = dict(solo_errors=int((solo & error).sum()), possible_hard=int((hard & eligible & error).sum()),
                           strata=strata, metrics=metrics)
        print(seed, metrics['passenger'], strata, flush=True)
    _, _, _, _, earlier, _, _ = panel(row, y, frame)
    previous = np.logical_or.reduce(list(earlier.values()))
    assert (previous & hard).sum() == 87
    stable = {}
    for name, sequence in history.items():
        hits = np.stack(sequence)
        stable[name] = dict(hard_by_success_count=np.bincount(hits[:, solo & hard].sum(0), minlength=4).tolist(),
                            new_stable=int((solo & hard & ~previous & hits.all(0)).sum()),
                            stable_harms=int((solo & right & ~hits.any(0)).sum()))
    write(REPORT/'neighbors.json', dict(solo=int(solo.sum()), passengers=len(y), hard_solo=int((solo & hard).sum()),
                                       hard_by_eligible_replays=np.bincount(np.stack(candidate_masks)[:, solo & hard].sum(0), minlength=4).tolist(),
                                       plans=plans, stable=stable))
    print(stable)


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
