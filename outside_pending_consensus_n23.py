"""Evaluate continuous expert disagreement where all eleven pending decisions retain the incumbent."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from audit_stable_rescue_placebo_n23 import family_scores
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import SEEDS, write
from broad_tree_field_dividends_n23 import hard_cohorts
from pending_family_coverage_n23 import PAIRS, REPORT, development_votes, read, rules


def designs(base, families):
    anchor = logit(base.clip(.002, .998))[:, None]
    deviations = []
    for estimates in families.values():
        logits = logit(estimates.clip(.002, .998))
        deviations.append(np.median(logits, axis=1) - anchor[:, 0])
        if logits.shape[1] > 1:
            # Retain the range as well as the median, allowing a minority expert to matter.
            deviations.append(logits.max(1) - logits.min(1))
    return {'base_only': anchor, 'family_evidence': np.column_stack([anchor, *deviations])}


def run():
    row, y, frame, groups, _ = load()
    _, _, hard, right = hard_cohorts(row, y)
    weight = read(PAIRS/'test_budget.json')['test']['change_150']['weight']
    plans, history = {}, {}
    for seed in [*map(str, SEEDS), 'v131']:
        ref, votes = development_votes(frame, groups, row, y, seed, weight)
        masks, _ = rules(ref, votes)
        outside = ~masks['all']
        fold, base, cached_ref, families = family_scores(seed, row, y, frame)
        np.testing.assert_array_equal(ref, cached_ref)
        fit_scope = outside & (ref == (base >= .5))
        expert = np.column_stack(list(families.values())) >= .5
        any_correct = (expert == y[:, None]).any(1)
        metrics = {}
        for name, features in designs(base, families).items():
            prob = base.copy()
            checked = np.zeros(len(y), bool)
            for f in np.unique(fold):
                ids = np.flatnonzero(fit_scope & (fold == f))
                for a, b in GroupKFold(3, shuffle=True, random_state=518).split(ids, groups=groups[ids]):
                    train, query = ids[a], ids[b]
                    assert not np.intersect1d(groups[train], groups[query]).size
                    fit = make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=300))
                    fit.fit(features[train], y[train])
                    prob[query] = fit.predict_proba(features[query])[:, 1]
                    checked[query] = True
            np.testing.assert_array_equal(checked, fit_scope)
            for method, score in [(name, prob), (name+'_quarter', .75*base + .25*prob)]:
                vote = np.where(fit_scope, score >= .5, ref)
                assert np.all(vote[~outside] == ref[~outside])
                hit = vote == y
                metrics[method] = paired(hit, ref == y)
                if seed != 'v131':
                    history.setdefault(method, []).append(hit)
        plans[seed] = dict(outside=int(outside.sum()), errors=int((outside & (ref != y)).sum()),
                           accessible_errors=int((fit_scope & (ref != y) & any_correct).sum()),
                           hard_accessible=int((fit_scope & hard & any_correct).sum()), rules=metrics)
        print(seed, plans[seed], flush=True)
    stable = {}
    for name, outcomes in history.items():
        hit = np.stack(outcomes)
        stable[name] = dict(hard_ever=int((hard & hit.any(0)).sum()), hard_stable=int((hard & hit.all(0)).sum()),
                            hard_by_correct_replays=np.bincount(hit[:, hard].sum(0), minlength=4).tolist(),
                            stable_harms=int((right & ~hit.any(0)).sum()))
    write(REPORT/'outside.json', dict(plans=plans, stable=stable))
    print(stable)


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
