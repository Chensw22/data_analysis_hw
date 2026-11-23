"""Compare vote coverage with a sharp monotone-score ceiling on the matched eight folds."""
import numpy as np
from scipy.special import logit
from sklearn.model_selection import GroupKFold

from broad_geographic_transfer_n23 import BILLS, load
from broad_ordered_ensemble_n23 import apply_rule, monotone_rule
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, write
from broad_tree_field_dividends_n23 import hard_cohorts
from expanded_training_roster_n23 import eight_fold
from learning_size_gate_n23 import fit_offset


def segmented(scores, outcome, groups, fold, regime, ref, base, hard):
    oracle, segmented_vote, pooled_logistic, varying_logistic = [np.zeros(len(outcome), bool) for _ in range(4)]
    flags = np.eye(4)[regime]
    logits = logit(scores[:, 1:].clip(.001, .999))
    raw = np.column_stack([logits, flags[:, 1:], (logits[:, :, None]*flags[:, None, 1:]).reshape(len(logits), -1)])
    offset = logit(base.clip(.001, .999))
    for cell in np.unique(regime):
        selected = regime == cell
        good, bad, errors = monotone_rule(scores[selected], outcome[selected])
        oracle[selected] = apply_rule(good, bad, scores[selected])
        assert np.sum(oracle[selected] != outcome[selected]) == errors
    for f in np.unique(fold):
        ids = np.flatnonzero(fold == f)
        for a, b in GroupKFold(3, shuffle=True, random_state=981).split(ids, groups=groups[ids]):
            t, q = ids[a], ids[b]
            assert not np.intersect1d(groups[t], groups[q]).size
            for cell in np.unique(regime[q]):
                train = t[regime[t] == cell]
                query = q[regime[q] == cell]
                assert len(np.unique(outcome[train])) == 2
                good, bad, _ = monotone_rule(scores[train], outcome[train])
                segmented_vote[query] = apply_rule(good, bad, scores[query])
            pooled_logistic[q] = fit_offset(logits, offset, outcome, t, q) >= .5
            varying_logistic[q] = fit_offset(raw, offset, outcome, t, q) >= .5
    return dict(oracle=int(np.sum(oracle == outcome)), scores={name: dict(**paired(vote == outcome, ref == outcome),
                hard_correct=int(np.sum(hard & (vote == outcome)))) for name, vote in
                [('stratified_monotone', segmented_vote), ('pooled_logistic', pooled_logistic),
                 ('stratified_logistic', varying_logistic)]})


def run():
    row, y, frame, groups, _ = load()
    _, _, hard, right = hard_cohorts(row, y)
    fold, base, ref = eight_fold(row, y)
    sources = [('TabICL', 'nov23_expanded_training_roster/eight_tabicl', 'expanded'),
               ('CatBoost', 'nov23_expanded_training_roster/eight', 'expanded'),
               ('v44', 'nov23_matched_learning_curve/v44', 'large'),
               ('budget', 'nov23_budget_quarter_replay/matched', 'alt')]
    fields, plans = [], {}
    previous_unanimous_error = np.zeros(len(y), bool)
    for name, path, key in sources:
        with np.load(ART / (path + '.npz')) as z:
            for k, v in [('row', row), ('y', y), ('fold', fold)]:
                np.testing.assert_array_equal(z[k], v)
            fields.append(z[key][1] if name == 'budget' else z[key])
        if len(fields) < 2:
            continue
        x = np.column_stack(fields)
        hits = (x >= .5) == y[:, None]
        unanimous_error = ~hits.any(1)
        # The mean column only specifies the fallback on incomparable scores.
        # Adding it cannot change any componentwise ordering of the other columns.
        scores = np.column_stack([x.mean(1), x])
        good, bad, errors = monotone_rule(scores, y)
        oracle = apply_rule(good, bad, scores)
        assert (oracle != y).sum() == errors
        out = np.empty(len(y), bool)
        seen = np.zeros(len(y), int)
        for f in np.unique(fold):
            ids = np.flatnonzero(fold == f)
            for a, b in GroupKFold(3, shuffle=True, random_state=981).split(ids, groups=groups[ids]):
                t, q = ids[a], ids[b]
                assert not np.intersect1d(groups[t], groups[q]).size
                good, bad, _ = monotone_rule(scores[t], y[t])
                out[q] = apply_rule(good, bad, scores[q])
                seen[q] += 1
        assert (seen == 1).all()
        new = previous_unanimous_error & ~unanimous_error
        left = frame.loc[unanimous_error]
        plans[name] = dict(models=[s[0] for s in sources[:len(fields)]], all_wrong=int(unanimous_error.sum()),
                           all_wrong_transported=int(y[unanimous_error].sum()),
                           all_wrong_known_positive_bill=int(left[BILLS].gt(0).any(axis=1).sum()),
                           all_wrong_solo=int(left.BookingSize.eq(1).sum()),
                           new_reachable=int(new.sum()), new_reachable_hard=int((new & hard).sum()),
                           hard_all_wrong=int((unanimous_error & hard).sum()), unconstrained_vote_oracle=int(hits.any(1).sum()),
                           monotone_oracle=int(len(y)-errors), mean=paired((scores[:,0] >= .5) == y, ref == y),
                           crossfit=paired(out == y, ref == y), crossfit_hard=int((hard & (out == y)).sum()),
                           crossfit_historical_harms=int((right & (out != y)).sum()))
        previous_unanimous_error = unanimous_error
        if name == 'budget':
            paid = frame[BILLS].gt(0).any(axis=1).to_numpy()
            regime = 2*paid.astype(int) + frame.BookingSize.eq(1).to_numpy(int)
            plans[name]['segmented'] = segmented(scores, y, groups, fold, regime, ref, base, hard)
            agreed = unanimous_error | hits.all(1)
            breakdown = []
            for has_bill in [False, True]:
                for prediction in [False, True]:
                    subset = agreed & (paid == has_bill) & ((x[:,0] >= .5) == prediction)
                    breakdown.append(dict(known_positive_bill=has_bill, predicted_transported=prediction,
                                          passengers=int(subset.sum()), wrong=int((subset & unanimous_error).sum())))
            assert sum(r['wrong'] for r in breakdown) == int(unanimous_error.sum())
            plans[name]['unanimous_breakdown'] = breakdown
        print(name, plans[name], flush=True)
    write(ROOT / 'reports/nov23_matched_learning_curve/expanded_frontier.json',
          dict(rows=len(y), reference_correct=int((ref == y).sum()), plans=plans))


if __name__ == '__main__':
    run()
