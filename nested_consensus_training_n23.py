"""Train error selectors on out-of-sample predictions from other orders in each outer fold."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from audit_stable_rescue_placebo_n23 import panel
from broad_geographic_transfer_n23 import BILLS, load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, write
from broad_tree_field_dividends_n23 import hard_cohorts
from exaone_validation_data import compact_features
from expanded_score_frontier_n23 import source_scores
from expanded_training_roster_n23 import eight_fold, fitting as raw_tree
from matched_consensus_errors_n23 import calibrated, fitting, matched, predict, roster_fields, tree_offset
from spend_simplex_likelihood_n23 import design, learn as budget_tree


def nested_scores(features, budget, y, groups, train, query):
    scores = np.full((len(y), 2), np.nan)
    # All calibration predictions are made by models blind to both the
    # eventual query order and the order receiving the calibration prediction.
    for a, b in GroupKFold(3, shuffle=True, random_state=542).split(train, groups=groups[train]):
        t, q = train[a], train[b]
        assert not np.isin(groups[t], groups[np.r_[q, query]]).any()
        scores[q, 0] = raw_tree(features, y[t], t, q)
        scores[q, 1] = budget_tree(budget, y, t, q)
    scores[query, 0] = raw_tree(features, y[train], train, query)
    scores[query, 1] = budget_tree(budget, y, train, query)
    return scores


def run():
    row, y, frame, groups, _ = load()
    _, _, hard, right = hard_cohorts(row, y)
    fold, _, ref = eight_fold(row, y)
    target = np.column_stack([p for _, p in source_scores(row, y, fold)])
    target_vote = target[:, 0] >= .5
    eligible = ((target >= .5) == target_vote[:, None]).all(1) & (
        target_vote != frame[BILLS].gt(0).any(axis=1).to_numpy())
    assert eligible.sum() == 5258 and np.sum(eligible & (target_vote != y)) == 732
    raw = pd.read_csv(ROOT / 'data/raw/train.csv', usecols=lambda c: c != 'Name')
    truth = raw.pop('Transported').to_numpy(bool)
    booking = raw.PassengerId.str.split('_').str[0].astype(int).to_numpy()
    features = compact_features(raw)
    budget = design(features)[0]
    paid = raw[BILLS].gt(0).any(axis=1).to_numpy()
    np.testing.assert_array_equal(truth[row], y)
    np.testing.assert_array_equal(booking[row], groups)
    personal, peers = roster_fields(np.arange(len(truth)), booking)
    fields = np.column_stack([personal, peers])
    names = ['prior', 'matched_personal', 'matched_peers', 'tree_peers']
    names.extend(f'small_{i}_{kind}' for i in range(3) for kind in ('matched', 'tree'))
    changes = {k: np.zeros(len(y), bool) for k in names}
    sizes = []
    path = ART / 'nov23_nested_consensus_training'
    path.mkdir(exist_ok=True)
    for f in np.unique(fold):
        ix = np.flatnonzero(fold == f)
        q = row[ix]
        t = np.flatnonzero(~np.isin(booking, booking[q]))
        cache = path / f'fold_{f}.npz'
        if cache.exists():
            with np.load(cache) as z:
                for k, v in [('train', t), ('query', q), ('training_labels', truth[t])]:
                    np.testing.assert_array_equal(z[k], v)
                p = z['p']
        else:
            p = nested_scores(features, budget, truth, booking, t, q)
            np.savez_compressed(cache, p=p, train=t, query=q, training_labels=truth[t])
        assert np.isfinite(p[np.r_[t, q]]).all()
        shadow_vote = p[:, 0] >= .5
        same = np.all((p >= .5) == shadow_vote[:, None], axis=1)
        risk = np.where(shadow_vote, 1-p.mean(1), p.mean(1))
        risk[q] = np.where(target_vote[ix], 1-target[ix].mean(1), target[ix].mean(1))
        offset = logit(risk.clip(.001, .999))
        error = shadow_vote != truth
        counts = {}
        subsamples = [t[b] for _, b in GroupKFold(12, shuffle=True, random_state=434).split(t, groups=booking[t])][:3]
        for direction in (False, True):
            train = t[same[t] & (shadow_vote[t] == direction) & (paid[t] != direction)]
            local = ix[eligible[ix] & (target_vote[ix] == direction)]
            query = row[local]
            pairs = matched(train, error, risk, booking)
            counts[str(direction)] = [len(train), int(error[train].sum())]
            methods = dict(prior=calibrated(offset, error, train, query),
                           matched_personal=fitting(personal, offset, error, train, query, pairs),
                           matched_peers=fitting(fields, offset, error, train, query, pairs),
                           tree_peers=tree_offset(fields, offset, error, train, query))
            small_sizes = []
            for i, sample in enumerate(subsamples):
                subset = np.intersect1d(train, sample)
                small_sizes.append(len(subset))
                small_pairs = matched(subset, error, risk, booking)
                methods[f'small_{i}_matched'] = fitting(fields, offset, error, subset, query, small_pairs)
                methods[f'small_{i}_tree'] = tree_offset(fields, offset, error, subset, query)
            counts[f'small_{direction}'] = small_sizes
            for k, estimates in methods.items():
                changes[k][local] = estimates >= .5
        agreement = int(np.sum(eligible[ix] & same[q] & (shadow_vote[q] == target_vote[ix])))
        sizes.append(dict(fold=int(f), training_rows=len(t), rows_and_errors=counts,
                          eligible=int(eligible[ix].sum()), both_surrogate_models_agree_with_target=agreement))
        print(f, counts, flush=True)
    changes['both_peers'] = changes['matched_peers'] & changes['tree_peers']
    original = np.where(eligible, target_vote, ref)
    records = {}
    for k, flip in changes.items():
        assert not (flip & ~eligible).any()
        vote = original ^ flip
        fixed = flip & (original != y)
        harmed = flip & (original == y)
        records[k] = dict(vs_consensus=paired(vote == y, original == y), vs_reference=paired(vote == y, ref == y),
                          hard_fixed=int((fixed & hard).sum()), previously_correct_harmed=int((harmed & right).sum()),
                          net_by_fold=[int(fixed[fold==f].sum())-int(harmed[fold==f].sum()) for f in np.unique(fold)],
                          by_paid={str(val): [int((fixed & (paid[row]==val)).sum()), int((harmed & (paid[row]==val)).sum())]
                                   for val in [False, True]})
    risk = np.where(target_vote, 1-target.mean(1), target.mean(1))
    _, _, _, _, previous, _, _ = panel(row, y, frame)
    new = hard & ~np.logical_or.reduce(list(previous.values()))
    before = []
    for seed in [981, 410, 213]:
        probs, _ = predict({'peers': fields[row]}, seed, fold, groups, target_vote != y, risk,
                           logit(risk.clip(.001, .999)), eligible, target_vote)
        before.append(eligible & (probs['peers'] >= .5))
    repeated = np.logical_and.reduce(before)
    assert (new & repeated).sum() == 1 and (hard & repeated).sum() == 5
    reproduction = {}
    for k, cohort in {'old_twelve': before[0] & hard, 'stable_five': repeated & hard,
                      'previously_uncovered_one': repeated & new}.items():
        reproduction[k] = dict(size=int(cohort.sum()), matched=int((cohort & changes['matched_peers']).sum()),
                               tree=int((cohort & changes['tree_peers']).sum()))
    write(ROOT / 'reports/nov23_matched_learning_curve/nested_error_training.json',
          dict(splits=sizes, methods=records, reproduction=reproduction))
    print(records)


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
