"""Train a correction model specifically on low-margin cases from the two largest common-error populations."""
import json

import numpy as np
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from dissent_likelihood_transfer_n27 import splits
from full_roster_booking_validation_n23 import ROOT
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART, cohort, count
from unanimous_error_reach_n27 import inputs


def evaluate():
    raw, frame, y, groups, folds, first, edits, _, scores, alternatives = inputs()
    n = len(y)
    _, _, known, fresh = cohort(raw, y, folds)
    anchor = first ^ edits
    vote = scores[:, 0] >= 0.5
    common = ((scores >= 0.5) == vote[:, None]).all(axis=1)
    paid = frame[FEES].gt(0).any(axis=1).to_numpy()
    populations = {
        'awake_paid': frame.CryoSleep.eq(False).to_numpy() & paid,
        'sleep_zero': frame.CryoSleep.eq(True).to_numpy() & ~paid,
    }
    fold_with_test = np.r_[folds, np.full(len(frame) - n, 8)]
    eligible = common & np.r_[known, np.ones(len(frame) - n, dtype=bool)]
    margin = np.min(np.abs(scores - 0.5), axis=1)
    # Use raw expert opinions. Every training label for a correction model belongs to
    # the same outer held fold as its query, so the underlying experts never saw either.
    opinions = np.column_stack([scores, *alternatives.values()])
    directed = logit(np.clip(np.where(vote[:, None], 1 - opinions, opinions), 0.005, 0.995))
    design = np.column_stack([vote, directed])
    target = vote[:n] != y
    wrong = anchor[:n] != y
    saved, report = {}, {}
    for name, population in populations.items():
        use = eligible & population
        review = np.zeros(len(frame), dtype=bool)
        for fold in range(9):
            rows = np.flatnonzero(use & (fold_with_test == fold))
            review[rows] = rankdata(margin[rows]) / len(rows) <= 0.25
        test_rows = np.flatnonzero(review & (fold_with_test == 8))
        for protocol in ('random', 'contiguous'):
            part = splits(groups, folds, protocol)
            output = {}
            for mode, scope in [('all_in_population', use), ('review_only', review)]:
                actions = np.zeros(len(frame), dtype=bool)
                test_votes = []
                for fold in range(8):
                    for held in [0, 1, 2, 3, None]:
                        train = (folds == fold) & scope[:n]
                        if held is None:
                            q = test_rows
                        else:
                            train &= part != held
                            q = np.flatnonzero(review[:n] & (folds == fold) & (part == held))
                            assert not np.isin(groups[train], groups[q]).any()
                        model = LogisticRegression(C=0.1, max_iter=300)
                        model.fit(design[:n][train], target[train])
                        flip = model.predict(design[q]) & (anchor[q] == vote[q])
                        if held is None:
                            test_votes.append(flip)
                        else:
                            actions[q] = flip
                actions[test_rows] = np.mean(test_votes, axis=0) > 0.5
                output[mode] = actions
            output['review_only_increment'] = output['review_only'] & ~output['all_in_population']
            for mode, flag in output.items():
                saved[f'{name}_{protocol}_{mode}'] = flag
                report.setdefault(name, {}).setdefault(protocol, {})[mode] = dict(
                    all=count(flag[:n], known, wrong), fresh=count(flag[:n], known & fresh, wrong),
                    test=int(flag[n:].sum()), by_fold=[count(flag[:n], known & (folds == k), wrong) for k in range(8)])
    for name in populations:
        np.testing.assert_array_equal(saved[f'{name}_random_review_only'][n:],
                                      saved[f'{name}_contiguous_review_only'][n:])
    path = ART / 'nov27_common_error_review_fit'
    path.mkdir(exist_ok=True)
    np.savez_compressed(path / 'aligned.npz', passenger=frame.PassengerId.to_numpy(str), fold=folds, y=y, **saved)
    (ROOT / 'reports/nov23_matched_learning_curve/common_error_review_fit.json').write_text(
        json.dumps(report, indent=2) + '\n')
    for name, protocols in report.items():
        for protocol, modes in protocols.items():
            print(name, protocol, {key: (v['all'], v['test']) for key, v in modes.items()})


if __name__ == '__main__':
    with threadpool_limits(1):
        evaluate()
