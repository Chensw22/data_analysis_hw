"""Track shared-error recovery when distinct field ablations agree across training sizes."""
import json

import numpy as np

from broad_geographic_transfer_n23 import BILLS
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, data
from full_roster_metric_adjudication_n23 import load_scores
from full_roster_neighborhood_disputes_n23 import DEST
from full_roster_sleep_bill_factorial_n23 import VIEWS

REPORT = ROOT / 'reports/nov23_matched_learning_curve/factorial_repeats.json'


def outcomes(mask, flip, wrong):
    repaired = int((mask & flip & wrong).sum())
    harmed = int((mask & flip & ~wrong).sum())
    return dict(rows=int(mask.sum()), baseline_wrong=int((mask & wrong).sum()),
                repaired=repaired, harmed=harmed, net=repaired-harmed)


def run():
    raw, y, groups, fold = data()
    for split in [fold, fold % 2]:
        for f in np.unique(split):
            assert not np.isin(groups[split == f], groups[split != f]).any()
    scores = np.column_stack([load_scores(raw, y, fold, OUT / f'{name}.npz', ['p'])[:, 0]
                              for name in (*FAMILIES, 'mitra96')])
    neighbors = load_scores(raw, y, fold, DEST / 'full.npz', ['without_near', 'only_near'])
    base = scores[:, -1] >= .5
    wrong = base != y
    seven = ((np.column_stack([scores, neighbors]) >= .5) == base[:, None]).all(1)
    assert (seven & wrong).sum() == 819
    with np.load(ROOT / 'artifacts/nov23_sleep_bill_factorial/oof.npz') as cache:
        for key, expected in [('passenger', raw.PassengerId), ('fold', fold), ('y', y)]:
            np.testing.assert_array_equal(cache[key], expected)
        banks = []
        for key in ['full', 'half']:
            p = cache[key]
            assert p.shape == (len(y), len(VIEWS)) and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
            banks.append((p >= .5) != base[:, None])
    full, half = banks
    patterns = full.astype(int) @ (1 << np.arange(len(VIEWS)))
    overlaps = {}
    for code in range(1 << len(VIEWS)):
        match = seven & (patterns == code)
        views = [name for i, name in enumerate(VIEWS) if code & (1 << i)]
        overlaps['+'.join(views) or 'none'] = dict(original_wrong=int((match & wrong).sum()),
                                                   original_right=int((match & ~wrong).sum()))

    # The same two field sets must oppose the full Mitra decision at both sizes.
    # The second comparison also admits a different pair of field sets at each size.
    full_majority = full.sum(1) >= 2
    repeated = (full & half).sum(1) >= 2
    rules = {name: full[:, j] for j, name in enumerate(VIEWS)}
    rules.update(full_majority=full_majority, repeated_same_views=repeated,
                 repeated_any_majorities=full_majority & (half.sum(1) >= 2))
    paid = raw[BILLS].gt(0).any(axis=1).to_numpy()
    sleep = raw.CryoSleep.astype('string').fillna('missing').to_numpy()
    all_rows = np.ones(len(y), bool)
    masks = dict(all=all_rows, seven_agree=seven, seven_divided=~seven)
    for pname, payment in [('positive', paid), ('no_known_positive', ~paid)]:
        for state in ('True', 'False', 'missing'):
            for direction in [False, True]:
                masks[f'{pname}_sleep_{state}_vote_{direction}'] = payment & (sleep == state) & (base == direction)
    subgroups = [v for k, v in masks.items() if k not in ('all', 'seven_agree', 'seven_divided')]
    np.testing.assert_array_equal(np.sum(subgroups, axis=0), 1)
    results = {}
    for name, flip in rules.items():
        results[name] = dict(by_group={k: outcomes(mask, flip, wrong) for k, mask in masks.items()},
                             seven_by_group={k: outcomes(mask & seven, flip, wrong) for k, mask in masks.items()
                                             if k not in ('seven_agree', 'seven_divided', 'all')},
                             net_by_fold=[outcomes(fold == f, flip, wrong)['net'] for f in np.unique(fold)])
    return dict(rows=len(y), baseline_correct=int((~wrong).sum()), seven_wrong=int((seven & wrong).sum()),
                seven_right=int((seven & ~wrong).sum()), full_only_overlap=overlaps, rules=results,
                design='Half-panel training orders are nested in full-fold training orders before fitting; '
                       'all six scores exclude the query order. Both votes compare to full-fold Mitra.')


if __name__ == '__main__':
    result = run()
    REPORT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name, rule in result['rules'].items():
        print(name, rule['by_group']['all'], rule['by_group']['seven_agree'])
