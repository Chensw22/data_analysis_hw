"""Count similar training orders for each passenger held out of five full-roster classifiers."""
import json

import numpy as np
import pandas as pd

from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, data

REPORT = ROOT / 'reports/nov23_matched_learning_curve/discrete_order_support.json'
BILLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


def matching_cells(raw):
    bands = pd.cut(raw.Age, [-np.inf, 13, 25, 45, np.inf], right=False,
                   labels=['0to12', '13to24', '25to44', '45plus'])
    keys = pd.DataFrame(dict(
        sleep=raw.CryoSleep.astype('string').fillna('?'),
        home=raw.HomePlanet.fillna('?'), deck=raw.Cabin.str.split('/').str[0].fillna('?'),
        positive=raw[BILLS].gt(0).any(axis=1), age=bands.astype('string').fillna('?')))
    return keys.groupby(list(keys)).ngroup().to_numpy()


def training_matches(cells, groups, y, fold):
    orders = np.zeros(len(y), int)
    rows = np.zeros(len(y), int)
    rate = np.full(len(y), np.nan)
    for f in np.unique(fold):
        train, query = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        assert not np.isin(groups[query], groups[train]).any()
        table = pd.DataFrame(dict(cell=cells[train], order=groups[train], outcome=y[train]))
        summary = table.groupby('cell').agg(orders=('order', 'nunique'), rows=('outcome', 'size'),
                                            rate=('outcome', 'mean')).reindex(cells[query])
        orders[query] = summary.orders.fillna(0).to_numpy(int)
        rows[query] = summary.rows.fillna(0).to_numpy(int)
        rate[query] = summary.rate.to_numpy()
    return orders, rows, rate


def run():
    raw, y, groups, fold = data()
    predictions = []
    families = [*FAMILIES, 'mitra96']
    for name in families:
        with np.load(OUT / f'{name}.npz') as cache:
            for key, expected in [('passenger', raw.PassengerId), ('fold', fold), ('y', y)]:
                np.testing.assert_array_equal(cache[key], expected)
            predictions.append(cache['p'])
    scores = np.column_stack(predictions)
    votes = scores >= .5
    unanimous = (votes == votes[:, :1]).all(1)
    direction = votes[:, 0]
    error = direction != y
    cells = matching_cells(raw)
    orders, rows, rate = training_matches(cells, groups, y, fold)

    # Standardize descriptively against errors in all unanimous cases of the
    # same predicted direction and minimum-confidence bin. These rates are not
    # used to select neighbors or to predict the validation outcomes.
    gap = np.abs(scores - .5).min(1)
    stratum = 10 * direction.astype(int) + np.minimum((gap * 20).astype(int), 9)
    n = np.bincount(stratum[unanimous], minlength=20)
    errors = np.bincount(stratum[unanimous], weights=error[unanimous], minlength=20)
    reference_rates = np.divide(errors, n, out=np.zeros(20), where=n > 0)

    levels = {}
    for key, selected in [('0_to_9', orders < 10), ('10_to_49', (orders >= 10) & (orders < 50)),
                          ('50_or_more', orders >= 50)]:
        u = selected & unanimous
        decisive = u & np.isfinite(rate) & (rate != .5)
        same = decisive & ((rate > .5) == direction)
        opposite = decisive & ~same
        by_direction = {}
        for label, positive in [('not_transported', False), ('transported', True)]:
            mask = u & (direction == positive)
            by_direction[label] = dict(unanimous=int(mask.sum()), wrong=int((mask & error).sum()))
        levels[key] = dict(
            passengers=int(selected.sum()), unanimous=int(u.sum()), common_wrong=int((u & error).sum()),
            error_rate_among_unanimous=float(error[u].mean()), by_direction=by_direction,
            median_matching_orders=float(np.median(orders[selected])),
            median_matching_passengers=float(np.median(rows[selected])),
            majority_by_training_passenger=dict(non_tied=int(decisive.sum()), same=int(same.sum()),
                same_wrong=int((same & error).sum()), opposite=int(opposite.sum()),
                opposite_wrong=int((opposite & error).sum())),
            expected_errors_at_same_direction_and_confidence=float(reference_rates[stratum[u]].sum()))
    return dict(families=families, rows=len(y), occupied_cells=int(np.unique(cells).size),
                no_matching_order=int((orders == 0).sum()), unanimous=int(unanimous.sum()),
                common_wrong=int((unanimous & error).sum()),
                median_orders_common_wrong=float(np.median(orders[unanimous & error])),
                median_orders_common_correct=float(np.median(orders[unanimous & ~error])), levels=levels)


if __name__ == '__main__':
    result = run()
    REPORT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for key, values in result['levels'].items():
        print(key, values['passengers'], values['unanimous'], values['common_wrong'], values['by_direction'])
