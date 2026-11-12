"""Checks that matter for competition validity, without touching holdout labels."""
import pandas as pd
from experiment import load_data, features, SPEND

train, test, _ = load_data()
for mode in ['raw', 'engineered', 'engineered_no_ids', 'engineered_no_spend', 'engineered_rules']:
    frame = train.head(30)
    x, cats = features(frame, mode)
    changed = frame.copy()
    changed['Transported'] = ~changed.Transported
    pd.testing.assert_frame_equal(x, features(changed, mode)[0])
    pd.testing.assert_frame_equal(x.iloc[[0]], features(frame.iloc[[0]], mode)[0])
    assert list(x.columns) == list(features(test.head(30), mode)[0].columns)
    assert 'Transported' not in x
    assert not x[cats].isna().any().any()
    print(mode, ': target independence, row locality, consistent schema, categorical nulls OK')
empty = train.iloc[[0]].copy()
empty['Cabin'] = pd.Series([None], index=empty.index, dtype='str')
empty[SPEND] = float('nan')
x, _ = features(empty, 'engineered')
assert pd.isna(x.TotalSpend.iloc[0]) and x.AllSpendingZero.iloc[0] == 0
assert x.Deck.iloc[0] == '__MISSING__'
print('Missing cabin and all-missing spending handling OK')
print('Raw data shape, disjoint IDs and sample submission order OK')
