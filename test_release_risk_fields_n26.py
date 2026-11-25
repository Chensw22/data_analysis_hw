"""Build the 24 validation risk features on the competition roster and replay every final stage."""
import json

import numpy as np
import pandas as pd

from evaluate_nested_release_n26 import outcome_scores, stages
from exaone_validation_data import compact_features
from export_budget_quarter_n23 import source
from export_consensus_review_n26 import ART, dig
from export_expert_vote_panel_n26 import NEW, OUT as PANEL, REPORT as PANEL_REPORT, frozen
from frozen_queue_shared_errors_n26 import RELEASES
from full_roster_booking_validation_n23 import ROOT
from held_expert_portfolios_n26 import SIX
from price_implied_deck_n26 import FEES
from release_confidence_review_n26 import summarize
from six_way_against_scored_nine_n26 import test_decisions as release_decisions
from target_mix_expert_routing_n26 import bank, batches

CACHE = ART/'nov26_test_release_risk_fields/fields.npz'
REPORT = ROOT/'reports/nov23_matched_learning_curve/test_release_risk_fields.json'
NAMES = ['parent', 'all_99_agree', 'native_signed_error_log_odds',
         'error_log_odds_q05', 'error_log_odds_q25', 'error_log_odds_q50', 'error_log_odds_q75',
         'error_log_odds_q95', 'positive_error_log_odds_fraction', 'anchor_to_review', 'review_to_fare',
         'fare_to_six', 'review_error_risk', 'review_outcome_risk', 'single_passenger', 'booking_at_least_four',
         'deck_g', 'side_s', 'planet_earth', 'age_below_thirteen', 'age_missing', 'cryo_true', 'missing_any_fee',
         'log1p_total_spend_div8']


def roster_fields(raw):
    return np.column_stack([raw.booking_size.eq(1), raw.booking_size.ge(4), raw.Deck.eq('G'),
                            raw.Side.eq('S'), raw.HomePlanet.eq('Earth'), raw.Age.lt(13), raw.Age.isna(),
                            raw.CryoSleep.eq('True'), raw[FEES].isna().any(axis=1), np.log1p(raw[FEES].sum(axis=1))/8])


def run():
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=lambda x:x != 'Name')
    ids = test.PassengerId.to_numpy(str)
    assert len(np.unique(ids)) == len(ids) == 4277
    frame = compact_features(test).assign(booking_size=test.groupby(test.PassengerId.str.split('_').str[0]).PassengerId.transform('size'))
    hard = {k:release_decisions(n, ids)[0] for k, n in RELEASES.items()}
    before = hard['nine']
    changes = {k: v != before for k, v in hard.items()}
    np.testing.assert_array_equal(changes['fields_awake'], changes['fields'] | changes['awake'])
    np.testing.assert_array_equal(changes['all_three'], changes['fields_awake'] | changes['band'])
    assert not (changes['filtered'] & ~changes['awake']).any()
    digest = dig(PANEL/'scores.npz')
    assert digest == json.loads(PANEL_REPORT.read_text())['sha256']
    with np.load(PANEL/'scores.npz') as z:
        np.testing.assert_array_equal(ids, z['passenger'])
        np.testing.assert_array_equal(hard['six'], z['prediction'])
        scores, names = z['scores'], z['names']
    raw_old = scores[:, :-len(NEW)]
    np.testing.assert_array_equal(names[-len(NEW):], NEW)
    native = source('proposal_online_first_counts15', test)[0]
    anchor = hard['all_three']
    block = summarize(raw_old, native, anchor)
    with np.load(ART/RELEASES['review']/'test_inputs.npz') as z:
        np.testing.assert_array_equal(ids, z['passenger'])
        np.testing.assert_array_equal(names[:-len(NEW)], z['names'])
        for key, a in block.items():
            np.testing.assert_array_equal(a, z[key])
    v = dict(**block, votes=raw_old>=.5)
    p = dict(zip(NEW, scores[:,-len(NEW):].T, strict=True))
    s = outcome_scores(v,p)
    with frozen(RELEASES['review'], 'review_scores.npz') as z:
        np.testing.assert_array_equal(ids, z['passenger'])
        np.testing.assert_array_equal(z['parent']>=.5, anchor)
        risk = z['scores']
    with frozen(RELEASES['fare'], 'scores.npz') as z:
        np.testing.assert_array_equal(ids, z['passenger'])
        np.testing.assert_array_equal(z['parent']>=.5, hard['review'])
        np.testing.assert_array_equal(z['raw'],p['neural_deck_raw'])
        np.testing.assert_array_equal(z['fine'],p['neural_deck_fine'])
        enabled = z['eligible']
    with frozen(RELEASES['six'], 'predictions.npz') as z:
        np.testing.assert_array_equal(ids, z['passenger'])
        np.testing.assert_array_equal(z['parent']>=.5, hard['fare'])
        np.testing.assert_array_equal(z['challengers'], np.column_stack([p[k] for k in SIX]))
        np.testing.assert_array_equal(z['family_votes'], v['votes'][:,:18])
        np.testing.assert_array_equal(z['eligible'], enabled)
    idx = np.arange(len(ids))
    for n, predictions in zip(('all_three','review','fare','six'), stages(v,p,idx,enabled,risk), strict=True):
        np.testing.assert_array_equal(predictions, hard[n])
    parent, summary, _, _ = bank(v, p, s, idx, risk, enabled, frame[FEES].gt(0).any(axis=1).to_numpy())
    np.testing.assert_array_equal(parent, hard['six'])
    common = (np.column_stack(list(hard.values())) == parent[:,None]).all(1)
    fields = np.column_stack([summary,roster_fields(frame)])
    assert fields.shape == (len(ids),len(NAMES)) and np.isfinite(fields).all()
    # Verify the independently constructed raw columns on every full OOF query from the original entry point.
    train = pd.read_csv(ROOT/'data/raw/train.csv', usecols=test.columns.tolist())
    tr = compact_features(train).assign(booking_size=train.groupby(train.PassengerId.str.split('_').str[0]).PassengerId.transform('size'))
    raw = roster_fields(tr)
    validated = []
    for b in batches(True):
        if b.tag == 'full':
            assert b.query_fields.shape[1] == len(NAMES)
            np.testing.assert_array_equal(raw[b.ids],b.query_fields[:,14:])
            validated.extend(b.ids)
    assert len(set(validated)) == len(validated) == 8532
    CACHE.parent.mkdir(exist_ok=True)
    tmp = CACHE.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, passenger=ids, fields=fields, parent=parent, common=common, names=np.array(NAMES))
    tmp.replace(CACHE)
    r = dict(cache=str(CACHE.relative_to(ROOT)), sha256=dig(CACHE), panel_sha256=digest, rows=len(ids),
             columns=NAMES, common=int(common.sum()), experts_unanimous=int(fields[:,1].sum()),
             ten_frozen_release_csvs=len(hard), replayed_decisions=['all_three','review','fare','six'],
             full_validation_rows_checked=len(validated), native_and_89_inputs_replayed=True)
    REPORT.write_text(json.dumps(r,indent=2)+'\n')
    print(r)


if __name__ == '__main__':
    run()
