"""Score true-order and shuffled-order Nori on the same held passengers."""
import json
import numpy as np
import pandas as pd

from arbitration_against_scored_nine_n26 import champions
from assess_nori_fare_party_n26 import counts, metrics
from exhaustive_family_quorums_n26 import inputs
from held_expert_portfolios_n26 import panel_predictions, sources
from nori_adjacent_manifest_n26 import ART, FEES, REPORT, ROOT, SHA, WIDTH, PERMUTATION_SEED, data, neighbor_features, packet_dir, roster
from nori_fare_party_n26 import ART as FARE
from paired_fare_foundations_n26 import policies


def paired_accuracy_interval(first, second, truth, groups, mask):
    # Cluster-robust SE of the passenger-weighted paired accuracy difference.
    delta = (first[mask] == truth[mask]).astype(float) - (second[mask] == truth[mask])
    orders, partition = np.unique(groups[mask], return_inverse=True)
    mean = delta.mean()
    sums = np.bincount(partition, weights=delta-mean)
    se = np.sqrt(len(orders)/(len(orders)-1) * (sums @ sums)) / len(delta)
    return [float(mean-1.96*se), float(mean+1.96*se)]


def report():
    raw, y, g, f = data()
    ids = raw.PassengerId.to_numpy(str)
    scores = {key:np.full(len(y), np.nan) for key in ['raw', 'real', 'permuted', 'plain', 'near', 'shifted']}
    for k in range(8):
        t, q = np.flatnonzero(f != k), np.flatnonzero(f == k)
        assert not np.isin(g[t], g[q]).any()
        for name in ['raw', 'real', 'permuted']:
            with np.load(packet_dir(0)/f'fold{k}.npz' if name == 'raw' else ART/f'{name}{k}.npz') as z:
                for key, expected in [('t',t), ('q',q), ('labels',y[t]), ('ids' if name == 'raw' else 'passenger', ids[q])]:
                    np.testing.assert_array_equal(z[key], expected)
                assert str(z['checkpoint' if name == 'raw' else 'sha']) == SHA
                scores[name][q] = z['p']
        with np.load(ROOT/f'artifacts/nov26_neighbor_booking_composition/full_{k}.npz') as z:
            for key, expected in [('t',t), ('q',q), ('labels',y[t])]:
                np.testing.assert_array_equal(z[key], expected)
            for name in ['plain','near','shifted']:
                scores[name][q] = z[name]
    assert all(np.isfinite(arr).all() for arr in scores.values())
    near, sham, original = [scores[key] >= .5 for key in ['real', 'permuted', 'raw']]
    with np.load(FARE/'aligned.npz') as z:
        for key, expected in [('passenger',ids), ('raw',scores['raw']), ('fold',f), ('y',y)]:
            np.testing.assert_array_equal(z[key], expected)
        anchor111, enabled, nori, known = [z[key] for key in ['champion111','eligible','own','known']]
    nine, valid, _, h = champions()
    np.testing.assert_array_equal(valid, known)
    bank = sources()[-1]['full']
    tab = panel_predictions(bank, 'full')
    before = np.column_stack([tab['neural_deck_raw'], scores['raw']])
    after = np.column_stack([tab['neural_deck_fine'], nori])
    price_flip = policies(anchor111, enabled & (anchor111 == nine), before, after)['fare_pair']
    anchor = anchor111 ^ price_flip
    verified = counts(price_flip, y, anchor111, known, f)
    assert (verified['fixed'], verified['harmed']) == (46,33)
    _, view, _ = inputs()
    frame = roster(raw)
    context, features = neighbor_features(frame, False)
    placebo, _ = neighbor_features(frame, True)
    paid_frame = frame[FEES].gt(0).any(axis=1)
    observed_paid = paid_frame | frame[FEES].notna().all(axis=1)
    personal = {'sleep':frame.CryoSleep.astype(float), 'any_spend':paid_frame.where(observed_paid).astype(float)}
    personal.update({f'{k}_used':frame[k].gt(0).where(frame[k].notna()).astype(float) for k in FEES})
    activity_check = {}
    for name, values in personal.items():
        j = features.index(name)
        truth = values.to_numpy()
        use = np.isfinite(truth) & np.isfinite(context[:,j]) & np.isfinite(placebo[:,j])
        activity_check[name] = dict(rows=int(use.sum()), **{
            key:float(np.mean((arr[use,j]-truth[use])**2)) for key, arr in [('real',context),('permuted',placebo)]})
    has = np.isfinite(context[:, 0])
    observed = has[:len(y)]
    paid = raw[FEES].gt(0).any(axis=1).to_numpy()
    single = raw.groupby(g).PassengerId.transform('size').eq(1).to_numpy()
    masks = dict(all=np.ones(len(y),bool), observed=observed, single=single, no_positive=~paid,
                 no_positive_single=(~paid)&single, shared_eleven=view['common'], new_orders=~np.isin(g,g[h]))
    catalogue = pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',
                            usecols=['row_id','PassengerId','y','all_recipes_still_wrong'], dtype={'PassengerId':str})
    rows = catalogue.row_id.to_numpy(int)
    np.testing.assert_array_equal(catalogue.PassengerId.to_numpy(), ids[rows])
    np.testing.assert_array_equal(catalogue.y.to_numpy(int), y[rows])
    old_errors = np.zeros(len(y),bool)
    old_errors[rows] = catalogue.all_recipes_still_wrong.to_numpy(bool)
    assert old_errors.sum() == 546
    masks['historically_all_wrong_546'] = old_errors
    flip = observed & (near != anchor)
    unique = flip & (sham == anchor) & (original == anchor)
    tree = scores['near'] >= .5
    tree_unique = (tree != anchor) & ((scores['plain'] >= .5) == anchor) & ((scores['shifted'] >= .5) == anchor)
    rules = dict(raw=original != anchor, shuffled=sham != anchor, true_neighbors=near != anchor,
                 true_neighbors_only=unique, corroborated=unique & (tree != anchor), independent_both=unique & tree_unique)
    tables = {}
    for scope, mask in masks.items():
        relevant = mask & known
        tables[scope] = dict(
            score={key:metrics(scores[key], y, mask) for key in ['raw','real','permuted']},
            vs_raw=counts(near != original, y, original, mask, f),
            vs_shuffled=counts(near != sham, y, sham, mask, f),
            vs_raw_cluster_accuracy_95=paired_accuracy_interval(near,original,y,g,mask),
            vs_shuffled_cluster_accuracy_95=paired_accuracy_interval(near,sham,y,g,mask),
            vs_161={name:counts(act,y,anchor,relevant,f) for name,act in rules.items()})
    common = view['common']
    tables['shared_eleven']['vs_eleven_decision'] = {
        name:counts((scores[name]>=.5) != view['parent'], y, view['parent'], common, f)
        for name in ['raw','real','permuted']}
    np.savez_compressed(ART/'aligned.npz', row=np.arange(len(y)), passenger=ids, y=y, fold=f,
                        reference161=anchor, known=known, common=common, observed=observed, **scores)
    result = dict(checkpoint=SHA, previous_related='neighbor_booking_composition_n26.py uses boosted trees, nearest 4/24 cabin numbers and a half-track displacement',
                  reference='frozen 161 (46 repairs and 33 harms vs 111 on 8532 known rows)',
                  context='all published passenger fields, excluding the target order; complete order assignments permuted within each deck/side',
                  neighbor_orders_each_side=WIDTH, permutation_seed=PERMUTATION_SEED, features=features,
                  train_coverage=int(observed.sum()), test_coverage=int(has[len(y):].sum()),
                  same_passenger_activity_brier_unlabeled=activity_check, results=tables)
    REPORT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(tables['all']))
