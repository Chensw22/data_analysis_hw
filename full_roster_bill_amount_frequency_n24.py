"""Test whether exact charge frequencies add outcome information beyond nearby amounts."""
import json

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT, BILLS, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_permutation_selection_n24 import cat, feature_bank

MODES = ('raw', 'frequency', 'frequency_outcome', 'frequency_permuted')
TAG = 'bill_amount_frequency'


def amount_features(amounts, labels, donors, query):
    """Use positive integer charges. Zero and missing amounts remain in the ordinary inputs."""
    features = [[], [], []]
    rng = np.random.default_rng(538)
    for amount in amounts.T:
        src = donors[amount[donors] > 0]
        values = amount[src]
        probe = amount[query]
        assert (values == values.astype(int)).all()
        # Preserve broad amount differences in the negative control.
        strata = values // 100
        permuted = labels[src].copy()
        for level in np.unique(strata):
            idx = np.flatnonzero(strata == level)
            permuted[idx] = rng.permutation(permuted[idx])
        sorted_sets = [np.sort(sample) for sample in (values, values[labels[src]], values[permuted])]

        def counts(radius):
            return [np.searchsorted(sample, probe + radius, side='right') -
                    np.searchsorted(sample, probe - radius) for sample in sorted_sets]

        broad, positive_broad, null_broad = counts(50)
        for radius in (0, 3):
            narrow, positive, null = counts(radius)
            enrichment = np.log((narrow + .5) / (broad + 1))
            features[0].append(np.where(probe > 0, enrichment, np.nan))
            for bucket, pos, wide in ((1, positive, positive_broad), (2, null, null_broad)):
                local_rate = (wide + 1) / (broad + 2)
                deviation = (pos - narrow * local_rate) / (narrow + 4)
                features[bucket].append(np.where(probe > 0, deviation, np.nan))
    return [np.column_stack(block) for block in features]


def scores(inputs, y, groups, train, query):
    raw, amounts = inputs
    inner = GroupKFold(3, shuffle=True, random_state=361)
    train_maps = []
    query_maps = []
    for fit, held in inner.split(train, groups=groups[train]):
        donors = train[fit]
        assert not np.isin(groups[donors], groups[np.r_[train[held],query]]).any()
        mapped = amount_features(amounts,y,donors,np.r_[train[held],query])
        if not train_maps:
            train_maps = [np.full((len(train), block.shape[1]), np.nan) for block in mapped]
        for target, block in zip(train_maps,mapped):
            target[held] = block[:len(held)]
        query_maps.append([block[len(held):] for block in mapped])
    query_maps = np.mean(query_maps,axis=0)
    results = []
    for use in ((),(0,),(0,1),(0,2)):
        source = np.column_stack([raw[train],*[train_maps[k] for k in use]])
        test = np.column_stack([raw[query],*[query_maps[k] for k in use]])
        model = cat(497).fit(source,y[train])
        results.append(model.predict_proba(test,thread_count=3)[:,1])
    return np.column_stack(results)


def run():
    raw,y,groups,fold = data()
    base,_,_ = feature_bank(raw)
    amounts = raw[BILLS].assign(total=raw[BILLS].sum(axis=1,min_count=len(BILLS))).fillna(-1).to_numpy()
    _,oy,old_groups,rows,old_fold,_,champ,_,_,_,_,old,_ = aligned()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,groups,fold,names,half,full)
    gate,_ = gate_probabilities(names,half,full,y,groups,fold)
    np.testing.assert_array_equal(gate >= .5, ref)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_archive.npz') as archive:
        for key, val in [('passenger', raw.PassengerId), ('fold', fold), ('y',y)]:
            np.testing.assert_array_equal(archive[key],val)
        common, residual = archive['baseline_misses'], archive['missed']
    assert common.sum()==223 and residual.sum()==199
    inputs = (base,amounts)
    past = protocol(inputs,y,groups,rows,old_fold,'historical',scores,MODES,TAG)
    present = protocol(inputs,y,groups,np.arange(len(y)),fold,'full',scores,MODES,TAG)
    unseen = ~np.isin(groups,old_groups)
    report = {}
    for i, mode in enumerate(MODES):
        report[mode] = {}
        for weight in (1., .25):
            pred = (weight*present[:,i] + (1-weight)*gate) >= .5
            historical = (weight*past[:,i] + (1-weight)*old[:,0]) >= .5
            if weight < 1:
                historical = np.where((old[:,0]>=.5)==champ,historical,champ)
            fixed = residual & (pred==y)
            report[mode][str(weight)] = dict(
                full=stats(pred,ref,y,groups,fold),
                historical=stats(historical,champ,oy,old_groups,old_fold),
                unseen=stats(pred[unseen],ref[unseen],y[unseen],groups[unseen],fold[unseen]),
                fixes_among_223=int((common & (pred==y)).sum()),
                fixes_among_199=int(fixed.sum()),
                repeated_fixes_among_199=int((fixed[rows] & (champ!=oy) & (historical==oy)).sum()),
                corrected_199=raw.loc[fixed,'PassengerId'].tolist())
        print(mode,report[mode]['1.0']['fixes_among_199'],report[mode]['0.25']['full']['net'],flush=True)
    for i,mode in enumerate(MODES[1:],1):
        report[mode]['versus_raw_cat'] = stats(present[:,i]>=.5,present[:,0]>=.5,y,groups,fold)
    path = ROOT/'reports/nov23_matched_learning_curve'/f'{TAG}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(3):
        run()
