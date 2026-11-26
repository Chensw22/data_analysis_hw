"""Test whether a booking's shared mistakes can be anticipated from its other members' scores."""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from exhaustive_family_quorums_n26 import summary
from latent_passenger_difficulty_n26 import OUT, ROOT

DESTINATION = ROOT / 'reports/nov23_matched_learning_curve/booking_peer_error_signal.json'


def pairs_in(group, mask):
    members = pd.Series(np.flatnonzero(mask)).groupby(group[mask])
    pairs = [pair for _, rows in members for pair in combinations(rows, 2)]
    return np.array(pairs, dtype=int).T


def permutation_audit(group, mask, anchor, truth, size, regime, confidence):
    left, right = pairs_in(group, mask)
    error = anchor != truth

    def count(mistakes):
        both = mistakes[left] & mistakes[right]
        same = both & (anchor[left] == anchor[right])
        return np.array([both.sum(), same.sum(), np.unique(np.r_[left[both], right[both]]).size])

    observed = count(error)
    # Preserve decision direction and a passenger's own panel confidence when redistributing mistakes.
    bands = np.digitize(confidence, [.55, .65, .8, .95])
    strata = pd.DataFrame(dict(size=np.minimum(size, 4), regime=regime, anchor=anchor, band=bands))
    cells = [np.asarray(ix) for ix in strata[mask].groupby(list(strata.columns)).groups.values() if len(ix)>1]
    rng = np.random.default_rng(2326)
    simulations = []
    for _ in range(499):
        permuted = error.copy()
        for ix in cells:
            permuted[ix] = rng.permutation(error[ix])
        simulations.append(count(permuted))
    simulations = np.asarray(simulations)
    return dict(rows=int(mask.sum()), errors=int(error[mask].sum()), pairs=len(left),
                quantities=['both_wrong_pairs', 'same_direction_wrong_pairs', 'wrong_people_with_wrong_peer'],
                observed=observed.tolist(), shuffle_mean=simulations.mean(axis=0).round(2).tolist(),
                shuffle_interval=np.quantile(simulations, [.025, .975], axis=0).tolist(),
                upper_tail=((1+(simulations>=observed).sum(axis=0))/500).tolist())


def context_features(group, scores, regime, anchor):
    size = pd.Series(group).groupby(group).transform('size').to_numpy()
    mean = scores.mean(axis=1)
    votes = (scores >= .5).mean(axis=1)
    orientation = np.where(anchor, 1., -1.)
    # Companion features use model scores and observed attributes only. Every mean excludes the focal person.
    own = np.column_stack([mean, votes, scores.std(axis=1), *scores[:, -3:].T,
                           regime//2 == 0, regime//2 == 1])
    peer_sum = pd.DataFrame(own).groupby(group).transform('sum').to_numpy() - own
    peer = peer_sum / np.maximum(size[:, None]-1, 1)
    local = np.column_stack([anchor, (mean-.5)*orientation, (votes-.5)*orientation, scores.std(axis=1),
                             (scores[:, -3:]-.5)*orientation[:, None], regime//2==0, regime//2==1, np.log(size)])
    def assemble(p):
        return np.column_stack([local, p, (p[:,0]-.5)*orientation, (p[:,1]-.5)*(mean-.5),
                                p[:,2]*scores.std(axis=1)])
    return size, local, peer, assemble


def run():
    with np.load(OUT/'panel.npz') as packet:
        scores, group, regime, anchor, truth, fold, known, common, ids = [
            packet[k] for k in ('scores','groups','regime','anchor','y','fold','known','common','passenger')]
    n = len(truth)
    size, local, peer, assemble = context_features(group, scores, regime, anchor)
    ensemble = scores[:n].mean(axis=1)
    confidence = np.maximum(ensemble, 1-ensemble)
    scopes = dict(all_known_parties=known & (size[:n]>1), two=known & (size[:n]==2),
                  three_or_more=known & (size[:n]>=3), release_consensus=common & (size[:n]>1))
    audit = {k:permutation_audit(group[:n], m, anchor[:n], truth, size[:n], regime[:n], confidence)
             for k,m in scopes.items() if k in ('all_known_parties', 'release_consensus')}
    # Match placebo companion summaries on fold/roster, booking size, regime and panel majority.
    # Strata with only one booking retain their features and are counted explicitly.
    strata = pd.DataFrame(dict(fold=np.r_[fold,np.full(len(group)-n, -1)], size=size,
                               regime=regime, majority=scores.mean(axis=1)>=.5))
    random_peer = peer.copy()
    rng = np.random.default_rng(9326)
    moved = np.zeros(len(group), bool)
    for ix in strata[size>1].groupby(list(strata.columns)).groups.values():
        rows = np.asarray(ix)
        rows = rows[np.lexsort((rng.random(len(rows)),group[rows]))]
        max_booking = pd.Series(group[rows]).value_counts().max()
        if 2*max_booking > len(rows):
            continue
        donors = np.roll(rows, max_booking)
        assert np.all(group[rows]!=group[donors])
        random_peer[rows] = peer[donors]
        moved[rows] = True
    predictors = dict(person=local, real_peers=assemble(peer), placebo_peers=assemble(random_peer))
    wrong = anchor[:n] != truth
    predictions = {k:np.zeros(n) for k in predictors}
    test_predictions = {k:[] for k in predictors}
    for f in np.unique(fold):
        rows = np.flatnonzero((fold==f)&scopes['all_known_parties'])
        folds = GroupKFold(4,shuffle=True,random_state=3026+int(f)).split(rows, groups=group[rows])
        test = np.flatnonzero(size[n:]>1)+n
        for fit,query in [*((rows[a],rows[b]) for a,b in folds), (rows,test)]:
            assert not np.isin(group[fit], group[query]).any()
            for name, x in predictors.items():
                model = make_pipeline(StandardScaler(), LogisticRegression(C=.1,max_iter=500))
                model.fit(x[fit], wrong[fit])
                p = model.predict_proba(x[query])[:,1]
                if query[0] >= n:
                    test_predictions[name].append(p)
                else:
                    predictions[name][query] = p
    votes = {k:v>=.5 for k,v in predictions.items()}
    test_votes = {k:np.mean(v,axis=0)>=.5 for k,v in test_predictions.items()}
    for v in [votes, test_votes]:
        v['real_peers_only'] = v['real_peers'] & ~v['person'] & ~v['placebo_peers']
    corrections = {k:dict(scopes={name:summary(v,wrong,m,fold) for name,m in scopes.items()},
                          test_changes=int(test_votes[k].sum())) for k,v in votes.items()}
    comparison = {base:summary(votes['real_peers']!=votes[base], votes[base]!=wrong,
                                scopes['all_known_parties'],fold) for base in ('person','placebo_peers')}
    np.savez_compressed(OUT/'companion_probabilities.npz', passenger=ids[:n], test_ids=ids[test],
                        **predictions, **{'test_'+k:np.mean(v,axis=0) for k,v in test_predictions.items()})
    result = dict(pair_audit=audit, placebo_changed_train=int((moved[:n]&scopes['all_known_parties']).sum()),
                  placebo_changed_test=int(moved[n:].sum()), corrections=corrections, real_vs_control=comparison)
    DESTINATION.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(pair_audit=audit, real_vs_control=comparison)))


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
