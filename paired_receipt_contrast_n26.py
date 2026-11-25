"""Test whether real booking partners' receipts carry conditional evidence for a focal passenger."""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logit

from audit_fare_routes_n26 import DECK, held_fare, reviews
from exaone_validation_data import compact_features
from export_fare_deck_n26 import combine
from full_roster_booking_validation_n23 import ROOT, data
from price_implied_deck_n26 import FEES
from tabicl_consensus_review_n26 import sources

REPORT = ROOT/'reports/nov23_matched_learning_curve/paired_receipt_contrast.json'


def companions(raw, group):
    """Choose a known cabin mate first, then the closest ticket position. Solo rows point to self."""
    position = raw.PassengerId.str.split('_').str[1].astype(int).to_numpy()
    cabin = raw.Cabin.fillna('').to_numpy()
    peer = np.arange(len(raw))
    size = raw.PassengerId.groupby(group).transform('size').to_numpy()
    for order in raw.groupby(group, sort=False).indices.values():
        for i in order:
            others = order[order != i]
            if len(others):
                same = (cabin[others] == cabin[i]) & (cabin[i] != '')
                peer[i] = others[np.lexsort((position[others], abs(position[others]-position[i]), ~same))[0]]
    assert (group == group[peer]).all() and np.all((peer == np.arange(len(group))) == (size == 1))
    return peer, size, (cabin == cabin[peer]) & (cabin != '')


def unrelated(rows, peer, group, strata, seed):
    """Replace the peer with the peer of a different focal passenger in the same split."""
    rng = np.random.default_rng(seed)
    matched = np.full(len(rows), -1)
    # Coarsen only when no example in a different order matches all available attributes.
    available = pd.DataFrame(strata[rows])
    for width in range(available.shape[1], 0, -1):
        cells = available.groupby(list(available.columns[:width]), sort=False).indices
        for cell in cells.values():
            waiting = cell[matched[cell] < 0]
            if not len(waiting):
                continue
            for loc in waiting:
                candidates = rows[cell[group[rows[cell]] != group[rows[loc]]]]
                if len(candidates):
                    matched[loc] = peer[rng.choice(candidates)]
        if np.all(matched >= 0):
            break
    assert np.all(matched >= 0) and np.all(group[rows] != group[matched])
    return matched


def model(x, y):
    cats = [k for k in x if pd.api.types.is_string_dtype(x[k])]
    return CatBoostClassifier(iterations=280, depth=4, learning_rate=.055, l2_leaf_reg=7,
                              cat_features=cats, random_seed=9472, thread_count=2,
                              verbose=False, allow_writing_files=False).fit(x, y.astype(int))


def pair_features(frame, bills, focus, peer):
    x = frame.iloc[focus].reset_index(drop=True)
    other = frame.iloc[peer].drop(columns=['CabinNumber','booking_size','same_room']).reset_index(drop=True)
    x = x.join(other.add_prefix('other_'))
    own, their = bills[focus], bills[peer]
    active = (own>0) & (their>0)
    x['common_services'] = active.sum(1)
    x['different_services'] = ((own>0) != (their>0)).sum(1)
    x['age_gap'] = x.Age - x.other_Age
    # Receipt composition is observable for both people even if their outcomes are unknown.
    for j, fee in enumerate(FEES):
        x['receipt_difference_'+fee] = np.log1p(own[:, j])-np.log1p(their[:, j])
    return x


def fit_scores(raw, y, g, fold, peer, size, same_room):
    frame = compact_features(raw)
    bills = raw[FEES].to_numpy(float)
    frame[FEES] = np.log1p(frame[FEES])
    frame['booking_size'] = size
    frame['same_room'] = same_room.astype(int)
    strata = np.column_stack([np.ones(len(g)), np.minimum(size,4), frame.HomePlanet,
                              same_room, frame.Deck, frame.CryoSleep, np.any(bills>0, axis=1)])
    names = ('personal','direct_real','direct_unrelated','ratio_real','ratio_unrelated',
             'rank_real','rank_unrelated','mixed_real','mixed_unrelated')
    scores = {name:np.full(len(y),np.nan) for name in names}
    scores['unrelated_index'] = np.arange(len(y))
    for k in np.unique(fold):
        train = np.flatnonzero(fold!=k)
        t = train[size[train]>1]
        q = np.flatnonzero((fold==k)&(size>1))
        assert not np.isin(g[t],g[q]).any()
        unrelated_t = unrelated(t,peer,g,strata,3451+int(k))
        unrelated_q = unrelated(q,peer,g,strata,4352+int(k))
        scores['unrelated_index'][q] = unrelated_q
        xt = pair_features(frame,bills,t,peer[t])
        nt = pair_features(frame,bills,t,unrelated_t)
        xq = pair_features(frame,bills,q,peer[q])
        nq = pair_features(frame,bills,q,unrelated_q)
        prior = model(frame.iloc[train],y[train]).predict_proba(frame.iloc[q])[:,1]
        scores['personal'][q] = prior
        direct = model(xt,y[t])
        # Equal real and synthetic samples estimate the conditional density ratio. Both training
        # pairs share exactly the same focal passenger and outcome; only the counterpart changes.
        real_fake = pd.concat([xt,nt],ignore_index=True).assign(hypothesis=np.tile(y[t],2).astype(int))
        discriminator = model(real_fake,np.repeat([1,0],len(t)))
        # A separate order-of-outcomes ranker learns only from unlike-outcome pairs. At query time
        # it can act only when a separately trained model predicts unlike outcomes from receipts.
        mixed = y[t]!=y[peer[t]]
        assert mixed.any()
        mix_model = model(xt,mixed)
        ranker = model(xt.loc[mixed], y[t][mixed])
        for name,x,partners in [('real',xq,peer[q]),('unrelated',nq,unrelated_q)]:
            scores['direct_'+name][q] = direct.predict_proba(x)[:,1]
            likelihood = [discriminator.predict(x.assign(hypothesis=h),prediction_type='RawFormulaVal') for h in [0,1]]
            scores['ratio_'+name][q] = expit(logit(prior)+likelihood[1]-likelihood[0])
            scores['mixed_'+name][q] = mix_model.predict_proba(x)[:,1]
            reverse = pair_features(frame,bills,partners,q)
            own = ranker.predict(x,prediction_type='RawFormulaVal')
            other = ranker.predict(reverse,prediction_type='RawFormulaVal')
            scores['rank_'+name][q] = expit((own-other)/2)
        print('fold',k,flush=True)
    return scores


def run():
    raw,y,g,fold = data()
    peer,size,same = companions(raw,g)
    views = sources()[-1]
    v = views['full']
    archive = held_fare(v,'full',with_routes=False)
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    assert not set(g).intersection(test.PassengerId.str.split('_').str[0])
    with np.load(DECK/'deck_estimates.npz') as z:
        eligible = z['eligible'][:len(y)]
        decks_agree = (z['label'][:len(y)]>=0)&(z['label'][:len(y)]==z['fine'][:len(y)].argmax(1))
    scores = fit_scores(raw,y,g,fold,peer,size,same)
    report = {'selection':'same known cabin, then closest ticket position; one companion per non-solo passenger',
              'unrelated_control':'different booking, matching focal size, origin, co-room status, deck, sleep and payment when possible',
              'train_and_test_shared_orders':0}
    paid = raw[FEES].gt(0).any(axis=1).to_numpy()
    sleeping = raw.CryoSleep.eq(True).to_numpy()
    awake = raw.CryoSleep.eq(False).to_numpy()
    batches = []
    for tag,q,review in reviews(views,g):
        if tag == 'bookingmix':
            batches.append((tag,q,review))
            fresh = ~np.isin(g[q],g[views['historical']['ids']])
            batches.append(('fresh_bookingmix',q[fresh],review[fresh]))
    for tag,q,review in batches:
        parent = combine(review,archive['neural_deck_raw'][q],archive['neural_deck_fine'][q],eligible[q])>=.5
        wrong = parent!=y[q]
        hard = v['common'][q]&eligible[q]&decks_agree[q]
        multi = size[q]>1
        masks = {'all':np.ones(len(q),bool),'multiple':multi,'two_passengers':size[q]==2,'common':v['common'][q],
                 'unanimous_matching_deck':hard,'unanimous_matching_deck_with_peer':hard&multi,
                 'unanimous_matching_deck_solo':hard&~multi}
        profile = {}
        for name,mask in [('single',~multi),('pair',size[q]==2),('three_plus',size[q]>=3),('same_cabin',multi&same[q]),
                          ('different_or_unknown_cabin',multi&~same[q]),('paid_peer',multi&paid[peer[q]]),
                          ('sleeping_peer',multi&sleeping[peer[q]]),('awake_peer',multi&awake[peer[q]]),
                          ('focal_predicted_transported',parent),
                          ('peer_base_vote_same_as_focal',multi&(v['anchor'][peer[q]]==parent)),
                          ('peer_base_vote_correct_posthoc',multi&(v['anchor'][peer[q]]==y[peer[q]])),
                          ('different_outcomes_posthoc',multi&(y[q]!=y[peer[q]]))]:
            profile[name] = dict(people=int((hard&mask).sum()),errors=int((hard&mask&wrong).sum()))
        actions = {}
        prior_ok = (scores['personal'][q]>=.5)==parent
        for name in ('real','unrelated'):
            opposed = {}
            for method in ('direct','ratio'):
                opposed[method] = multi&((scores[method+'_'+name][q]>=.5)!=parent)
                actions[method+'_'+name] = opposed[method]
                actions[method+'_induced_'+name] = opposed[method]&prior_ok
            rank_change = multi&((scores['rank_'+name][q]>=.5)!=parent)
            actions['rank_if_mixed_'+name] = rank_change&(scores['mixed_'+name][q]>=.5)
            # When the pair is forecast to split, use the independently predicted counterpart
            # to resolve the focal outcome. This uses a held-out vote, never their outcome.
            counterpart = peer[q] if name=='real' else scores['unrelated_index'][q]
            counterpart_vote = scores['personal'][counterpart]>=.5
            actions['opposite_peer_if_mixed_'+name] = multi&(scores['mixed_'+name][q]>=.5)&(parent==counterpart_vote)
            actions['rank_without_mixed_guard_'+name] = rank_change
            actions['all_three_'+name] = opposed['direct']&opposed['ratio']&actions['rank_if_mixed_'+name]
        for method in ('direct_induced','ratio_induced','rank_if_mixed','opposite_peer_if_mixed','rank_without_mixed_guard'):
            actions[method+'_only_with_real_peer'] = actions[method+'_real']&~actions[method+'_unrelated']
        def summary(mask,flip):
            return dict(rows=int(mask.sum()),errors=int((mask&wrong).sum()),fixed=int((mask&wrong&flip).sum()),
                        harmed=int((mask&~wrong&flip).sum()))
        report[tag] = dict(profile=profile,actions={key:{scope:summary(mask,flip) for scope,mask in masks.items()}
                                                   for key,flip in actions.items()})
        print(tag,len(q),profile,flush=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    run()
