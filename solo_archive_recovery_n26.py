"""Learn a short list of repeatable dissent rules, then replay it on entirely new solo bookings."""
import itertools
import json

import numpy as np
from scipy.special import expit
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from cross_manifest_views_n26 import predictions
from disjoint_tree_fields_n25 import ART, DEST
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from full_roster_solo_coverage_n24 import EXPERIMENTS, cached
from outstanding_error_families_n26 import observed
from receipt_expert_transfer_n26 import read as receipt_scores


def archive(tag, idx, fold, g, y, path, *, probabilities=False):
    with np.load(path) as z:
        names = z['names'].tolist()
        matrices = [z[tag]]
    for family, variants in EXPERIMENTS.items():
        matrices.append(cached(family, tag, idx, fold, g, y))
        names.extend(f'{family}/{v}' for v in variants)
    _, bills = receipt_scores(tag, idx, fold)
    _, novel = predictions(tag, idx, fold)
    names.extend(['expected_bill', 'bill_template', 'rounding', 'charge_tree', 'charge_tabicl',
                  'cabin_tree', 'cabin_tabicl', 'peer_tree', 'peer_tabicl'])
    plain = np.column_stack(matrices)
    novel = novel[:, [0, 2]].reshape(len(idx), 4)
    assert len(set(names)) == len(names) == plain.shape[1] + bills.shape[1] + novel.shape[1]
    votes = np.column_stack([plain >= .5, bills >= 0, novel >= .5])
    if probabilities:
        scores = np.column_stack([plain, expit(bills), novel])
        assert np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))
        np.testing.assert_array_equal(scores >= .5, votes)
        return names, scores
    return names, votes


def rules(dissent, anchor, prior, names):
    # A dissent rule may be corroborated by another family or conditioned on the incumbent.
    flags = [*dissent.T, ~anchor, anchor, abs(prior-.5)<.2, abs(prior-.5)>=.2]
    labels = [*names, 'negative', 'positive', 'low_confidence', 'high_confidence']
    indices = [(i,) for i in range(len(names))]
    indices += [(i,j) for i,j in itertools.combinations(range(len(labels)),2)
                if i<len(names) and labels[i].split('/')[0] != labels[j].split('/')[0]]
    matrix = np.column_stack([flags[i[0]] if len(i)==1 else flags[i[0]]&flags[i[1]] for i in indices])
    return matrix, [' & '.join(labels[j] for j in i) for i in indices]


def learn(matrix, wrong, train):
    # Each additional rule must provide positive incremental net in both booking splits.
    rng = np.random.default_rng(26110)
    a,b = np.array_split(rng.permutation(train),2)
    choices = []
    for _ in range(6):
        gains = []
        for t in [a,b]:
            gains.append(matrix[t[wrong[t]]].sum(0,dtype=int) - matrix[t[~wrong[t]]].sum(0,dtype=int))
        lower = np.minimum(*gains)
        best = int(lower.argmax())
        if lower[best] < 2:
            break
        choices.append(best)
        a = a[~matrix[a,best]]
        b = b[~matrix[b,best]]
    return choices


def ticker(mask, wrong, action):
    return dict(rows=int(mask.sum()), errors=int((wrong&mask).sum()), fixed=int((mask&wrong&action).sum()),
                harmed=int((mask&~wrong&action).sum()))


def run():
    raw,y,g,f = data()
    _,_,_,rows,model_names,p,h,native,old_native,ref = panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0] = z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par = z['fold'],z['nine'],z['v131']
    path=ART/'nov24_remaining_cases/archive_scores.npz'
    with np.load(path) as z:
        for k,v in [('row',rows),('old_fold',hf),('fold',f),('y',y),('passenger',raw.PassengerId)]:
            np.testing.assert_array_equal(z[k],v)
    awake = raw.CryoSleep.eq(False).to_numpy()
    solo = ~raw.PassengerId.str.split('_').str[0].duplicated(keep=False).to_numpy()
    results = {}; records = {}
    for tag,ids,fold,bank,prior,branch in [('historical',rows,hf,h,old_native,[par,old,old]),
                                            ('full',np.arange(len(y)),f,p,native,ref)]:
        v = observed(ids,fold,bank,model_names,prior,*branch,awake,tag)
        anchor = v['anchor']
        family = v['eligible'] & (v['votes']==anchor[:,None]).all(1) & solo[ids]
        eligible = v['known'] & solo[ids]
        names,votes = archive(tag,ids,fold,g,y,path)
        wrong = y[ids]!=anchor
        disagree = votes!=anchor[:,None]
        matrix,labels = rules(disagree,anchor,prior,names)
        if tag=='full':
            assert (wrong&family).sum()==396 and names==records['historical']['names']
        out = np.zeros(len(ids),bool)
        for k in np.unique(fold):
            ix = np.flatnonzero(eligible&(fold==k))
            for t,q in KFold(3,shuffle=True,random_state=147).split(ix):
                assert not np.isin(g[ids[ix[t]]],g[ids[ix[q]]]).any()
                choices = learn(matrix,wrong,ix[t])
                out[ix[q]] = matrix[ix[q]][:,choices].any(1)
        any_dissent = disagree.any(1)
        masks = {'solo':eligible, 'family_consensus':family}
        if tag=='historical':
            hist_choices = learn(matrix,wrong,np.flatnonzero(eligible))
        else:
            unseen = ~np.isin(g[ids],g[rows])
            fixed_rules = matrix[:,hist_choices].any(1)
            results['historical_rules_on_unseen'] = {k:ticker(m&unseen,wrong,fixed_rules) for k,m in masks.items()}
            results['historical_rules']=[labels[i] for i in hist_choices]
            fresh_folds=fold[eligible&unseen]
            results['historical_rules_unseen_fold_net']=[int((fixed_rules*(2*wrong.astype(int)-1))[eligible&unseen][fresh_folds==k].sum()) for k in np.unique(fresh_folds)]
        results[tag]={k:{'oracle':ticker(m,wrong,any_dissent),'learned_rules':ticker(m,wrong,out),
                         'unreached_errors':int((m&wrong&~any_dissent).sum())} for k,m in masks.items()}
        if tag=='full':
            paid = raw[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].gt(0).sum(axis=1).to_numpy()
            all_yes = family & ~any_dissent
            features = {'all':np.ones(len(y),bool), 'zero_recorded_spend':paid==0,
                        'some_recorded_spend':paid>0, 'awake':awake, 'asleep':raw.CryoSleep.eq(True).to_numpy(),
                        'earth':raw.HomePlanet.eq('Earth').to_numpy(), 'positive_prediction':anchor,
                        'negative_prediction':~anchor, 'native_at_least_80pct':np.maximum(prior,1-prior)>=.8}
            results['all_experts_agree_context'] = {k:{'rows':int((all_yes&m).sum()),
                'errors':int((all_yes&m&wrong).sum())} for k,m in features.items()}
        records[tag]=dict(names=names,dissent=disagree,family=family,anchor=anchor)
    hist,cur=records['historical'],records['full']
    matched=hist['family']&cur['family'][rows]&(hist['anchor']==cur['anchor'][rows])
    w=hist['anchor']!=y[rows]
    overlap=hist['dissent']&cur['dissent'][rows]
    results['repeat_same_passenger']={'same_expert':ticker(matched,w,overlap.any(1)),
                                      'each_expert':{s:ticker(matched,w,mask) for s,mask in zip(names,overlap.T)}}
    results['model_count']=len(names)
    DEST.with_name('solo_archive_recovery.json').write_text(json.dumps(results,indent=2)+'\n')
    for k in ('full','historical','historical_rules_on_unseen','historical_rules'):
        print(k,results[k],flush=True)
    print('repeated',results['repeat_same_passenger']['same_expert'])


if __name__=='__main__':
    with threadpool_limits(2):
        run()
