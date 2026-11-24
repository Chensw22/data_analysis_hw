"""Test class-specific sampling offsets when upweighting passenger outcomes both views missed."""
import json

import numpy as np
from scipy.special import expit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import xgboost as xgb

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_permutation_selection_n24 import feature_bank
from full_roster_two_view_reliability_n24 import design, tree

TAG = 'failure_sampling_offset'
MODES = ('natural','enriched','remove_offset_at_prediction','offset_during_fit','matched_class_weights')


def weights(votes, labels):
    negative_weight = np.where(votes.all(1),4.,1.)
    positive_weight = np.where((~votes).all(1),4.,1.)
    offset = np.log(positive_weight/negative_weight)
    return np.where(labels,positive_weight,negative_weight),offset


def fit(x,y,q,weight,offset):
    source = xgb.DMatrix(x,label=y,weight=weight,base_margin=offset,nthread=2)
    query = xgb.DMatrix(q,base_margin=np.zeros(len(q)),nthread=2)
    model = xgb.train(dict(objective='binary:logistic',tree_method='hist',max_depth=4,
                           min_child_weight=5,eta=.055,nthread=2,seed=431),source,num_boost_round=300)
    return model.predict(query,output_margin=True)


def predict(inputs,y,g,t,q):
    x,views = inputs
    estimates = []
    for train,donor in GroupKFold(3,shuffle=True,random_state=673).split(t,groups=g[t]):
        s,d = t[train],t[donor]
        assert not np.isin(g[d],g[s]).any() and not np.isin(g[t],g[q]).any()
        ask = np.r_[s,q]
        votes = np.column_stack([tree(15).fit(v[d],y[d]).predict(v[ask]) for v in views])
        source_votes,query_votes = votes[:len(s)],votes[len(s):]
        weight,offset = weights(source_votes,y[s])
        _,query_offset = weights(query_votes,np.zeros(len(q),bool))
        # Normalization keeps the total weight, and hence regularization scale, fixed.
        weight = weight/weight.mean()
        blank = np.zeros(len(s))
        natural = fit(x[s],y[s],x[q],np.ones(len(s)),blank)
        enriched = fit(x[s],y[s],x[q],weight,blank)
        corrected = fit(x[s],y[s],x[q],weight,offset)
        # Same total weight in each class, without selecting specific shared failures.
        class_weight = np.array([weight[y[s]==label].mean() for label in (False,True)])
        placebo = fit(x[s],y[s],x[q],class_weight[y[s].astype(int)],
                      np.full(len(s),np.log(class_weight[1]/class_weight[0])))
        estimates.append(expit(np.column_stack([natural,enriched,enriched-query_offset,corrected,placebo])))
    return np.mean(estimates,axis=0)


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    gate,_ = gate_probabilities(names,h,full,y,g,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    baseline = json.loads((ROOT/'reports/nov23_matched_learning_curve/honest_failure_partitions.json').read_text())
    earlier = raw.PassengerId.isin(baseline['failures']['1.0']['fixed_ids']).to_numpy()
    assert hard.sum()==147 and earlier.sum()==7 and not (earlier & ~hard).any()
    inputs = (feature_bank(raw)[0],design(raw,g))
    a = protocol(inputs,y,g,rows,of,'historical',predict,MODES,TAG)
    b = protocol(inputs,y,g,np.arange(len(y)),fold,'full',predict,MODES,TAG)
    report = {}
    unseen = ~np.isin(g,og)
    for j,name in enumerate(MODES):
        report[name] = {}
        for weight in (1.,.25):
            pred = (weight*b[:,j]+(1-weight)*gate)>=.5
            past = (weight*a[:,j]+(1-weight)*old[:,0])>=.5
            if weight<1:
                past = np.where((old[:,0]>=.5)==champ,past,champ)
            rescued = hard & (pred==y)
            result = dict(full=stats(pred,ref,y,g,fold),historical=stats(past,champ,oy,og,of),
                          unseen=stats(pred[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                          fixed_among_147=int(rescued.sum()), repeated=int((rescued[rows] & (past==oy) & (champ!=oy)).sum()),
                          recovered_of_seven=int((rescued & earlier).sum()),fixed_ids=raw.loc[rescued,'PassengerId'].tolist())
            report[name][str(weight)] = result
            print(name,weight,[result[k]['net'] for k in ('full','historical','unseen')],rescued.sum(),flush=True)
        report[name]['versus_natural'] = stats(b[:,j]>=.5,b[:,0]>=.5,y,g,fold)
    path = ROOT/'reports/nov23_matched_learning_curve'/f'{TAG}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
