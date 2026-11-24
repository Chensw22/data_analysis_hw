"""Grow forests on enriched shared failures and estimate leaf rates on independent natural orders."""
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_two_view_reliability_n24 import design, tree

MODES = ('natural','natural_honest','failures','failures_honest','placebo','placebo_honest')
TAG = 'honest_failure_partitions'


def fit_partition(x,y,structure,calibration,q,weights,seed):
    forest = RandomForestClassifier(n_estimators=48,min_samples_leaf=12,max_features=.8,
                n_jobs=2,random_state=seed)
    forest.fit(x[structure],y[structure],sample_weight=weights)
    calibrated = np.zeros(len(q))
    prior = y[calibration].mean()
    for tr in forest.estimators_:
        leaves = tr.apply(x[calibration])
        counts = np.bincount(leaves,minlength=tr.tree_.node_count)
        positives = np.bincount(leaves,weights=y[calibration],minlength=tr.tree_.node_count)
        calibrated += ((positives+5*prior)/(counts+5))[tr.apply(x[q])]
    return forest.predict_proba(x[q])[:,1], calibrated/len(forest.estimators_)


def predict(views,y,g,t,q):
    x = np.nan_to_num(np.column_stack(views),nan=-99.).astype(np.float32)
    parts = [t[b] for _,b in GroupKFold(3,shuffle=True,random_state=673).split(t,groups=g[t])]
    estimates = []
    for k,structure in enumerate(parts):
        source,calibration = parts[(k+1)%3],parts[(k+2)%3]
        assert not np.isin(g[t],g[q]).any()
        assert not np.isin(g[structure],g[np.r_[source,calibration]]).any()
        assert not np.isin(g[source],g[calibration]).any()
        failures = np.ones(len(structure),bool)
        for view in views:
            head = tree(15).fit(view[source],y[source])
            failures &= head.predict(view[structure])!=y[structure]
        # Shuffle failure membership within actual class to preserve the enrichment's class balance.
        shuffled = failures.copy()
        rng = np.random.default_rng(733+k)
        for side in (False,True):
            mask = y[structure]==side
            shuffled[mask] = rng.permutation(failures[mask])
        preds = []
        for weight in (np.ones(len(structure)),1+3*failures,1+3*shuffled):
            preds.extend(fit_partition(x,y,structure,calibration,q,weight,229+k))
        estimates.append(np.column_stack(preds))
    return np.mean(estimates,axis=0)


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    gate,_ = gate_probabilities(names,half,full,y,g,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for key,val in [('passenger',raw.PassengerId),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[key],val)
        hard = z['remaining']
    assert hard.sum()==147
    views = design(raw,g)
    a = protocol(views,y,g,rows,of,'historical',predict,MODES,TAG)
    b = protocol(views,y,g,np.arange(len(y)),fold,'full',predict,MODES,TAG)
    report = {}
    unseen = ~np.isin(g,og)
    for j,name in enumerate(MODES):
        report[name] = {}
        pred0 = b[:,j]>=.5
        for weight in (1.,.25):
            pred = (weight*b[:,j]+(1-weight)*gate)>=.5
            past = (weight*a[:,j]+(1-weight)*old[:,0])>=.5
            if weight<1:
                past = np.where((old[:,0]>=.5)==champ,past,champ)
            fixed = hard & (pred==y)
            report[name][str(weight)] = dict(
                full=stats(pred,ref,y,g,fold), historical=stats(past,champ,oy,og,of),
                unseen=stats(pred[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                fixed_among_147=int(fixed.sum()), repeated=int((fixed[rows] & (champ!=oy) & (past==oy)).sum()),
                fixed_ids=raw.loc[fixed,'PassengerId'].tolist())
        report[name]['vs_matching_natural'] = stats(pred0,b[:,j%2]>=.5,y,g,fold)
        print(name,[report[name]['1.0'][k]['net'] for k in ('historical','full','unseen')],
              report[name]['1.0']['fixed_among_147'],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve'/f'{TAG}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
