"""Estimate which independent booking or spending opinion is reliable on unseen orders."""
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import BILLS, ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities

MODES = ('registration','spending','mean','pooled','specialized','specialized_margins')
TAG = 'two_view_reliability'


def design(raw, groups):
    cabin = raw.Cabin.str.split('/', expand=True)
    reg = raw[['Age','HomePlanet','Destination','VIP']].copy()
    reg['Deck'], reg['Side'] = cabin[0], cabin[2]
    reg['CabinNumber'] = pd.to_numeric(cabin[1])
    reg['OrderSize'] = raw.groupby(groups).PassengerId.transform('size')
    reg['Position'] = raw.PassengerId.str.split('_').str[1].astype(int)
    reg['CabinOccupancy'] = raw.groupby('Cabin').PassengerId.transform('size')
    for col in ('Age','CabinNumber'):
        peer = reg[col].groupby(groups)
        count = peer.transform('count')-reg[col].notna().astype(int)
        reg['Other'+col] = (peer.transform('sum')-reg[col].fillna(0))/count.replace(0,np.nan)
    reg = pd.get_dummies(reg,columns=['HomePlanet','Destination','VIP','Deck','Side'],dummy_na=True)
    amounts = raw[BILLS]
    total = amounts.sum(axis=1,min_count=1)
    spend = np.column_stack([np.log1p(amounts), amounts.isna(), amounts.gt(0).sum(axis=1),
                             np.log1p(total), amounts.div(1+total,axis=0),
                             raw.CryoSleep.eq(True),raw.CryoSleep.isna()])
    return reg.to_numpy(float),spend


def tree(leaves):
    return HistGradientBoostingClassifier(max_iter=115, max_leaf_nodes=leaves,
            learning_rate=.06, min_samples_leaf=28, l2_regularization=3,
            early_stopping=False, random_state=411)


def context(scores):
    z = logit(np.clip(scores,.002,.998))
    return np.column_stack([z,np.abs(z[:,0])-np.abs(z[:,1]),z[:,0]*z[:,1]])


def predict(views,y,g,t,q):
    assert not np.isin(g[t],g[q]).any()
    inputs = np.column_stack(views)
    inner = np.zeros((len(t),2))
    query_scores = []
    for a,b in GroupKFold(3,shuffle=True,random_state=407).split(t,groups=g[t]):
        assert not np.isin(g[t[a]],g[t[b]]).any()
        parts = []
        for j,x in enumerate(views):
            head = tree(15).fit(x[t[a]],y[t[a]])
            inner[b,j] = head.predict_proba(x[t[b]])[:,1]
            parts.append(head.predict_proba(x[q])[:,1])
        query_scores.append(np.column_stack(parts))
    # The target stays the same across disagreement and agreement: is the spending view correct?
    bill = inner[:,1]>=.5
    conflict = (inner[:,0]>=.5)!=bill
    target = bill==y[t]
    margins = context(inner)
    joined = np.column_stack([margins,inputs[t]])
    pooled = tree(9).fit(joined,target)
    specialists = []
    for active in (~conflict,conflict):
        specialists.append((tree(9).fit(joined[active],target[active]),
                            tree(9).fit(margins[active],target[active])))
    predictions = []
    for p in query_scores:
        bill = p[:,1]>=.5
        split = (p[:,0]>=.5)!=bill
        z = context(p)
        x = np.column_stack([z,inputs[q]])
        estimates = np.zeros((len(q),3))
        estimates[:,0] = pooled.predict_proba(x)[:,1]
        for j,(special,score_only) in enumerate(specialists):
            mask = split == bool(j)
            if mask.any():
                estimates[mask,1] = special.predict_proba(x[mask])[:,1]
                estimates[mask,2] = score_only.predict_proba(z[mask])[:,1]
        predictions.append(np.column_stack([p,p.mean(axis=1),np.where(bill[:,None],estimates,1-estimates)]))
    return np.mean(predictions,axis=0)


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    gate,_ = gate_probabilities(names,half,full,y,g,fold)
    np.testing.assert_array_equal(ref,gate>=.5)
    views = design(raw,g)
    past = protocol(views,y,g,rows,of,'historical',predict,MODES,TAG)
    current = protocol(views,y,g,np.arange(len(y)),fold,'full',predict,MODES,TAG)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for key,val in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[key],val)
        remaining,shared = z['remaining'],z['shared']
    assert remaining.sum()==147 and shared.sum()==199
    unseen = ~np.isin(g,og)
    heads = current[:,:2]>=.5
    conflicts = heads[:,0]!=heads[:,1]
    report = dict(all_conflicts=int(conflicts.sum()), conflicts_among_147=int((conflicts & remaining).sum()),
                  both_right_among_147=int((remaining & (heads==y[:,None]).all(1)).sum()),
                  both_wrong_among_147=int((remaining & (heads!=y[:,None]).all(1)).sum()))
    for j,name in enumerate(MODES):
        report[name] = {}
        for weight in (1.,.25):
            pred = (weight*current[:,j]+(1-weight)*gate)>=.5
            historical = (weight*past[:,j]+(1-weight)*old[:,0])>=.5
            if weight<1:
                historical = np.where((old[:,0]>=.5)==champ,historical,champ)
            fixed = remaining & (pred==y)
            result = dict(full=stats(pred,ref,y,g,fold),historical=stats(historical,champ,oy,og,of),
                    unseen=stats(pred[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                    fixed_among_199=int((shared & (pred==y)).sum()), fixed_among_147=int(fixed.sum()),
                    same_person_fixed_in_both=int((fixed[rows] & (champ!=oy) & (historical==oy)).sum()),
                    fixed_147_ids=raw.loc[fixed,'PassengerId'].tolist())
            report[name][str(weight)] = result
            print(name,weight,[result[k]['net'] for k in ('full','historical','unseen')],fixed.sum(),flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve'/f'{TAG}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
