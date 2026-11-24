"""Measure when two failure-enriched learners jointly offer a useful contrary decision."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities


def archived(tag,folder,rows,fold,g,y):
    chunks = []
    ix = []
    for f in np.unique(fold):
        mask = fold==f
        q = rows[mask]
        t = rows[~mask & ~np.isin(g[rows],g[q])]
        with np.load(ROOT/f'artifacts/nov24_{folder}/{tag}_{f}.npz') as z:
            for k,v in [('q',q),('t',t),('truth',y[t])]:
                np.testing.assert_array_equal(z[k],v)
            chunks.append(z['p'])
        ix.append(np.flatnonzero(mask))
    return np.vstack(chunks)[np.argsort(np.concatenate(ix))]


def decisions(raw,y,g,rows,fold,baseline,base_p,tag):
    forest = archived(tag,'honest_failure_partitions',rows,fold,g,y)
    boost = archived(tag,'failure_sampling_offset',rows,fold,g,y)
    both = ((forest[:,2]>=.5)!=baseline) & ((boost[:,1]>=.5)!=baseline)
    restored = ((forest[:,3]>=.5)!=baseline) & ((boost[:,3]>=.5)!=baseline)
    natural = ((forest[:,0]>=.5)!=baseline) & ((boost[:,0]>=.5)!=baseline)
    policies = dict(two_weighted=both,plus_two_restored=both&restored,plus_two_natural=both&natural,
                    weighted_only=both & ((forest[:,0]>=.5)==baseline) & ((boost[:,0]>=.5)==baseline))
    def strength(p):
        return (1-2*baseline.astype(float))[:,None]*logit(p.clip(.002,.998))
    core = strength(np.column_stack([base_p,forest[:,2],boost[:,1]]))
    signals = strength(np.column_stack([forest[:,[0,3,4]],boost[:,[0,2,3,4]]]))
    cases = raw.iloc[rows]
    context = np.column_stack([cases.Age.lt(13),cases.Age.isna(),cases.CryoSleep.eq(True),
                               cases.CryoSleep.isna(),cases.VIP.eq(True),
                               cases[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].gt(0).sum(axis=1),
                               cases.HomePlanet.eq('Earth'),baseline])
    for name,features in [('margins',core),('margins_and_controls',np.column_stack([core,signals,context]))]:
        flip = np.zeros(len(rows),bool)
        for f in np.unique(fold):
            ar = np.flatnonzero((fold==f) & both)
            for a,b in GroupKFold(4,shuffle=True,random_state=399).split(ar,groups=g[rows[ar]]):
                t,q = ar[a],ar[b]
                assert not np.isin(g[rows[t]],g[rows[q]]).any()
                clf = make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=400))
                clf.fit(features[t],y[rows[t]]!=baseline[t])
                flip[q] = clf.predict_proba(features[q])[:,1]>=.5
        policies[name] = flip
    return {k:baseline^mask for k,mask in policies.items()}


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    gate,_ = gate_probabilities(names,h,full,y,g,fold)
    np.testing.assert_array_equal(ref,gate>=.5)
    # Keep the historical Mitra score and boolean v131 override as distinct inputs.
    a = decisions(raw,y,g,rows,of,champ,old[:,0],'historical')
    b = decisions(raw,y,g,np.arange(len(y)),fold,ref,gate,'full')
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    unseen = ~np.isin(g,og)
    report = {}
    for k,p in b.items():
        fixed = hard & (p==y)
        report[k] = dict(full=stats(p,ref,y,g,fold), historical=stats(a[k],champ,oy,og,of),
                         unseen=stats(p[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                         fixed_among_147=int(fixed.sum()),fixed_ids=raw.loc[fixed,'PassengerId'].tolist(),
                         repeated=int((fixed[rows]&(champ!=oy)&(a[k]==oy)).sum()))
        report[k]['directions'] = {}
        for side in (False,True):
            pr = np.where(ref==side,p,ref)
            past = np.where(champ==side,a[k],champ)
            report[k]['directions'][str(side)] = dict(full=stats(pr,ref,y,g,fold),
                historical=stats(past,champ,oy,og,of),
                unseen=stats(pr[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                fixed_among_147=int((hard & (pr==y)).sum()))
        r = report[k]
        print(k,r['full']['corrected'],r['full']['harmed'],r['historical']['net'],r['unseen']['net'],fixed.sum())
    path = ROOT/'reports/nov23_matched_learning_curve/weighted_rescue_agreement.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
