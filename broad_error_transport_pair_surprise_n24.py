"""Test whether joint field surprises add information beyond their two marginal surprises."""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import BILLS, ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_solo_coverage_n24 import cached
from full_roster_two_view_reliability_n24 import tree

PAIRS = [('HomePlanet','CryoSleep'),('HomePlanet','Destination'),('HomePlanet','Deck'),
         ('CryoSleep','Age'),('Spa','VRDeck'),('FoodCourt','ShoppingMall')]
MODES = ('raw','pair_margins','pair_dependence','margins_dependence')
TAG = 'broad_error_transport_pair_surprise'


def fields(frame):
    cabin = frame.Cabin.str.split('/', expand=True)
    group = frame.PassengerId.str.split('_').str[0]
    values = {key:frame[key] for key in ['HomePlanet','CryoSleep','Destination','Age','VIP',*BILLS]}
    values.update(Deck=cabin[0], Side=cabin[2], CabinNumber=pd.to_numeric(cabin[1]),
                  OrderSize=group.map(group.value_counts()), Position=frame.PassengerId.str.split('_').str[1].astype(int))
    blocks, labels = {}, {}
    for key, col in values.items():
        if key in ['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']:
            codes = pd.factorize(col.astype('string').fillna('?'), sort=True)[0]
            blocks[key] = np.eye(codes.max()+1)[codes]
            labels[key] = codes
        else:
            a = col.to_numpy(float)
            blocks[key] = np.column_stack([np.nan_to_num(np.log1p(a),nan=-1.), ~np.isfinite(a)])
            if key=='Age' or key in BILLS:
                cuts = [13,18,30,55] if key=='Age' else [1,100,500,1500]
                labels[key] = np.where(np.isfinite(a),1+np.digitize(a,cuts),0)
    return blocks,labels,group.to_numpy()


def conditional_pairs(frame):
    blocks, labels, group = fields(frame)
    baseline = np.column_stack(list(blocks.values()))
    margins, interactions = [], []
    for a,b in PAIRS:
        remainder = np.column_stack([v for k,v in blocks.items() if k not in {a,b}])
        first,second = labels[a],labels[b]
        na,nb = first.max()+1,second.max()+1
        joint = first*nb+second
        features = np.empty((len(frame),3))
        for t,q in GroupKFold(4, shuffle=True, random_state=219).split(frame, groups=group):
            assert not np.isin(group[t],group[q]).any()
            classifier = ExtraTreesClassifier(n_estimators=56, min_samples_leaf=16, max_features=.8,
                                              n_jobs=2, random_state=81)
            classifier.fit(remainder[t],joint[t])
            # Prior smoothing handles a field pair absent from all training orders.
            prior = np.bincount(joint[t], minlength=na*nb)+1.
            p = np.tile(.03*prior/prior.sum(), (len(q),1))
            p[:,classifier.classes_] += .97*classifier.predict_proba(remainder[q])
            p = p.reshape(len(q),na,nb)
            np.testing.assert_allclose(p.sum((1,2)),1.,rtol=1e-10)
            idx = np.arange(len(q))
            la = np.log(p.sum(2)[idx,first[q]])
            lb = np.log(p.sum(1)[idx,second[q]])
            interaction = np.log(p[idx,first[q],second[q]]) - la - lb
            features[q] = np.column_stack([-la, -lb, interaction])
        margins.append(features[:,:2])
        interactions.append(features[:,2])
    return baseline, np.column_stack(margins), np.column_stack(interactions)


def run():
    raw,y,g,fold = data()
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda k:k!='Name')
    roster = pd.concat([raw,test],ignore_index=True)
    assert roster.PassengerId.is_unique
    full_features = conditional_pairs(roster)
    base, margin, inter = [matrix[:len(raw)] for matrix in full_features]
    inputs = [base,np.c_[base,margin],np.c_[base,inter],np.c_[base,margin,inter]]
    assert all(np.isfinite(x).all() for x in inputs)
    def predict(_, y, groups, t, q):
        assert not np.isin(groups[t],groups[q]).any()
        return np.column_stack([tree(19).fit(x[t],y[t]).predict_proba(x[q])[:,1] for x in inputs])
    names,half,bank = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,bank)
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    ix = np.arange(len(y))
    used = np.column_stack([cached(folder,'full',ix,fold,g,y) for folder in
                            ('response_channels','broad_error_transport_projection')])
    missed = (ref!=y) & ((used>=.5)!=y[:,None]).all(1)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    assert missed.sum()==848 and hard.sum()==147 and np.all(ref[hard]!=y[hard])
    full = protocol(base,y,g,ix,fold,'full',predict,MODES,TAG)
    past = protocol(base,y,g,rows,of,'historical',predict,MODES,TAG)
    unseen = ~np.isin(g,og)
    anchor = bank[:,names.index('mitra96')]
    report = dict(fields=PAIRS, auxiliary_roster_rows=len(roster),auxiliary_training='four held-booking folds; no outcome labels',
                  full_rows=len(y),baseline_errors=int((ref!=y).sum()),models={})
    for j,key in enumerate(MODES):
        for weight in (1.,.25):
            def decide(p,previous,alt):
                blended = (1-weight)*p+weight*alt>=.5
                return blended if weight==1. else np.where((p>=.5)==previous,blended,previous)
            prediction = decide(anchor,ref,full[:,j])
            historical = decide(old[:,0],champ,past[:,j])
            comparison = decide(anchor,ref,full[:,0])
            margin_only = decide(anchor,ref,full[:,1])
            def on(mask):
                fixed = mask & (ref!=y) & (prediction==y)
                return dict(n=int(mask.sum()),corrected=int(fixed.sum()),
                    harmed=int((mask & (ref==y) & (prediction!=y)).sum()),
                    extra_vs_raw=int((fixed & (comparison!=y)).sum()),
                    lost_vs_raw=int((mask & (comparison==y) & (prediction!=y)).sum()))
            report['models'][f'{key}_{weight}'] = dict(
                full=stats(prediction,ref,y,g,fold),historical=stats(historical,champ,oy,og,of),
                new_orders=stats(prediction[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                versus_raw=stats(prediction,comparison,y,g,fold),
                versus_margins=stats(prediction,margin_only,y,g,fold),six_model_errors=on(missed),common=on(hard))
    path = ROOT/'reports/nov23_matched_learning_curve'/f'{TAG}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    for name,r in report['models'].items():
        print(name, r['full']['corrected'],r['full']['harmed'],r['historical']['net'],r['new_orders']['net'],
              r['six_model_errors']['corrected'],r['common']['corrected'],flush=True)


if __name__=='__main__':
    with threadpool_limits(2):
        run()
