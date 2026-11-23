"""Fit a classifier with exemplar features built from separate, labeled reference bookings."""
import numpy as np
from catboost import CatBoostClassifier
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_dependence_anchors_n23 import REPORT
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts, coordinates
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_passenger_prototype_cover_n23 import BUDGET, choose, neighborhood, similarity
from exaone_validation_data import CATEGORICAL


def represented(frame,groups,y,t,ids):
    features,_ = coordinates(frame,t)
    neighbors,radius,distance = neighborhood(features,groups,t)
    strength = similarity(distance,radius[:,None])
    rng = np.random.default_rng(722)
    tables = {}
    for key in ('random','discriminative'):
        data = frame.iloc[ids].copy()
        for label in (False,True):
            pool = np.flatnonzero(y[t]==label)
            support = (rng.choice(pool,BUDGET,replace=False) if key=='random'
                       else choose(neighbors,strength,y[t],label,1.))
            dist = cdist(features[ids],features[t[support]])/radius[support]
            closest = np.partition(dist,4,axis=1)[:,:5]
            prefix = f'example_{int(label)}_'
            data[prefix+'closest'] = closest.min(1)
            data[prefix+'first_five_mean'] = closest.mean(1)
            data[prefix+'supported'] = (dist<1).sum(1)
            data[prefix+'affinity'] = np.maximum(1-dist**2,0).sum(1)
        tables[key] = data
    return tables


def predict(train,y,query):
    fitted = CatBoostClassifier(iterations=360,depth=5,learning_rate=.055,l2_leaf_reg=4,
                                verbose=False,allow_writing_files=False,thread_count=2,random_seed=394)
    fitted.fit(train,y,cat_features=[k for k in CATEGORICAL if k in train])
    return fitted.predict_proba(query)[:,1]


def run_fold(frame,groups,y,t,q):
    # A reference half selects and labels exemplars. The other half trains the tree;
    # swapping the halves lets each booking contribute in both roles, never to itself.
    result = {'direct':np.zeros(len(q)),'random':np.zeros(len(q)),'discriminative':np.zeros(len(q))}
    for a,b in GroupKFold(2,shuffle=True,random_state=722).split(t,groups=groups[t]):
        ref,train = t[a],t[b]
        assert not np.intersect1d(groups[ref],groups[train]).size
        tables = represented(frame,groups,y,ref,np.r_[train,q])
        direct = predict(frame.iloc[train],y[train],frame.iloc[q])
        result['direct'] += .5*direct
        for key,data in tables.items():
            result[key] += .5*predict(data.iloc[:len(train)],y[train],data.iloc[len(train):])
    return result


def main():
    row,y,frame,groups,_ = load()
    wrong,right,hard = cohorts(row,y)
    plans, history = {}, {}
    directory = ART/'nov23_exemplar_feature_transfer'
    directory.mkdir(exist_ok=True)
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as data:
            np.testing.assert_array_equal(row,data['row'])
            np.testing.assert_array_equal(y,data['y'])
            fold,base,ref = data['fold'],data['base'],data['ref']
        bank = {}
        for f in np.unique(fold):
            q = np.flatnonzero(fold==f)
            t = np.flatnonzero((fold!=f)&~np.isin(groups,groups[q]))
            assert not np.intersect1d(groups[t],groups[q]).size
            for k,p in run_fold(frame,groups,y,t,q).items():
                bank.setdefault(k,np.full(len(y),np.nan))[q] = p
        hits = {'reference':ref==y}
        for k,p in bank.items():
            assert np.isfinite(p).all()
            hits[k] = (p>=.5)==y
            hits['quarter_'+k] = np.where(ref==(base>=.5),.75*base+.25*p>=.5,ref)==y
        plans[seed] = {k:paired(v,ref==y) for k,v in hits.items()}
        if seed!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        np.savez_compressed(directory/f'{seed}.npz',row=row,y=y,fold=fold,base=base,ref=ref,**bank)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in plans[seed].items()},flush=True)
    baseline = np.stack(history['reference'])
    stable = {}
    for k,h in history.items():
        h = np.stack(h)
        stable[k] = dict(hard_stable=int((hard & h.all(0)).sum()),hard_ever=int((hard & h.any(0)).sum()),
                         common_stable=int((wrong & h.all(0)).sum()),stable_harms=int((right & ~h.any(0)).sum()),
                         gain95_pp=interval((h.astype(float)-baseline).mean(0),groups))
    write(REPORT/'exemplar_features.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
