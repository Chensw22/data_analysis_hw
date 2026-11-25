"""Match an entire held-out passenger roster to labeled prototypes with and without capacity constraints."""
import json

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logit
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

from deployment_booking_mix_n23 import passenger_vectors
from disjoint_tree_fields_n25 import ART, DEST
from family_identity_transfer_n26 import fit, panels_and_reference
from full_roster_booking_validation_n23 import data
from outstanding_error_families_n26 import observed

OUT = ART/'nov26_booking_transport_classifier'


def allocate(x, bank_labels, bank, q, seed):
    centers, masses, classes = [], [], []
    for label in (False, True):
        members = bank[bank_labels == label]
        km = KMeans(n_clusters=56, n_init=1, max_iter=80, random_state=seed).fit(x[members])
        weights = np.bincount(km.labels_, minlength=56)
        present = weights > 0
        centers.extend(km.cluster_centers_[present])
        masses.extend(weights[present])
        classes.extend([label] * present.sum())
    weights = np.asarray(masses, dtype=float) / len(bank)
    labels = np.asarray(classes, dtype=float)
    distances = cdist(x[q], centers, metric='sqeuclidean')
    distances -= distances.min(1, keepdims=True)
    scale = np.median(np.partition(distances, 7, axis=1)[:, 7]).clip(.001)
    kernel = np.exp(-np.minimum(32., 4*distances/scale)) * weights
    prob = kernel / kernel.sum(1, keepdims=True)
    local = prob @ labels
    for step in range(600):
        mass = prob.mean(0)
        if np.abs(mass - weights).sum() < 1e-5:
            break
        prob *= weights / mass
        prob /= prob.sum(1, keepdims=True)
    assert np.abs(prob.mean(0) - weights).sum() < .001
    capacity = prob @ labels
    assert abs(capacity.mean() - bank_labels.mean()) < .001
    return np.column_stack([local,capacity])


def fold_predictions(x,y,g,t,q,seed):
    first,second = next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=seed).split(t,groups=g[t]))
    result = np.zeros((len(q),5))
    for j,(learn,bank) in enumerate([(t[first],t[second]),(t[second],t[first])]):
        assert not np.isin(g[q],g[t]).any() and not np.isin(g[learn],g[bank]).any()
        forest = ExtraTreesClassifier(n_estimators=80,min_samples_leaf=8,max_features=.8,random_state=seed+j,n_jobs=2)
        forest.fit(x[learn],y[learn])
        indices = np.r_[bank,q]
        branches = np.column_stack([tree.predict_proba(x[indices])[:,1] for tree in forest.estimators_])
        result[:,0] += branches[len(bank):].mean(1)
        # Each representation contributes one mean squared distance regardless of dimension.
        raw = x[indices]/np.sqrt(x.shape[1])
        learned = branches/np.sqrt(branches.shape[1])
        for k, features in enumerate([raw, np.column_stack([raw,learned])]):
            result[:,1+2*k:3+2*k] += allocate(features,y[bank],np.arange(len(bank)),
                                               np.arange(len(bank),len(indices)),seed+j)
    return result/2


def run():
    raw,y,g,f = data()
    x = passenger_vectors(raw).astype(np.float32)
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    awake = raw.CryoSleep.eq(False).to_numpy()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    current = observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    history = observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    OUT.mkdir(exist_ok=True)
    columns = ['forward','raw_local','raw_capacity','learned_local','learned_capacity']
    results = {}
    for tag,ids,fold,base,view in [('historical',rows,hf,old_native,history),('full',np.arange(len(y)),f,native,current)]:
        clean = ~np.isin(g[ids],g[rows])
        probs = np.zeros((len(ids),len(columns)))
        flags = {}
        for k in np.unique(fold):
            loc = np.flatnonzero(fold==k)
            q = ids[loc]
            seen = np.isin(g[ids],g[q])
            t = ids[(fold!=k)&~seen]
            if tag=='historical':
                clean[loc] = ~np.isin(g[q],g[ids[fold!=k]])
            file = OUT/f'{tag}_{k}.npz'
            if not file.exists():
                pred = fold_predictions(x,y,g,t,q,4200+int(k))
                with file.with_suffix('.tmp').open('wb') as b:
                    np.savez_compressed(b,p=pred,t=t,q=q,train_y=y[t],passenger=raw.PassengerId.iloc[q].to_numpy(str))
                file.with_suffix('.tmp').replace(file)
            with np.load(file) as z:
                for name,value in [('q',q),('t',t),('train_y',y[t]),('passenger',raw.PassengerId.iloc[q])]:
                    np.testing.assert_array_equal(z[name], value)
                probs[loc] = z['p']
            odds = logit(np.clip(probs[loc],.002,.998))
            offset=logit(base[loc].clip(.002,.998))
            controls = np.column_stack([np.ones(len(loc)),offset,odds[:,0]-offset])
            local = np.column_stack([controls, odds[:,[1,3]]-offset[:,None]])
            designs=dict(forward=controls,local=local,capacity=np.column_stack([local, odds[:,2]-odds[:,1], odds[:,4]-odds[:,3]]))
            for mode in ['all','new']:
                if mode=='new' and tag=='historical':
                    continue
                batches = ([(np.flatnonzero(~clean[loc]),np.flatnonzero(clean[loc]))] if mode=='new' else
                           GroupKFold(4,shuffle=True,random_state=3846).split(loc,groups=g[q]))
                for tr,test in batches:
                    assert not np.isin(g[q[tr]],g[q[test]]).any()
                    for model,design in designs.items():
                        coef=fit(design[tr],offset[tr],y[q[tr]])
                        pred=offset[test]+design[test]@coef>=0
                        flags.setdefault(mode+'_'+model,np.zeros(len(ids),bool))[loc[test]] = pred!=(base[loc[test]]>=.5)
            print(tag,k,len(t),len(q),flush=True)
        for j,name in enumerate(columns):
            flags[name]=((probs[:,j]>=.5)!=(base>=.5))
            flags[name+'_quarter']=((.25*probs[:,j]+.75*base>=.5)!=(base>=.5))
        wrong = view['anchor']!=y[ids]
        same = view['anchor']==(base>=.5)
        common = view['eligible']&(view['votes']==view['anchor'][:,None]).all(1)
        table={}
        for name,flag in flags.items():
            per={}
            for group,mask in [('all',view['known']),('unseen',view['known']&clean),('families',common),('unseen_families',common&clean)]:
                if name.startswith('new_') and not group.startswith('unseen'):
                    continue
                change=flag&same&mask
                per[group]=dict(rows=int(mask.sum()),fixed=int((change&wrong).sum()),harmed=int((change&~wrong).sum()),
                                net_by_fold=[int((change&wrong&(fold==k)).sum())-int((change&~wrong&(fold==k)).sum()) for k in np.unique(fold)])
            table[name]=per
        results[tag]=table
        print([(name,r.get('all',r['unseen'])['fixed'],r.get('all',r['unseen'])['harmed']) for name,r in table.items()],flush=True)
    DEST.with_name('booking_transport_classifier.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
