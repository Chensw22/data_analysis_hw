"""Predict both members' outcomes as a four-state distribution, marginalizing unknown peer outcomes."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
BILLS = ['RoomService', 'Spa', 'VRDeck', 'FoodCourt', 'ShoppingMall']


def attributes(raw):
    cabin = raw.Cabin.str.split('/',expand=True)
    group = raw.PassengerId.str.split('_').str[0]
    nominal = pd.concat([raw[['HomePlanet','CryoSleep','Destination','VIP']].astype('string'),
                         cabin[[0,2]].set_axis(['Deck','Side'],axis=1)],axis=1).fillna('missing')
    spend = raw[BILLS]
    service = spend[BILLS[:3]].sum(axis=1,min_count=3)
    retail = spend[BILLS[3:]].sum(axis=1,min_count=2)
    total = spend.sum(axis=1,min_count=5)
    continuous = pd.DataFrame(dict(age=raw.Age, cabin=pd.to_numeric(cabin[1]), size=group.map(group.value_counts()),
                                   service=np.log1p(service),retail=np.log1p(retail),total=np.log1p(total),
                                   share=(retail+1)/(total+2),missing=spend.isna().sum(axis=1)))
    frame = pd.concat([continuous,np.log1p(spend),pd.get_dummies(nominal,dtype=float)],axis=1)
    # Final sentinel indexes the absent companion without reading another passenger's values.
    return np.vstack([frame.to_numpy(float), np.full((1,len(frame.columns)),np.nan)]),group.to_numpy()


def edges(focal, pool, groups):
    members = pd.Series(pool).groupby(groups[pool]).agg(list).to_dict()
    left,right,weight=[],[],[]
    for i in focal:
        companions=[j for j in members[groups[i]] if j!=i] or [-1]
        left.extend([i]*len(companions))
        right.extend(companions)
        weight.extend([1/len(companions)]*len(companions))
    return np.array(left),np.array(right),np.array(weight)


def pair_matrix(x,left,right):
    a,b = x[left],x[right]
    return np.column_stack([a,b,a[:,:7]-b[:,:7],right>=0])


def fold_train(x,groups,t,q,y,loss,seed):
    left,right,w = edges(t,t,groups)
    # A passenger without a labeled companion contributes a uniform soft target for
    # the unobserved second outcome. Its marginal contribution to the first label is 1.
    no_peer=right<0
    known=~no_peer
    u=np.concatenate([np.flatnonzero(known),np.flatnonzero(no_peer),np.flatnonzero(no_peer)])
    second=np.concatenate([y[right[known]], np.zeros(no_peer.sum(),int), np.ones(no_peer.sum(),int)])
    yy=np.column_stack([y[left[u]],second])
    w[no_peer] *= .5
    model=CatBoostClassifier(loss_function=loss,iterations=430,depth=5,learning_rate=.055,
                             l2_leaf_reg=6,random_seed=seed,verbose=False,thread_count=2,allow_writing_files=False)
    model.fit(pair_matrix(x,left[u],right[u]),yy if loss=='MultiLogloss' else 2*yy[:,0]+yy[:,1],
              sample_weight=w[u])
    left,right,w=edges(q,np.arange(len(groups)),groups)
    p=model.predict_proba(pair_matrix(x,left,right))
    first=p[:,0] if loss=='MultiLogloss' else p[:,2]+p[:,3]
    real=right>=0
    r=model.predict_proba(pair_matrix(x,right[real],left[real]))
    # Covariance under exchanging named roles: second marginal in reversed order.
    first[real]=.5*(first[real]+(r[:,1] if loss=='MultiLogloss' else r[:,1]+r[:,3]))
    return np.bincount(left,weights=w*first,minlength=len(groups))[q]


def run():
    raw=pd.concat([pd.read_csv(ROOT/'data/raw'/f,usecols=lambda c:c not in ['Transported','Name'])
                   for f in ['train.csv','test.csv']],ignore_index=True)
    x,groups=attributes(raw)
    cache=pd.read_csv(ART/'nov21_case_checks/v136_final_oof.csv')
    rows,fold,labels=[cache[k].to_numpy() for k in ['row','fold','true']]
    assert len(rows)==6954
    np.testing.assert_array_equal(raw.PassengerId.to_numpy()[rows],cache.PassengerId)
    with np.load(ART/'v108_exact_oof/oof.npz') as z:
        np.testing.assert_array_equal(z['dev_ids'],rows)
        np.testing.assert_array_equal(z['fold_id'],fold)
    y=np.full(len(raw),-1)
    y[rows]=labels
    anchor,hard=cache.v131.to_numpy(),cache.hard.to_numpy()
    out,metrics={},{}
    for key,loss in [('independent','MultiLogloss'),('joint','MultiClass')]:
        p=np.full(len(cache),np.nan)
        for k in range(3):
            q=rows[fold==k]
            t=rows[(fold!=k)&~np.isin(groups[rows],groups[q])]
            assert (y[t]>=0).all() and not np.intersect1d(groups[t],groups[q]).size
            p[fold==k] = fold_train(x,groups,t,q,y,loss,20261123+k)
            print(key,k,len(t),flush=True)
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        out[key]=p
        correct=(p>=.5)==labels
        fix=correct&(labels!=anchor)
        harm=~correct&(labels==anchor)
        metrics[key]=dict(accuracy=float(correct.mean()), fixed=int(fix.sum()),harmed=int(harm.sum()),
            hard_fixed=int((fix&hard).sum()),hard_harmed=int((harm&hard).sum()),
            fold_net=[int(fix[fold==k].sum())-int(harm[fold==k].sum()) for k in range(3)])
        print(metrics[key],flush=True)
    folder=ART/'nov23_joint_booking'
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'oof.npz',row=rows,fold=fold,y=labels,p=out['joint'],**out)
    directory=ROOT/'reports/nov23_joint_booking'
    directory.mkdir(exist_ok=True)
    plus=(out['joint']>=.5)==labels
    minus=(out['independent']>=.5)==labels
    review=dict(metrics=metrics,labels_used=len(cache),hard_joint_exclusive=int((plus&~minus&hard&(anchor!=labels)).sum()),
                joint_only=int((plus&~minus&(anchor!=labels)).sum()),hard_total=int(hard.sum()),
                protocol='Original three folds, excluding all validation bookings from training. Every eligible training pair is represented; synthetic missing-companion labels split weight evenly. Predictions integrate over all manifest companions, no peer outcome used at inference. Independent outputs use MultiLogloss; joint uses 4-class cross entropy. Each passenger gets total fitting weight one.')
    (directory/'result.json').write_text(json.dumps(review,indent=2)+'\n')
    print(review,flush=True)


if __name__=='__main__':
    run()
