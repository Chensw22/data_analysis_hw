"""Detect missed positives by contrasting service use with companions in the same booking."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from kaggle_workflow import sha256
from threadpoolctl import threadpool_limits

from awake_negative_recovery import ART, ROOT, BILLS, matrices
from confirm_sleeper_scope import confirm
from exact_fold_error_selector import cached, counts
from exaone_validation_data import compact_features
from export_nov21_error_batch import test_probabilities
import export_consensus_routes as route
import relational_error_corrector as ref
from route_consensus_errors import FAMILIES
from stable_consensus_selector import persistent_mask


def family_contrasts(roster):
    x = compact_features(roster)
    family = roster.PassengerId.str.split('_').str[0]
    size = family.map(family.value_counts()).to_numpy()
    complete = x[BILLS].notna().all(axis=1).to_numpy()
    cost = x[BILLS]
    adverse = cost[['RoomService','Spa','VRDeck']].sum(axis=1,min_count=3)
    favorable = cost[['FoodCourt','ShoppingMall']].sum(axis=1,min_count=2)
    svc = np.column_stack([cost, adverse, favorable, cost.sum(axis=1,min_count=5)])
    detail = pd.DataFrame(np.column_stack([np.log1p(svc),svc>0,x.CryoSleep.eq('True'),x.Age.lt(13),
                                          x.Age, complete, x.CabinNumber]))
    num = detail.groupby(family).transform('count').to_numpy() - detail.notna().to_numpy(int)
    total = detail.groupby(family).transform('sum').to_numpy() - detail.fillna(0).to_numpy()
    mean_other = total / np.where(num>0,num,np.nan)
    raw = detail.to_numpy()
    # Model which occupants differ from their companions, separating booking size from observed bills.
    profile, _ = matrices(x)
    fixed = np.column_stack([profile, svc, x.CabinNumber, size])
    full = np.column_stack([fixed, mean_other, raw-mean_other])
    # Counterfactual holds the query passenger fixed, replacing companions with an
    # unrelated passenger's companions matched by party size and home planet.
    rng = np.random.default_rng(2111)
    group = family.to_numpy()
    substitutes = np.tile(np.arange(len(roster)), (5, 1))
    cells = pd.DataFrame(dict(size=size,planet=x.HomePlanet)).groupby(['size','planet']).indices
    for cell in cells.values():
        for i in cell[size[cell]>1]:
            options = cell[group[cell]!=group[i]]
            if len(options):
                substitutes[:,i] = rng.choice(options,5)
    controls = [np.column_stack([fixed,mean_other[peer],raw-mean_other[peer]]) for peer in substitutes]
    return full, controls, size, x


def risk(features, controls, y, train, query, size, awake):
    rows = train[awake[train] & (size[train]>1)]
    model = CatBoostClassifier(iterations=400,depth=4,learning_rate=.035,l2_leaf_reg=8,
        verbose=False,allow_writing_files=False,thread_count=2,random_seed=2111)
    model.fit(features[rows],y[rows])
    return np.column_stack([model.predict_proba(features[query])[:,1],
        np.mean([model.predict_proba(c[query])[:,1] for c in controls],axis=0)])


def decide(base,p,h,r,active):
    majority_negative = (h<.5).sum(1)>=3
    target = ~base & (p<.5) & active & majority_negative
    real, placebo = r.T
    return {'family_contrast':base | (target & (p+real>=1)),
            'family_consensus':base | (target & (h<.5).all(1) & (real>=.60)),
            'peer_attributed':base | (target & (p+real>=1) & (p+placebo<1))}


def run():
    inp,_,fold,bases,_=ref.overlap.load_authenticated()
    dev,y=inp['dev'],inp['y'].astype(bool)
    train=pd.read_csv(ROOT/'data/raw/train.csv',usecols=lambda c:c!='Name')
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c not in ['Name','Transported'])
    roster=pd.concat([train.drop(columns='Transported'),test],ignore_index=True)
    features,controls,size,x=family_contrasts(roster)
    awake=x.CryoSleep.ne('True').to_numpy()
    active=awake & (size>1)
    p=cached('v108_exact_oof/oof.npz','v108',dev,fold)
    base=confirm(p>=.5,cached('sleeper_consensus_scope.npz','original',dev,fold),
        cached('sleeper_manifest_risk.npz','manifest_berth_party_a',dev,fold))
    est=np.empty((len(y),2))
    for k in np.unique(fold):
        q=np.flatnonzero(fold==k)
        t=np.flatnonzero((fold!=k) & ~np.isin(inp['groups'],inp['groups'][q]))
        est[q]=risk(features[dev],[c[dev] for c in controls],y,t,q,size[dev],awake[dev])
    hard=pd.read_csv(ART/'error_rescue_20261117/catalog_rows.csv')
    np.testing.assert_array_equal(hard.row_id,dev)
    hard=hard.all_recipes_still_wrong.to_numpy(bool)
    persistent=persistent_mask(inp,bases,hard)
    h=np.column_stack([bases[k] for k in FAMILIES])
    decisions=decide(base,p,h,est,active[dev])
    result={k:counts(d,base,y,fold,hard,persistent) for k,d in decisions.items()}
    out=ART/'nov21_party_negative'
    out.mkdir(exist_ok=True)
    pd.DataFrame(dict(row=dev,target=y,v128=base,fold=fold,hard=hard,**decisions)).to_csv(out/'oof.csv',index=False)
    q=np.arange(len(train),len(roster))
    est=risk(features,controls,train.Transported.to_numpy(bool),np.arange(len(train)),q,size,awake)
    ids=test.PassengerId.to_numpy(str)
    base,_=route.official_decisions('proposal_v128_manifest_confirmation',ids)
    p=test_probabilities(route.BASE,ids)
    h=np.column_stack([test_probabilities(k,ids) for k in route.HISTORICAL])
    releases = {'family_consensus':'v131','family_contrast':'v132'}
    for k,d in decide(base,p,h,est,active[q]).items():
        pd.DataFrame(dict(PassengerId=ids,Transported=d)).to_csv(out/(k+'.csv'),index=False)
        result[k]['test_changes']=int((d!=base).sum())
        if k in releases:
            path = out/(releases[k]+'.csv')
            path.write_bytes((out/(k+'.csv')).read_bytes())
            recovered = pd.read_csv(path)
            np.testing.assert_array_equal(recovered.PassengerId, ids)
            np.testing.assert_array_equal(recovered.Transported, d)
            result[k].update(version=releases[k], sha256=sha256(path))
    ref.write(ROOT/'reports/nov21_party_negative/result.json',dict(results=result,scope='Party size > 1, away from recorded cryosleep. Held-out groups excluded from expert fits.'))
    print(result)


if __name__=='__main__':
    with threadpool_limits(2):
        run()
