"""Recover archived minority-vote bands from deployable scores without using any outcome to fit the gate."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from adjudicate_amount_dissent_n25 import counts
from disjoint_tree_fields_n25 import ART,DEST
from family_identity_transfer_n26 import panels_and_reference
from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import data,FAMILIES
from full_roster_conditional_pool_n23 import banks


def deployable(raw,tree,nine):
    # Encode query-observable fields alone; keep missing and zero expenses distinct.
    cabin=raw.Cabin.str.split('/',expand=True)
    a=raw[['HomePlanet','Destination','CryoSleep','VIP']].copy()
    a['deck'],a['side']=cabin[0],cabin[2]
    hot=pd.get_dummies(a.astype('string').fillna('?'),dtype=float)
    fees=raw[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]
    age=raw.Age.fillna(-1).to_numpy()/40
    num=pd.to_numeric(cabin[1]).fillna(-100).to_numpy()/500
    group=raw.PassengerId.str.split('_').str[0]
    size=group.map(group.value_counts()).to_numpy()
    detail=np.column_stack([hot,age,num,np.log1p(size),fees.isna(),fees.gt(0),np.log1p(fees.fillna(0))/5])
    opinions=np.column_stack([tree,nine])
    transformed=logit(opinions.clip(.005,.995))
    return np.column_stack([transformed,opinions>=.5,detail])


def estimates(x,archive,train,query,copy_cols):
    assert not np.intersect1d(train,query).size
    target=logit(archive[train].clip(.005,.995))
    source=x[train]
    mu,sigma=source.mean(0),source.std(0).clip(.15,None)
    norm=(source-mu)/sigma
    linear=Ridge(alpha=22).fit(norm,target)
    residual=target-linear.predict(norm)
    # Learn remaining nonlinear deviations jointly. Shared trees give the same neighborhoods for every voter.
    extra=ExtraTreesRegressor(n_estimators=48,min_samples_leaf=6,max_features=.8,n_jobs=2,random_state=279)
    extra.fit(source,residual)
    initial=linear.predict((x[query]-mu)/sigma)
    corrected=initial+extra.predict(x[query])
    outputs={}
    for tag,values in [('linear',initial),('nonlinear',corrected)]:
        hard=values>=0
        hard[:,copy_cols]=archive[query][:,copy_cols]>=.5
        n=hard.sum(1)
        outputs[tag]=np.minimum(n,68-n)
    return outputs


def run():
    y,g,f,rows,names,p,h,native,old_native,reference=panels_and_reference()
    raw,_,_,_=data()
    state,_,full=banks(raw,y,f)
    nine=full[:,[state.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]]
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        old_nine=z['scores']; old_champion=z['nine'].astype(bool); hf=z['fold']
    np.testing.assert_array_equal(nine[:,4],native)
    np.testing.assert_array_equal(old_nine[:,4],old_native)
    copy_cols=np.array([i for i,name in enumerate(names) if name.startswith('permutation_selection/')])
    assert len(copy_cols)==5
    archive=np.vstack([p,h])
    # Build registration features on the same full unlabeled roster in both fits.
    historic_tree=np.zeros_like(p[:,copy_cols]);historic_tree[rows]=h[:,copy_cols]
    historic_scores=np.zeros_like(nine);historic_scores[rows]=old_nine
    x=np.vstack([deployable(raw,p[:,copy_cols],nine),deployable(raw,historic_tree,historic_scores)[rows]])
    v=(archive>=.5).sum(1); minority=np.minimum(v,68-v)
    history_idx=len(y)+np.arange(len(rows))
    original_idx=np.r_[np.arange(len(y)),rows]
    labels=np.r_[y,y[rows]]
    tree=archive[:,copy_cols].mean(1)>=.5
    ref=np.r_[np.where(reference[0]>=0,np.where(reference[0].astype(bool),reference[2],reference[1]),-1),old_champion]
    known=ref>=0
    unseen=~np.isin(g,g[rows])
    protocols={'full':[],'historical':[],'old_to_new':[(history_idx,np.flatnonzero(unseen))],'same_model_new':[]}
    for k in np.unique(f):
        block=np.flatnonzero(f==k)
        for a,b in GroupKFold(4,shuffle=True,random_state=2621).split(block,groups=g[block]):
            protocols['full'].append((block[a],block[b]))
        protocols['same_model_new'].append((block[~unseen[block]],block[unseen[block]]))
    for k in np.unique(hf):
        block=history_idx[hf==k]
        for a,b in GroupKFold(4,shuffle=True,random_state=2621).split(block,groups=g[original_idx[block]]):
            protocols['historical'].append((block[a],block[b]))
    report={}
    for tag,parts in protocols.items():
        bank={}
        for train,query in parts:
            assert not np.isin(g[original_idx[train]],g[original_idx[query]]).any()
            preds=estimates(x,archive,train,query,copy_cols)
            band={n:(a>=4)&(a<=16) for n,a in preds.items()}
            band.update(both=band['linear']&band['nonlinear'],exact=(minority[query]>=4)&(minority[query]<=16))
            for name,eligible in band.items():
                bank.setdefault(name,[]).append((query,eligible))
        report[tag]={}
        expected=(history_idx if tag=='historical' else np.arange(len(y)) if tag=='full' else np.flatnonzero(unseen))
        for name,outputs in bank.items():
            ix,gate=(np.concatenate([e[i] for e in outputs]) for i in range(2))
            np.testing.assert_array_equal(np.sort(ix),expected)
            valid=known[ix]
            proposed=tree[ix]
            delta=np.where(gate&valid,(proposed==labels[ix]).astype(int)-(ref[ix]==labels[ix]),0)
            truth=(minority[ix]>=4)&(minority[ix]<=16)
            changed=gate&valid&(proposed!=ref[ix])
            fold_ids=hf[ix-len(y)] if tag=='historical' else f[ix]
            r=counts(delta,valid,fold_ids)
            r.update(vote_band_matched=int((gate==truth).sum()),gate_rows=len(ix),
                     changed_in_true_band=int((changed&truth).sum()),edits=int(changed.sum()),
                     outside_known=int((gate&~valid).sum()))
            report[tag][name]=r
            print(tag,name,r['fixed'],r['harmed'],r['changed_in_true_band'],r['edits'],flush=True)
    DEST.with_name('distill_dispute_band.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
