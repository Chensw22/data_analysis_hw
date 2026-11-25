"""Measure which independent bookings surround a cabin using the entire unlabeled roster."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART, DEST
from exaone_validation_data import ROOT, compact_features
from family_identity_transfer_n26 import fit
from solo_archive_recovery_n26 import archive, ticker
from solo_selection_likelihood_n26 import fit_model, regimes

FEES = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


def compose(raw):
    groups = raw.PassengerId.str.split('_').str[0]
    size = groups.map(groups.value_counts()).to_numpy()
    cabin = raw.Cabin.str.split('/', expand=True)
    number = pd.to_numeric(cabin[1]).to_numpy()
    lane = cabin[0] + cabin[2]
    fees = raw[FEES]
    personal = pd.DataFrame({
        'asleep': raw.CryoSleep.map({True: 1., False: 0.}),
        'party_size': np.log(size), 'child': raw.Age.lt(13).where(raw.Age.notna()),
        'young_adult': raw.Age.between(18,29).where(raw.Age.notna()),
        'earth': raw.HomePlanet.eq('Earth').where(raw.HomePlanet.notna()),
        'europa': raw.HomePlanet.eq('Europa').where(raw.HomePlanet.notna()),
        'paid': fees.gt(0).any(axis=1),
        'budget': np.log1p(fees.sum(axis=1,min_count=1))/6.,
    })
    for name in FEES:
        personal[name] = fees[name].gt(0).where(fees[name].notna())
    # Each neighbor casts one vote, even when several members or cabins appear in the lane.
    local_profile = personal.astype(float).groupby(groups).transform('mean').to_numpy()
    group = groups.to_numpy()
    near = np.full((len(raw),2,personal.shape[1]),np.nan)
    shifted = np.full_like(near,np.nan)
    distances = np.full((len(raw),2),np.nan)
    for key in lane.dropna().unique():
        ids = np.flatnonzero((lane==key).to_numpy() & np.isfinite(number))
        places = np.unique(number[ids])
        if len(places)<3:
            continue
        for position in places:
            # The control retains the deck and side but is displaced by half the occupied cabins.
            fake = places[(np.searchsorted(places,position)+len(places)//2)%len(places)]
            who = ids[number[ids]==position]
            for point,array in [(position,near),(fake,shifted)]:
                dist = np.abs(number[ids]-point)
                order = np.lexsort((ids,dist))
                for booking in np.unique(group[who]):
                    members = who[group[who]==booking]
                    ranked = ids[order][group[ids[order]]!=booking]
                    _,first = np.unique(group[ranked],return_index=True)
                    neighbors = ranked[np.sort(first)[:24]]
                    assert booking not in group[neighbors]
                    for j,k in enumerate([4,24]):
                        subset = neighbors[:k]
                        if len(subset):
                            values = local_profile[subset]
                            count = np.isfinite(values).sum(0)
                            array[members,j] = np.divide(np.nansum(values,axis=0),count,
                                out=np.full(values.shape[1],np.nan),where=count>0)
                            if array is near:
                                distances[members,j] = np.max(np.abs(number[subset]-position))
    base = compact_features(raw).assign(log_group_size=np.log(size), radius4=distances[:,0], radius24=distances[:,1])
    out = {'plain':base}
    for name,array in [('near',near),('shifted',shifted)]:
        extra = {f'{field}_{k}':array[:,j,i] for i,field in enumerate(personal.columns) for j,k in enumerate([4,24])}
        out[name] = pd.concat([base,pd.DataFrame(extra)],axis=1)
    diagnostic = {}
    for j,field in enumerate(personal.columns):
        val = personal[field].to_numpy(float)
        mask = np.isfinite(val)&np.isfinite(near[:,0,j])&np.isfinite(shifted[:,0,j])
        diagnostic[field] = dict(rows=int(mask.sum()),near_mse=float(np.mean((val[mask]-near[mask,0,j])**2)),
                                 shifted_mse=float(np.mean((val[mask]-shifted[mask,0,j])**2)))
    # Agreeing with one's neighbors is different from merely occupying a particular lane.
    # Center the signed products on the lane background to control its composition.
    center=personal.astype(float).groupby(lane).transform('mean').to_numpy()
    deviation=np.nan_to_num(personal.to_numpy(float)-center)
    interactions={}
    for name,array in [('near',near),('shifted',shifted)]:
        contrasts=np.nan_to_num(array-center[:,None,:])
        interactions[name]=np.column_stack([contrasts[:,1],contrasts[:,0]-contrasts[:,1],
                                              deviation*contrasts[:,0]])
    return out,diagnostic,interactions


def run():
    raw,y,g,views=regimes()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    x,diagnostic,interactions=compose(pd.concat([raw,test],ignore_index=True))
    path=ART/'nov26_neighbor_booking_composition'
    path.mkdir(exist_ok=True)
    results={'composition':diagnostic}
    historical_coefficients={}
    for tag,ids,fold,native,view in views:
        scores={name:np.empty(len(ids)) for name in x}
        fresh=~np.isin(g[ids],g[views[0][1]])
        for k in np.unique(fold):
            loc=fold==k
            q=ids[loc]
            other=ids[~loc]
            t=other[~np.isin(g[other],g[q])]
            assert not np.isin(g[q],g[t]).any()
            if tag=='historical':
                fresh[loc]=~np.isin(g[q],g[other])
            out=path/f'{tag}_{k}.npz'
            if not out.exists():
                estimates={name:fit_model(frame.iloc[t],y[t]).predict_proba(frame.iloc[q])[:,1] for name,frame in x.items()}
                with out.with_suffix('.tmp').open('wb') as h:
                    np.savez_compressed(h,t=t,q=q,labels=y[t],**estimates)
                out.with_suffix('.tmp').replace(out)
            with np.load(out) as z:
                for name,v in [('q',q),('t',t),('labels',y[t])]:
                    np.testing.assert_array_equal(z[name],v)
                for name in x:
                    scores[name][loc]=z[name]
        anchor=view['anchor']
        known=view['known']
        _,ballots=archive(tag,ids,fold,g,y,ART/'nov24_remaining_cases/archive_scores.npz')
        common=view['eligible']&(view['votes']==anchor[:,None]).all(1)&(ballots==anchor[:,None]).all(1)
        truth=y[ids]
        if tag=='full':
            assert int(((anchor!=truth)&common).sum())==245
        audit={'losses':{n:log_loss(truth,p) for n,p in scores.items()}}
        scores_odds={k:logit(p.clip(.002,.998)) for k,p in scores.items()}
        native_odds=logit(native.clip(.002,.998))
        for name in ('near','shifted'):
            direction=scores[name]>=.5
            changes={'all':direction!=anchor,'induced':(direction!=anchor)&((scores['plain']>=.5)==anchor)}
            step=scores_odds[name]-scores_odds['plain']
            for w in (.5,1.):
                changes[f'offset_{w}']=((native>=.5)==anchor)&((native_odds+w*step>=0)!=anchor)
            for action,flip in changes.items():
                audit[name+'_'+action]={k:ticker(scope,truth!=anchor,flip) for k,scope in
                    [('all',known),('common',common),('fresh',fresh&known),('fresh_common',fresh&common)]}
        designs={}
        ordinary=np.column_stack([np.ones(len(ids)),native_odds,scores_odds['plain']-native_odds])
        designs['place']=np.column_stack([ordinary,scores_odds['shifted']-scores_odds['plain']])
        designs['real_place']=np.column_stack([designs['place'],scores_odds['near']-scores_odds['shifted']])
        designs['shifted_profile']=np.column_stack([ordinary,interactions['shifted'][ids]])
        designs['real_profile']=np.column_stack([ordinary,interactions['near'][ids]])
        for name,features in designs.items():
            score=np.zeros(len(ids))
            for k in np.unique(fold):
                pool=np.flatnonzero(known&(fold==k))
                for train,query in GroupKFold(3,shuffle=True,random_state=6257).split(pool,groups=g[ids[pool]]):
                    t,q=pool[train],pool[query]
                    assert not np.isin(g[ids[t]],g[ids[q]]).any()
                    score[q]=native_odds[q]+features[q]@fit(features[t],native_odds[t],truth[t])
            local=(score>=0)!=anchor
            audit['crossfit_'+name]={k:ticker(scope,truth!=anchor,local) for k,scope in
                    [('all',known),('common',common),('fresh',fresh&known)]}
            if tag=='historical':
                historical_coefficients[name]=fit(features[known],native_odds[known],truth[known])
            else:
                p=expit(native_odds+features@historical_coefficients[name])
                audit['from_history_'+name]={k:ticker(scope&fresh,truth!=anchor,(p>=.5)!=anchor)
                    for k,scope in [('all',known),('common',common)]}
        results[tag]=audit
        print(tag,audit['losses'],[(n,v['all']['fixed'],v['all']['harmed'],v['common']['fixed'],v['common']['harmed']) for n,v in audit.items() if n!='losses'],flush=True)
    DEST.with_name('neighbor_booking_composition.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
