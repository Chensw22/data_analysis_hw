"""Measure the feasible allocation tracks after deleting one booking from the full passenger manifest."""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from disjoint_tree_fields_n25 import ART, DEST
from exaone_validation_data import CATEGORICAL, compact_features
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import ROOT, data
from outstanding_error_families_n26 import observed


def choice_set(frame, return_intervals=False):
    booking = frame.PassengerId.str[:4].astype(int).to_numpy()
    cabin = frame.Cabin.str.split('/', expand=True)
    track = (cabin[0]+cabin[2]).fillna('missing').to_numpy()
    number = pd.to_numeric(cabin[1]).to_numpy()
    features, info, supported = {}, {}, []
    lows, highs = [], []
    for name in sorted(set(track)-{'missing'}):
        mask = track == name
        reservations = pd.DataFrame(dict(group=booking[mask], num=number[mask])).drop_duplicates().sort_values('group')
        group = reservations.group.to_numpy()
        n = reservations.num.to_numpy()
        assert (np.diff(group)>0).all() and (np.diff(n)>0).all()
        left = np.searchsorted(group, booking, side='left')-1
        right = np.searchsorted(group, booking, side='right')
        low = np.where(left>=0,n[left]+1,0.)
        upper = np.where(right<len(group),n[right.clip(max=len(group)-1)]-1,np.inf)
        assert ((low[mask]<=number[mask])&(number[mask]<=upper[mask])).all()
        lows.append(low)
        highs.append(upper)
        available = low<=upper
        supported.append(available)
        # The last observed cabin only provides a lower bound for later bookings.
        features['capacity_'+name] = np.minimum(np.maximum(upper-low+1,0),4)
        features['open_'+name] = np.isinf(upper).astype(int)
        info[name] = dict(distinct_reservations=len(group), internal_gaps=int((np.diff(n)-1).sum()))
    if return_intervals:
        return np.array(sorted(set(track)-{'missing'})), np.column_stack(lows), np.column_stack(highs)
    support = np.stack(supported, axis=1)
    count = support.sum(axis=1)
    features['track_choices'] = count
    missing = track=='missing'
    fact = {'tracks':info,'individuals':len(track),'unknown':int(missing.sum()),
            'observed_track_excluded':0, 'missing_rows_with_one_choice':int((missing&(count==1)).sum()),
            'known_rows_with_one_choice_when_hidden':int((~missing&(count==1)).sum()),
            'missing_choice_histogram':pd.Series(count[missing]).value_counts().sort_index().to_dict()}
    actual = pd.DataFrame(features)
    rng = np.random.default_rng(20671)
    shuffled = actual.copy()
    # Permute each row's full choice set within its observed track;
    # for missing cabins all passengers belong to the same missing stratum.
    for name in np.unique(track):
        ix = np.flatnonzero(track==name)
        shuffled.iloc[ix] = actual.iloc[rng.permutation(ix)].to_numpy()
    base = compact_features(frame)
    filled = base.copy()
    keys = np.array(sorted(set(track)-{'missing'}))
    single = missing & (count==1)
    forced = keys[support[single].argmax(1)]
    filled.loc[single,'Deck']=[v[0] for v in forced]
    filled.loc[single,'Side']=[v[1] for v in forced]
    return fact, {'plain':base,'filled':filled,'real':pd.concat([filled,actual],axis=1),
                  'shuffled':pd.concat([filled,shuffled],axis=1)}


def run():
    raw,y,g,f=data()
    _,_,_,rows,names,p,h,native,old_native,r=panels_and_reference()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf, old, parent=z['fold'],z['nine'],z['v131']
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],r[0])
        r[0]=z['prior']
    awake=raw.CryoSleep.eq(False).to_numpy()
    full=observed(np.arange(len(y)),f,p,names,native,*r,awake,'full')
    hist=observed(rows,hf,h,names,old_native,parent,old,old,awake,'historical')
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=list(raw))
    geom,frames=choice_set(pd.concat([raw,test],ignore_index=True))
    print('geometry',geom,flush=True)
    counts=frames['real'].track_choices.to_numpy()
    missing=frames['plain'].Deck.eq('__MISSING__').to_numpy()
    for part, sl in [('train',slice(0,len(raw))),('competition',slice(len(raw),None))]:
        geom[part]=dict(missing=int(missing[sl].sum()),single=int((missing[sl]&(counts[sl]==1)).sum()))
    report={'geometry':geom}
    cache=ART/'nov26_booking_cabin_choice_set'
    cache.mkdir(exist_ok=True)
    predictions={}
    for tag,idx,fold,view,base in [('historical',rows,hf,hist,old_native),('full',np.arange(len(y)),f,full,native)]:
        fresh=~np.isin(g[idx],g[rows]) if tag=='full' else np.zeros(len(idx),bool)
        scores={name:np.zeros(len(idx)) for name in frames}
        for k in np.unique(fold):
            q=idx[fold==k]
            train=idx[(fold!=k)&~np.isin(g[idx],g[q])]
            assert not np.isin(g[q],g[train]).any()
            if tag=='historical':
                fresh[fold==k]=~np.isin(g[q],g[idx[fold!=k]])
            for name,x in frames.items():
                model=CatBoostClassifier(iterations=360,depth=5,learning_rate=.055,l2_leaf_reg=5,
                    random_seed=2381,thread_count=2,verbose=False,allow_writing_files=False)
                model.fit(x.iloc[train],y[train],cat_features=CATEGORICAL)
                scores[name][fold==k]=model.predict_proba(x.iloc[q])[:,1]
            print(tag,k,flush=True)
        np.savez_compressed(cache/f'{tag}.npz',row=idx,fold=fold,y=y[idx],passenger=raw.PassengerId.iloc[idx].to_numpy(str),**scores)
        anchor=view['anchor'];known=view['known']
        hard=view['eligible']&(view['votes']==anchor[:,None]).all(1)
        predictions[tag]={}
        results={}
        for name,v in scores.items():
            direct=v>=.5
            methods={'direct':direct, 'quarter':np.where((base>=.5)==anchor,.75*base+.25*v>=.5,anchor)}
            if name!='plain':
                methods['changes_vs_plain']=np.where(direct!=(scores['plain']>=.5),direct,anchor)
            for scheme,dec in methods.items():
                key=name+'_'+scheme
                predictions[tag][key]=dec
                def effect(mask,other=anchor):
                    return dict(rows=int(mask.sum()),errors=int(((other!=y[idx])&mask).sum()),
                                fixed=int(((other!=y[idx])&(dec==y[idx])&mask).sum()),
                                harmed=int(((other==y[idx])&(dec!=y[idx])&mask).sum()))
                results[key]={scope:effect(mask) for scope,mask in [('all',known),('new',known&fresh),
                    ('families',hard),('families_new',hard&fresh),('missing_cabin',known&missing[idx]),('missing_one',known&missing[idx]&(counts[idx]==1)),
                    ('missing_many',known&missing[idx]&(counts[idx]>1)),('recorded_one',known&~missing[idx]&(counts[idx]==1)),
                    ('recorded_many',known&~missing[idx]&(counts[idx]>1))]}
                if scheme=='direct':
                    results[key]['vs_plain']=effect(np.ones(len(idx),bool),scores['plain']>=.5)
        report[tag]=results
    stable={}
    hi,fu=hist['anchor'],full['anchor'][rows]
    eligible=hist['known']&full['known'][rows]&(hi==fu)
    for name,hdec in predictions['historical'].items():
        fdec=predictions['full'][name][rows]
        agree=(hdec==fdec)&(hdec!=hi)&eligible
        stable[name]=dict(same_anchor_errors=int(((hi!=y[rows])&eligible).sum()),
                          repeated_fixed=int((agree&(hi!=y[rows])).sum()),repeated_harmed=int((agree&(hi==y[rows])).sum()))
    report['paired']=stable
    DEST.with_name('booking_cabin_choice_set.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['full']['real_quarter'],stable['real_quarter'])

if __name__=='__main__':
    run()
