"""Test unique global cabin recoveries and masked-cabin predictability on transport outcomes."""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART, DEST
from exaone_validation_data import CATEGORICAL, compact_features
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import ROOT, data
from global_cabin_assignment_n26 import OUT
from outstanding_error_families_n26 import observed


def cabin_views(raw):
    record = json.loads((OUT/'masked_known.json').read_text())+json.loads((OUT/'missing.json').read_text())
    record.sort(key=lambda r:r['row'])
    np.testing.assert_array_equal([r['row'] for r in record],np.arange(len(raw)))
    frame=compact_features(raw)
    g=raw.PassengerId.str[:4]
    masked=frame.drop(columns=['CabinNumber','Side','Deck']).copy()
    masked['party_size']=g.map(g.value_counts())
    masked['registration']=g.astype(int)
    x=pd.get_dummies(masked,dtype=float).fillna(-1).to_numpy(np.float32)
    real_track=frame.Deck+frame.Side
    observed_cabin=raw.Cabin.notna().to_numpy()
    tracks=np.sort(real_track[observed_cabin].unique())
    idx=pd.Index(tracks)
    codes=idx.get_indexer(real_track)
    bins=np.array_split(np.random.default_rng(9983).permutation(np.unique(g)),40)
    prior=np.zeros((len(raw),len(tracks)))
    for batch in bins:
        q=g.isin(batch).to_numpy()
        t=~q&observed_cabin
        trees=ExtraTreesClassifier(n_estimators=64,min_samples_leaf=6,n_jobs=2,random_state=1562).fit(x[t],codes[t])
        prior[np.ix_(q,trees.classes_)]=trees.predict_proba(x[q])
    np.testing.assert_allclose(prior.sum(1),1.)
    filled=frame.copy()
    posterior=np.zeros_like(prior)
    midpoint=np.zeros_like(prior)
    sizes=np.zeros(len(raw))
    for i,r in enumerate(record):
        opts=r['global_options']
        sizes[i]=len(opts)
        for tr,(lo,hi) in opts.items():
            j=idx.get_loc(tr)
            posterior[i,j]=prior[i,j]+.0001
            midpoint[i,j]=lo if hi is None else (lo+hi)/2
        if len(opts)==1 and not observed_cabin[i]:
            tr,(lo,hi)=next(iter(opts.items()))
            filled.loc[i,['Deck','Side']]=[tr[0],tr[1]]
            if hi==lo:
                filled.loc[i,'CabinNumber']=lo
    posterior/=posterior.sum(1,keepdims=True)
    mass=posterior[np.arange(len(raw)),codes]
    mass[~observed_cabin]=np.nan
    geom=pd.DataFrame(posterior,columns=['available_'+name for name in tracks])
    geom['choice_count']=sizes
    geom['observed_surprise']=-np.log(mass.clip(1e-8))
    geom['expected_cabin']=(posterior*midpoint).sum(1)
    geom['choice_entropy']=-(posterior*np.log(posterior.clip(1e-8))).sum(1)
    placebo=geom.copy()
    rng=np.random.default_rng(988)
    for condition in np.unique(real_track):
        ix=np.flatnonzero(real_track==condition)
        placebo.iloc[ix]=geom.iloc[rng.permutation(ix)].to_numpy()
    correct=(posterior.argmax(1)==codes)&observed_cabin
    recovery=dict(hidden=int(observed_cabin.sum()),correct_top=int(correct.sum()),
                  raw_top_without_constraints=int(((prior.argmax(1)==codes)&observed_cabin).sum()),
                  unique=int((sizes[observed_cabin]==1).sum()))
    return dict(plain=frame,filled=filled,geometry=pd.concat([filled,geom],axis=1),
                shuffled=pd.concat([filled,placebo],axis=1)),sizes,observed_cabin,recovery


def run():
    raw,y,g,f=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    views,sizes,had_cabin,recovery=cabin_views(pd.concat([raw,test],ignore_index=True))
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    full=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    report={'track_recovery':recovery}
    print(recovery,flush=True)
    for tag,ids,fold,base,view in [('historical',rows,hf,old_native,hist),('full',np.arange(len(y)),f,native,full)]:
        clean=~np.isin(g[ids],g[rows])
        prob={name:np.zeros(len(ids)) for name in [*views,'query_fill']}
        for k in np.unique(fold):
            part=fold==k
            q=ids[part]
            t=ids[(~part)&~np.isin(g[ids],g[q])]
            assert not np.isin(g[t],g[q]).any()
            if tag=='historical':
                clean[part]=~np.isin(g[q],g[ids[~part]])
            for name,frame in views.items():
                model=CatBoostClassifier(iterations=440,depth=6,learning_rate=.055,l2_leaf_reg=5,
                                         random_seed=9570,thread_count=2,verbose=False,allow_writing_files=False)
                model.fit(frame.iloc[t],y[t],cat_features=CATEGORICAL)
                prob[name][part]=model.predict_proba(frame.iloc[q])[:,1]
                if name=='plain':
                    prob['query_fill'][part]=model.predict_proba(views['filled'].iloc[q])[:,1]
            print(tag,k,flush=True)
        np.savez_compressed(OUT/f'geometry_{tag}.npz',q=ids,fold=fold,**prob)
        anchor=view['anchor']
        common=view['eligible']&(view['votes']==anchor[:,None]).all(1)
        wrong=anchor!=y[ids]
        tables={}
        for name,score in prob.items():
            binary=score>=.5
            plain=prob['plain']>=.5
            for mode,edits in [('direct',binary!=anchor),('quarter',((.75*base+.25*score>=.5)!=anchor)&
                                ((base>=.5)==anchor)),('induced',(binary!=plain)&(binary!=anchor))]:
                counts={}
                for title,mask in [('all',view['known']),('new',view['known']&clean),('families',common),('fresh_families',common&clean),
                        ('missing',view['known']&~had_cabin[ids]),('unique_missing',view['known']&~had_cabin[ids]&(sizes[ids]==1)),
                        ('known',view['known']&had_cabin[ids])]:
                    flagged=edits&mask
                    counts[title]=dict(rows=int(mask.sum()),fixed=int((flagged&wrong).sum()),harmed=int((flagged&~wrong).sum()))
                tables[name+'_'+mode]=counts
        report[tag]=tables
        print(tag,[(key,tab['all']['fixed'],tab['all']['harmed']) for key,tab in tables.items()],flush=True)
    DEST.with_name('global_cabin_assignment_outcome.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
