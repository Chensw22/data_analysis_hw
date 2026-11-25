"""Blindly restore an observed booking's track, then measure transport odds for real missing cabins."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold

from booking_cabin_choice_set_n26 import ART, DEST, ROOT, CATEGORICAL, choice_set, compact_features, data
from family_identity_transfer_n26 import panels_and_reference
from outstanding_error_families_n26 import observed


def model():
    return CatBoostClassifier(iterations=260,depth=5,learning_rate=.06,random_seed=4199,
                              verbose=False,thread_count=2,allow_writing_files=False)


def run():
    raw,y,g,f=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=list(raw))
    roster=pd.concat([raw,test],ignore_index=True)
    groups=roster.PassengerId.str[:4].to_numpy()
    key,low,high=choice_set(roster,True)
    x=compact_features(roster)
    track=(x.Deck+x.Side).to_numpy()
    known=roster.Cabin.notna().to_numpy()
    index=pd.Index(key).get_indexer(track)
    blind=x.drop(columns=['Deck','Side','CabinNumber'])
    categories=[s for s in CATEGORICAL if s in blind]
    prob=np.zeros_like(low)
    for t,q in GroupKFold(4,shuffle=True,random_state=1342).split(x,groups=groups):
        t=t[known[t]]
        m=model().fit(blind.iloc[t],index[t],cat_features=categories)
        prob[np.ix_(q,m.classes_)]=m.predict_proba(blind.iloc[q])
    adjusted=prob*(low<=high)
    assert (adjusted.sum(1)>0).all()
    adjusted/=adjusted.sum(axis=1,keepdims=True)
    check={}
    size=roster.PassengerId.groupby(roster.PassengerId.str[:4]).transform('size').to_numpy()
    for name,mask in [('known',known),('solo',known&(size==1)),('multi',known&(size>1))]:
        check[name]=dict(rows=int(mask.sum()),plain=int((prob.argmax(1)[mask]==index[mask]).sum()),
                         constrained=int((adjusted.argmax(1)[mask]==index[mask]).sum()))
    # Interpolate cabin coordinates within each track from other observed bookings.
    # Search ranges for truly missing cabins can include the same booking's known companions.
    cabin=np.empty_like(prob)
    for j,k in enumerate(key):
        at=np.flatnonzero(known&(track==k))
        order=np.argsort(groups[at]);at=at[order]
        n=np.round(np.interp(groups.astype(int),groups[at].astype(int),x.CabinNumber.iloc[at]))
        cabin[:,j]=np.clip(n,low[:,j],np.maximum(low[:,j],high[:,j]))
    print('blinded reconstruction',check,flush=True)
    _,_,_,rows,names,p,h,native,old_native,r=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        r[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,parent=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    full=observed(np.arange(len(y)),f,p,names,native,*r,awake,'full')
    hist=observed(rows,hf,h,names,old_native,parent,old,old,awake,'historical')
    report={'track_reconstruction':check}
    for tag,ids,fold,view in [('full',np.arange(len(y)),f,full),('historical',rows,hf,hist)]:
        preds={n:np.zeros(len(ids)) for n in ['raw','plain_fill','constrained_fill']}
        for k in np.unique(fold):
            q=(fold==k)&~known[ids]; at=ids[q]
            t=ids[(fold!=k)&~np.isin(g[ids],g[ids[fold==k]])]
            m=model().fit(x.iloc[t],y[t],cat_features=CATEGORICAL)
            preds['raw'][q]=m.predict_proba(x.iloc[at])[:,1]
            simulated=[]
            for j,loc in enumerate(key):
                proposal=x.iloc[at].copy()
                proposal['Deck']=loc[0];proposal['Side']=loc[1];proposal['CabinNumber']=cabin[at,j]
                simulated.append(m.predict_proba(proposal)[:,1])
            sim=np.stack(simulated,axis=1)
            for name,w in [('plain_fill',prob),('constrained_fill',adjusted)]:
                preds[name][q]=(sim*w[at]).sum(1)
        report[tag]={}
        anchor=view['anchor']; missing=~known[ids]&view['known']
        for name in ['raw','plain_fill','constrained_fill']:
            dec=preds[name]>=.5
            for scheme,eligible in [('apply',missing),('changed_by_imputation',missing&(dec!=(preds['raw']>=.5))),
                                    ('changed_by_feasibility',missing&(dec!=(preds['plain_fill']>=.5)))]:
                results={}
                for scope,mask in [('missing',missing),('single',missing&((low[ids]<=high[ids]).sum(1)==1)),
                                   ('multiple',missing&((low[ids]<=high[ids]).sum(1)>1))]:
                    alter=eligible&mask&(dec!=anchor)
                    results[scope]=dict(rows=int(mask.sum()),fixed=int((alter&(dec==y[ids])).sum()),
                                        harmed=int((alter&(dec!=y[ids])).sum()))
                report[tag][name+'_'+scheme]=results
        print(tag,report[tag]['constrained_fill_changed_by_imputation'],flush=True)
        np.savez_compressed(ART/f'nov26_booking_cabin_choice_set/imputation_{tag}.npz',row=ids,fold=fold,
                            missing=~known[ids],**preds)
    DEST.with_name('likelihood_constrained_cabin.json').write_text(json.dumps(report,indent=2)+'\n')

def compare_odds_shifts():
    path=DEST.with_name('likelihood_constrained_cabin.json')
    report=json.loads(path.read_text())
    raw,y,g,f=data()
    _,_,_,rows,names,p,h,native,old_native,r=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        r[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,parent=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    for tag,ids,fold,base,archive,ref in [
        ('full',np.arange(len(y)),f,native,p,r),('historical',rows,hf,old_native,h,[parent,old,old])]:
        view=observed(ids,fold,archive,names,base,*ref,awake,tag)
        anchor=view['anchor']
        with np.load(ART/f'nov26_booking_cabin_choice_set/imputation_{tag}.npz') as z:
            np.testing.assert_array_equal(z['row'],ids)
            np.testing.assert_array_equal(z['fold'],fold)
            active=z['missing']&view['known']&(anchor==(base>=.5))
            for name,reference in [('plain_fill','raw'),('constrained_fill','raw'),('constrained_fill','plain_fill')]:
                def odds(s):
                    return logit(z[s].clip(.001,.999))
                score=logit(base.clip(.001,.999))+odds(name)-odds(reference)
                new=score>=0
                delta=active&(new!=anchor)
                report[tag][name+'_odds_from_'+reference]=dict(rows=int(active.sum()),
                    fixed=int((delta&(new==y[ids])).sum()),harmed=int((delta&(new!=y[ids])).sum()))
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
    compare_odds_shifts()
