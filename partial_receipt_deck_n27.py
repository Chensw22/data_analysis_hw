"""Infer deck from the observed subtotal of incomplete bills using identically censored reference receipts."""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from exaone_validation_data import CATEGORICAL, compact_features
from full_roster_booking_validation_n23 import ROOT, data
from price_implied_deck_n26 import DECKS, FEES
from repeated_archive_misses_n26 import ART, cohort, count
from weighted_bill_ridges_n26 import deck_posterior

OUT=ART/'nov27_partial_receipt_deck'
REPORT=ROOT/'reports/nov23_matched_learning_curve/partial_receipt_deck_n27.json'


def prepare(frame,groups):
    n=len(frame)
    bills=frame[FEES].to_numpy()
    bits=1<<np.arange(5)
    mask=np.isnan(bills)@bits
    complete=mask==0
    deck=pd.Index(DECKS).get_indexer(frame.Cabin.str.split('/').str[0])
    planet=frame.HomePlanet.fillna('?').to_numpy()
    # Unlabelled, whole-booking cross-fitting of auxiliary deck features, including the test manifest.
    synth=mask.copy()
    samples=np.random.default_rng(7113).choice(mask[~complete],size=int(complete.sum()))
    synth[complete]=samples
    designs=[]
    audits=[]
    for codes in (mask,synth):
        masked=np.where((codes[:,None]&bits)>0,np.nan,bills)
        x=compact_features(frame)
        x[FEES]=masked
        x['observed_total']=np.nansum(masked,axis=1)
        x['booking_size']=pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
        post=np.full((2,n,len(DECKS)),np.nan)
        for t,q in GroupKFold(5,shuffle=True,random_state=9664).split(frame,groups=groups):
            assert not np.isin(groups[t],groups[q]).any()
            bank=t[complete[t]&(deck[t]>=0)]
            for code in np.unique(codes[q]):
                visible=(code&bits)==0
                if not visible.any():continue
                observed=np.nan_to_num(bills[:,visible],nan=0.)
                cost=(2*observed.sum(1)+(observed>0).sum(1)).astype(int)
                for orig in np.unique(planet[q[codes[q]==code]]):
                    pool=bank[planet[bank]==orig]
                    if len(pool)<25:pool=bank
                    ask=q[(codes[q]==code)&(planet[q]==orig)&(cost[q]>0)]
                    if len(ask):
                        for j,shift in enumerate((False,True)):
                            post[j,ask]=deck_posterior(cost,pool,ask,deck,1.,shift)
        modes={'base':x}
        good=np.isfinite(post[0]).all(1)&(deck>=0)&(codes!=0)
        audits.append({k:dict(count=int(good.sum()),accuracy=float((post[j,good].argmax(1)==deck[good]).mean()))
                       for j,k in enumerate(['aligned','shifted'])})
        for j,k in enumerate(('aligned','shifted')):
            extra=pd.DataFrame(post[j],columns=['estimated_'+d for d in DECKS])
            extra['registered']=np.where(deck>=0,post[j,np.arange(n),deck.clip(0)],np.nan)
            modes[k]=pd.concat([x,extra],axis=1)
        designs.append(modes)
    return designs,mask,audits


def run():
    train,y,g,f=data()
    base,blind,known,fresh=cohort(train,y,f)
    frame=pd.concat([train,pd.read_csv(ROOT/'data/raw/test.csv',usecols=train.columns)],ignore_index=True)
    groups=frame.PassengerId.str.split('_').str[0].to_numpy()
    views,mask,aux=prepare(frame,groups)
    n=len(y)
    scores={key:np.full(len(frame),np.nan) for key in ('base','augmented','aligned','shifted')}
    for k in range(9):
        t=np.flatnonzero(f!=k) if k<8 else np.arange(n)
        q=np.flatnonzero(f==k) if k<8 else np.arange(n,len(frame))
        assert not np.isin(groups[t],groups[q]).any()
        full=t[mask[t]==0]
        for key,p in scores.items():
            mode='base' if key=='augmented' else key
            xt=views[0][mode].iloc[t]
            labels=y[t]
            weights=np.ones(len(t))
            if key!='base':
                xt=pd.concat([xt,views[1][mode].iloc[full]],ignore_index=True)
                labels=np.r_[labels,y[full]]
                weights=np.r_[np.where(mask[t]==0,.8,1.),np.full(len(full),.2)]
            model=CatBoostClassifier(iterations=380,depth=5,learning_rate=.055,l2_leaf_reg=6,
                                     cat_features=CATEGORICAL,random_seed=3027,thread_count=2,
                                     verbose=False,allow_writing_files=False)
            model.fit(xt,labels,sample_weight=weights)
            p[q]=model.predict_proba(views[0][mode].iloc[q])[:,1]
        print(k,flush=True)
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
        anchor=z['anchor']
        np.testing.assert_array_equal(anchor[:n],base)
    incomplete=mask!=0
    paid=frame[FEES].gt(0).any(axis=1).to_numpy()
    acts={k:incomplete&((p>=.5)!=anchor) for k,p in scores.items()}
    acts['aligned_exclusive']=acts['aligned']&~acts['augmented']&~acts['shifted']
    acts['two_deck_estimates']=acts['aligned']&acts['shifted']&~acts['augmented']
    acts['censor_only']=acts['augmented']&~acts['base']
    scopes=dict(all=known,missing=known&incomplete[:n],missing_paid=known&incomplete[:n]&paid[:n],
                fresh=known&fresh,common31=blind)
    report=dict(auxiliary_deck=aux,missing_train=int(incomplete[:n].sum()),missing_test=int(incomplete[n:].sum()),
                missing_accuracy={k:float(((p[:n]>=.5)==y)[incomplete[:n]].mean()) for k,p in scores.items()},rules={})
    for k,flag in acts.items():
        delta=flag[:n]*np.where(base!=y,1,-1)
        report['rules'][k]=dict(test_edits=int(flag[n:].sum()),net_by_fold=[int(delta[known&(f==j)].sum()) for j in range(8)],
                                 scopes={s:count(flag[:n],m,base!=y) for s,m in scopes.items()})
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=f,missing=incomplete,**scores)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['missing_accuracy'],report['rules']['aligned_exclusive'])


if __name__=='__main__':
    with threadpool_limits(2):
        run()
