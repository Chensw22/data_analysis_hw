"""Learn a passenger location effect from discordant outcomes within identical manifest profiles."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from archive_source_harm_veto_n28 import inputs
from repeated_archive_misses_n26 import ART, ROOT, cohort, count
from price_implied_deck_n26 import FEES


def contexts(frame):
    cabin=frame.Cabin.str.split('/',expand=True)
    category=cabin[0].fillna('?')
    side=cabin[2].fillna('?')
    group=frame.PassengerId.str[:4]
    position=frame.PassengerId.str[5:].astype(int)
    cab_num=pd.to_numeric(cabin[1]).groupby(category).rank(pct=True).fillna(.5).to_numpy()
    cat=pd.get_dummies(category,dtype=float).to_numpy()
    curves=np.column_stack([cab_num-.5,*[np.maximum(0,cab_num-k) for k in [.2,.4,.6,.8]]])
    continuous=np.column_stack([curves,*[curves*v[:,None] for v in cat.T],
        np.log1p(group.map(group.value_counts())),np.log1p(position),np.log1p(frame.Cabin.map(frame.Cabin.value_counts()).fillna(1)),
        pd.get_dummies(category+'/'+side,dtype=float)])
    x=np.column_stack([continuous,*[continuous*col.to_numpy(float)[:,None] for col in [
        frame.CryoSleep.eq(True),frame.HomePlanet.eq('Europa'),frame.HomePlanet.eq('Mars')]]])
    fields=['HomePlanet','CryoSleep','Destination','Age','VIP',*FEES]
    profile=frame.groupby(fields,dropna=False,sort=False).ngroup().to_numpy()
    return x,profile


def score_fold(train,query,x,profile,group,y):
    pairs=[];weights=[]
    grouped=pd.Series(train).groupby(profile[train]).apply(np.asarray)
    for members in grouped:
        a=members[y[members]];b=members[~y[members]]
        if len(a)==0 or len(b)==0:continue
        i,j=np.where(group[a,None]!=group[b][None,:])
        if len(i)==0:continue
        pairs.append(x[a[i]]-x[b[j]])
        weights.append(np.full(len(i),len(members)/len(i)))
    delta=np.vstack(pairs);weights=np.concatenate(weights)
    model=LogisticRegression(C=.3,solver='newton-cholesky',fit_intercept=False)
    model.fit(np.vstack([delta,-delta]),np.repeat([1,0],len(delta)),sample_weight=np.tile(weights/2,2))
    raw=x@model.coef_[0]
    score=np.zeros((len(query),3))
    for key,ix in pd.Series(np.arange(len(query))).groupby(profile[query]):
        loc=ix.to_numpy()
        if key not in grouped:continue
        ref=grouped[key]
        if len(np.unique(group[ref]))<2:continue
        prior=(y[ref].sum()+1)/(len(ref)+2)
        reference=raw[ref]
        # Match the expected number transported under the fitted within-profile ranking.
        lo,hi=-40.,40.
        for _ in range(40):
            mid=(lo+hi)/2
            if expit(reference+mid).mean()>prior:hi=mid
            else:lo=mid
        absolute=raw[query[loc]]+(lo+hi)/2
        score[loc]=np.column_stack([absolute,absolute-logit(prior),np.full(len(loc),len(ref))])
    return score


def run():
    frame,y,g,f,ids,_,_,seven,v=inputs();n=len(y);parent=v['random'][:,0]
    _,_,known,fresh=cohort(frame.iloc[:n],y,f)
    wrong=parent[:n]!=y
    margin=np.where(parent,1.,-1.)*abs(logit(np.median(seven[:,1:4],axis=1).clip(.002,.998)))
    x,profile=contexts(frame);groups=frame.PassengerId.str[:4].to_numpy()
    ordered=np.empty(n,int)
    for k,chunk in enumerate(np.array_split(np.unique(g),8)):ordered[np.isin(g,chunk)]=k
    report,saved={},{}
    for tag,part in [('random',f),('ordered',ordered)]:
        scores=np.zeros((len(frame),3))
        for k in range(9):
            train=np.flatnonzero(part!=k)
            query=np.flatnonzero(part==k) if k<8 else np.arange(n,len(frame))
            assert not np.isin(groups[train],groups[query]).any()
            if k==8 and tag=='ordered':scores[query]=saved['random'][query]
            else:scores[query]=score_fold(train,query,x,profile,groups,y)
        absolute,increment,support=scores.T
        opposite=(absolute>=0)!=parent
        moved=(margin+increment>=0)!=parent
        rules={'direct':(support>0)&opposite,'marginal':(support>0)&moved,'corroborated':(support>0)&moved&opposite}
        same=(v[tag]==parent[:,None]).all(1)
        scope={'all':known,'new_orders':known&fresh,'four':known&same[:n],
               'four_seven':known&same[:n]&((seven[:n]>=.5)==parent[:n,None]).all(1)}
        report[tag]={'matched_known':int((known&(support[:n]>0)).sum()),'matched_test':int((support[n:]>0).sum()),'rules':{}}
        for label,flag in rules.items():
            row={s:count(flag[:n],m,wrong) for s,m in scope.items()}
            row.update(test=int(flag[n:].sum()),test_four=int((flag[n:]&same[n:]).sum()))
            report[tag]['rules'][label]=row
        saved[tag]=scores
        print(tag,report[tag],flush=True)
    dest=ART/'nov28_exact_profile_location_rank';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'aligned.npz',ids=ids,y=y,fold=f,**saved)
    (ROOT/'reports/nov23_matched_learning_curve/exact_profile_location_rank.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):run()
