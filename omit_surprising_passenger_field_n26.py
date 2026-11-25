"""Test omitting the passenger field least consistent with other fields in nearby training records."""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from distinct_challenger_error_reach_n26 import population
from exaone_validation_data import CATEGORICAL,compact_features
from target_mix_expert_routing_n26 import ROOT,ticker

REPORT=ROOT/'reports/nov23_matched_learning_curve/omit_surprising_passenger_field.json'
CACHE=ROOT/'artifacts/nov26_omitted_passenger_fields'


def field_values(frame,ref):
    encoded=[]
    observed=[]
    for n in frame:
        s=frame[n]
        if n in CATEGORICAL:
            valid=s.ne('__MISSING__').to_numpy()
            x=pd.factorize(s)[0]
        else:
            valid=s.notna().to_numpy()
            z=s.to_numpy(float)
            vals=np.sort(z[ref][valid[ref]])
            # Mid-ranks keep tied zero bills at their own probability position.
            x=(np.searchsorted(vals,z,'left')+np.searchsorted(vals,z,'right'))/(2*len(vals))
        encoded.append(x)
        observed.append(valid)
    return np.column_stack(encoded),np.column_stack(observed)


def surprise(frame,ref,rows):
    x,obs=field_values(frame,ref)
    categorical=np.array([n in CATEGORICAL for n in frame])
    scores=np.full((len(rows),x.shape[1]),np.nan)
    for start in range(0,len(rows),20):
        q=rows[start:start+20]
        diff=np.abs(x[q,None,:]-x[ref][None,:,:]).astype(np.float32)
        diff[:,:,categorical]=diff[:,:,categorical]!=0
        available=obs[q,None,:]&obs[ref][None,:,:]
        diff=np.where(available,diff,.5)
        total=diff.sum(2)
        for j in range(x.shape[1]):
            distance=total-diff[:,:,j]
            distance[:,~obs[ref,j]]=np.inf
            close=np.argpartition(distance,31,axis=1)[:,:32]
            mismatch=np.take_along_axis(diff[:,:,j],close,axis=1).mean(1)
            scores[start:start+len(q),j]=np.where(obs[q,j],mismatch,np.nan)
    return scores


def rank_by_calibration(cal,scores):
    rank=np.full_like(scores,-1.)
    for j in range(scores.shape[1]):
        ref=np.sort(cal[np.isfinite(cal[:,j]),j])
        good=np.isfinite(scores[:,j])
        value=scores[good,j]
        rank[good,j]=(np.searchsorted(ref,value,'left')+np.searchsorted(ref,value,'right'))/(2*len(ref))
    return rank


def mask(frame,columns):
    result=frame.copy()
    names=list(frame)
    for j,n in enumerate(names):
        result.loc[columns==j,n]='__HIDDEN__' if n in CATEGORICAL else np.nan
    result['hidden_field']=[names[j] if j>=0 else 'none' for j in columns]
    return result


def run():
    raw,y,g,fold,parent,same,count,_,_,wrong,_,_=population()
    frame=compact_features(raw)
    frame['OrderSize']=raw.groupby(g).PassengerId.transform('size')
    CACHE.mkdir(exist_ok=True)
    scores=np.zeros((len(y),3))
    selected=np.zeros(len(y),int)
    sham=np.zeros(len(y),int)
    for k in range(8):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        bucket=partition(g[t],'field-reconstruction:',6)
        ref,cal=t[bucket!=0],t[bucket==0]
        assert not np.isin(g[ref],g[cal]).any() and not np.isin(g[t],g[q]).any()
        s=surprise(frame,ref,np.r_[cal,q])
        rank=rank_by_calibration(s[:len(cal)],s[len(cal):])
        rng=np.random.default_rng(9207+k)
        choice=rank.argmax(1)
        # Displace the score vectors within the fold, excluding fields missing on the recipient.
        shuffled=rank[rng.permutation(len(q))].copy()
        shuffled[rank<0]=-2
        alt=shuffled.argmax(1)
        assert (rank[np.arange(len(q)),choice]>=0).all() and (rank[np.arange(len(q)),alt]>=0).all()
        train=frame.iloc[t]
        x=pd.concat([mask(train,np.full(len(t),-1)),mask(train,rng.integers(0,len(frame.columns),len(t)))])
        fitted=CatBoostClassifier(iterations=380,depth=5,learning_rate=.055,l2_leaf_reg=6,random_seed=3027,
                                  thread_count=3,verbose=False,allow_writing_files=False,cat_features=[*CATEGORICAL,'hidden_field'])
        fitted.fit(x,np.tile(y[t],2))
        for col,which in enumerate([np.full(len(q),-1),choice,alt]):
            scores[q,col]=fitted.predict_proba(mask(frame.iloc[q],which))[:,1]
        selected[q],sham[q]=choice,alt
        print(k,flush=True)
    assert np.isfinite(scores).all()
    path=CACHE/'scores.npz'
    np.savez_compressed(path,ids=raw.PassengerId.to_numpy(str),fold=fold,y=y,names=frame.columns.to_numpy(str),
                        selected=selected,sham=sham,p=scores)
    summarize(scores,selected,sham,frame.columns,y,fold,parent,same,count,wrong)


def summarize(scores,selected,sham,columns,y,fold,parent,same,count,wrong):
    baseline=scores[:,0]>=.5
    chosen,control=(scores[:,j]>=.5 for j in (1,2))
    report=dict(base_accuracy=float((baseline==y).mean()),omission_accuracy=float((chosen==y).mean()),
                displaced_accuracy=float((control==y).mean()),by_field={},scopes={})
    for j,n in enumerate(columns):
        report['by_field'][n]=dict(selected=int((selected==j).sum()),displaced=int((sham==j).sum()))
    for n,scope in [('ten_agree',same),('any_challenger',same&(count>0)),('none',same&(count==0))]:
        results={}
        for key,changed in [('original',baseline!=parent),('omission',chosen!=parent),
                            ('new_after_omission',(chosen!=parent)&(baseline==parent)),
                            ('new_after_displacement',(control!=parent)&(baseline==parent)),
                            ('recorded_field_required',(baseline!=parent)&(chosen==parent)),
                            ('required_only_with_actual_choice',(baseline!=parent)&(chosen==parent)&(control!=parent)),
                            ('only_real',(chosen!=parent)&(control==parent)&(baseline==parent))]:
            r=ticker(scope,wrong,changed)
            r['net_by_fold']=[int((np.where(wrong,1,-1)*scope*changed*(fold==k)).sum()) for k in range(8)]
            results[key]=r
        report['scopes'][n]=results
        print(n,{k:(v['fixed'],v['harmed']) for k,v in results.items()},flush=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
