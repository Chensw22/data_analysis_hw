"""Cross-fit opposite-side cabin residual evidence with signed displacement controls."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold

from joint_booking_outcomes_n23 import attributes

ROOT = Path(__file__).resolve().parent
ART = ROOT/'artifacts'
SHIFTS = {'opposite':0,'minus_seven':-7,'plus_seven':7}


def split(ix,groups,seed):
    for a,b in GroupKFold(2,shuffle=True,random_state=seed).split(ix,groups=groups[ix]):
        yield ix[a],ix[b]


def cross_risks(x,groups,y,t,q,seed):
    """Both donor self-estimates and query predictions exclude query labels."""
    assert not np.intersect1d(groups[t],groups[q]).size
    risk=np.full(len(groups),np.nan)
    q_risk=[]
    for a,b in split(t,groups,seed):
        model=CatBoostClassifier(depth=5,iterations=270,learning_rate=.07,l2_leaf_reg=6,
                                 verbose=False,allow_writing_files=False,random_seed=seed,thread_count=2)
        assert (y[a]>=0).all() and not np.intersect1d(groups[a],groups[b]).size
        model.fit(x[a],y[a])
        risk[b]=model.predict_proba(x[b])[:,1]
        q_risk.append(model.predict_proba(x[q])[:,1])
    return risk[t],np.mean(q_risk,axis=0)


def layout(raw):
    parts=raw.Cabin.str.extract(r'^(\w+)/(\d+)/(P|S)$')
    return [None if pd.isna(side) else (deck,int(num),side) for deck,num,side in parts.itertuples(index=False,name=None)]


def gather(berths,groups,t,q,residual,shift):
    lookup={}
    for i,r in zip(t,residual):
        if berths[i] is not None:
            lookup.setdefault(berths[i],[]).append((groups[i],r))
    signal=np.zeros(len(q))
    count=np.zeros(len(q),dtype=int)
    for j,i in enumerate(q):
        if berths[i] is None:
            continue
        deck,num,side=berths[i]
        key=(deck,num+shift,'S' if side=='P' else 'P')
        values=[r for g,r in lookup.get(key,[]) if g!=groups[i]]
        if values:
            signal[j]=np.mean(values)
            count[j]=len(values)
    return signal,count


def estimate(signals,risks,labels):
    offset=logit(np.clip(risks,1e-5,1-1e-5))
    def loss(coef):
        z=offset+coef*signals
        return (np.logaddexp(0,z)-labels*z).sum()+5*coef**2
    return float(minimize_scalar(loss,bounds=(-4,4),method='bounded').x)


def run():
    raw=pd.concat([pd.read_csv(ROOT/'data/raw'/f,usecols=lambda k:k not in ['Transported','Name'])
                   for f in ['train.csv','test.csv']],ignore_index=True)
    x,groups=attributes(raw)
    berths=layout(raw)
    cache=pd.read_csv(ART/'nov21_case_checks/v136_final_oof.csv')
    rows,fold,labels=[cache[k].to_numpy() for k in ['row','fold','true']]
    with np.load(ART/'v108_exact_oof/oof.npz') as z:
        np.testing.assert_array_equal(z['dev_ids'],rows)
        np.testing.assert_array_equal(z['fold_id'],fold)
        p108=z['v108']
    np.testing.assert_array_equal(raw.PassengerId.to_numpy()[rows],cache.PassengerId)
    # v131's cached release is a binary override on v108: use reflected v108 confidence.
    anchor=cache.v131.to_numpy()
    p0=np.where((p108>=.5)==anchor,p108,1-p108)
    p0=np.where(p0==.5,np.where(anchor,.5000001,.4999999),p0)
    y=np.full(len(raw),-1)
    y[rows]=labels
    results={key:p0.copy() for key in SHIFTS}
    support={key:np.zeros(len(rows),int) for key in SHIFTS}
    signal_oof={key:np.zeros(len(rows)) for key in SHIFTS}
    slopes={key:[] for key in SHIFTS}
    for k in range(3):
        q=rows[fold==k]
        t=rows[(fold!=k)&~np.isin(groups[rows],groups[q])]
        design={key:[] for key in SHIFTS}
        reference,train_y=[],[]
        for part,(a,b) in enumerate(split(t,groups,123+k)):
            pa,pb=cross_risks(x,groups,y,a,b,230+k*2+part)
            reference.extend(pb)
            train_y.extend(y[b])
            for key,d in SHIFTS.items():
                s,_=gather(berths,groups,a,b,y[a]-pa,d)
                design[key].extend(s)
        pt,_=cross_risks(x,groups,y,t,q,630+k)
        for key,d in SHIFTS.items():
            s,n=gather(berths,groups,t,q,y[t]-pt,d)
            coef=estimate(np.asarray(design[key]),np.asarray(reference),np.asarray(train_y))
            slopes[key].append(coef)
            support[key][fold==k]=n
            signal_oof[key][fold==k]=s
            results[key][fold==k]=expit(logit(np.clip(p0[fold==k],1e-7,1-1e-7))+coef*s)
        print('fold',k,{key:round(v[-1],4) for key,v in slopes.items()},flush=True)
    hard=cache.hard.to_numpy()
    ntrain=len(pd.read_csv(ROOT/'data/raw/train.csv',usecols=['PassengerId']))
    _,official_n=gather(berths,groups,np.arange(ntrain),np.arange(ntrain,len(groups)),np.zeros(ntrain),0)
    metrics={}
    for key,p in results.items():
        fixed=(labels!=anchor)&((p>=.5)==labels)
        harmed=(labels==anchor)&((p>=.5)!=labels)
        active=support[key]>0
        corr=np.corrcoef(labels[active]-p0[active],signal_oof[key][active])[0,1]
        metrics[key]=dict(accuracy=float(((p>=.5)==labels).mean()), fixed=int(fixed.sum()),harmed=int(harmed.sum()),
            hard_fixed=int((fixed&hard).sum()),hard_harmed=int((harmed&hard).sum()),covered=int(active.sum()),
            fold_net=[int(fixed[fold==k].sum())-int(harmed[fold==k].sum()) for k in range(3)],
            fold_slope=slopes[key],residual_correlation=float(corr))
    out=ART/'nov23_opposite_cabin'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'oof.npz',row=rows,fold=fold,y=labels,p=results['opposite'],p0=p0,**results,
                        **{k+'_count':v for k,v in support.items()},**{k+'_signal':v for k,v in signal_oof.items()})
    report=ROOT/'reports/nov23_opposite_cabin'
    report.mkdir(exist_ok=True)
    (report/'result.json').write_text(json.dumps(dict(metrics=metrics,test_layout_coverage=int((official_n>0).sum()),
        test_rows=len(official_n),scope='Training and calibration each exclude query bookings. Donor residual outcomes minus grouped cross-fitted CatBoost predictions. Signed transfers fit only on inner calibration rows from different bookings. Incumbent v131 hard overrides use reflected v108 score, not calibrated v131 probabilities. Each fit uses only its training subset of labels. Query labels are used only to score the final predictions.'),indent=2)+'\n')
    print(metrics,flush=True)


if __name__=='__main__':
    run()
