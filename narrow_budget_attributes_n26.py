"""Measure what an exact amount identifies after the registered planet and deck are held fixed."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from cross_deck_budget_multiples_n26 import fields
from full_roster_booking_validation_n23 import ROOT,data
from latent_bill_rounding_n26 import BILLS
from tabicl_consensus_review_n26 import sources


def target_codes(frame,g,y):
    n=len(y)
    cabin=frame.Cabin.str.split('/',expand=True)
    fields={'destination':frame.Destination,'side':cabin[2],'age':pd.cut(frame.Age,[-1,12,17,24,39,59,150]),
            'party_size':pd.Series(g).map(pd.Series(g).value_counts()).clip(upper=4),
            'order_period':pd.Series(g).astype(int)//1000,'dominant_fee':frame[BILLS].fillna(-1).idxmax(axis=1)}
    for name,s in fields.items():
        ids,levels=pd.factorize(s)
        yield name,ids,len(levels)
    yield 'transport',np.r_[y.astype(int),np.full(len(g)-n,-1)],2


def distributions(frame,g,y,t,q):
    total,count,paid,origin,deck=fields(frame)
    cost=2*total+count
    target=list(target_codes(frame,g,y))
    stats={n:np.full((len(q),3),np.nan) for n,_,_ in target}
    for category in set(zip(origin[q],deck[q])):
        p,d=category
        pool=t[paid[t]&(origin[t]==p)&(deck[t]==d)]
        if len(pool)<15:continue
        ordered=pool[np.argsort(cost[pool],kind='stable')]
        selected=np.flatnonzero(paid[q]&(origin[q]==p)&(deck[q]==d))
        for i in selected:
            person=q[i]
            # Both kernels use the same other orders. Pick one representative without seeing their labels.
            width=160.
            lower=np.searchsorted(cost[ordered],cost[person]-3*width,side='left')
            upper=np.searchsorted(cost[ordered],cost[person]+3*width,side='right')
            bank=ordered[lower:upper]
            if not len(bank):continue
            delta=cost[bank]-cost[person]
            nearest=np.argsort(abs(delta),kind='stable')
            _,distinct=np.unique(g[bank[nearest]],return_index=True)
            use=nearest[distinct]
            donors=bank[use];delta=delta[use]
            assert not np.any(g[donors]==g[person])
            kernel=np.exp(-.5*(delta[:,None]/np.array([42,2.4]))**2)
            # Integrate a displaced narrow peak from the same 64-unit range as a control.
            displaced=2*(((total[person]+23)%64)-(total[person]%64))
            ker_sham=np.exp(-.5*((delta-displaced)/2.4)**2)
            for name,labels,width_target in target:
                outcome=labels[person]
                ok=labels[donors]>=0
                if outcome<0 or not ok.any():continue
                hit=labels[donors[ok]]==outcome
                weights=kernel[ok]
                prior=(hit.sum()+1)/(len(hit)+width_target)
                wide=(weights[:,0]@hit+5*prior)/(weights[:,0].sum()+5)
                v=weights[:,1]
                fine=(v@hit+5*wide)/(v.sum()+5)
                w=ker_sham[ok]
                fake=(w@hit+5*wide)/(w.sum()+5)
                stats[name][i]=np.log([wide,fine,fake])
    return stats


def run():
    raw,y,g,_=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    full=pd.concat([raw,test],ignore_index=True)
    group=full.PassengerId.str.split('_').str[0].to_numpy()
    log={}
    folds=np.empty(len(full),int)
    for k,(t,q) in enumerate(GroupKFold(5,shuffle=True,random_state=9664).split(full,groups=group)):
        assert not np.isin(group[t],group[q]).any()
        folds[q]=k
        for n,block in distributions(full,group,y,t,q).items():
            log.setdefault(n,np.empty((len(full),3)))[q]=block
    cur=sources()[-1]['full']
    common=np.r_[cur['common'],np.zeros(len(test),bool)]
    report={}
    for n,values in log.items():
        report[n]={}
        for scope,mask in [('all',np.ones(len(full),bool)),('training',np.arange(len(full))<len(y)),
                           ('consensus_training',common),('competition',np.arange(len(full))>=len(y))]:
            present=mask&np.isfinite(values).all(1)
            if not present.any():continue
            w,s,f=values[present].mean(0)
            shifts=values[:,1]-values[:,0]
            report[n][scope]=dict(rows=int(present.sum()),fine_over_wide_nats=float(s-w),fine_over_displaced_nats=float(s-f),
                                  mean_nll=[float(-v) for v in (w,s,f)],
                                  fold_gains=[float(shifts[present&(folds==j)].mean()) for j in range(5)])
    print([(n,round(sc['all']['fine_over_wide_nats'],4),round(sc['all']['fine_over_displaced_nats'],4)) for n,sc in report.items()])
    (ROOT/'reports/nov23_matched_learning_curve/narrow_budget_attributes.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
