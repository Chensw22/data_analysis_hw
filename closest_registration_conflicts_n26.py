"""Count the minimum number of differing registration fields to each outcome in other training bookings."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from distinct_challenger_error_reach_n26 import population
from target_mix_expert_routing_n26 import ROOT,ticker
from price_implied_deck_n26 import FEES

REPORT=ROOT/'reports/nov23_matched_learning_curve/closest_registration_conflicts.json'


def profile(raw,g,coarse):
    table=raw[['HomePlanet','CryoSleep','Destination','VIP','Age',*FEES]].copy()
    table[['deck','berth','side']]=raw.Cabin.str.split('/',expand=True)
    table['party']=raw.groupby(g).PassengerId.transform('size')
    if coarse:
        table.Age=np.floor(table.Age/8)
        table['berth']=np.floor(pd.to_numeric(table.berth)/50)
        table['party']=table.party.clip(upper=4)
        table[FEES]=np.ceil(np.log2(1+table[FEES]))
    return np.column_stack([pd.factorize(table[key],sort=False)[0] for key in table])


def neighbors(table,t,q,g,y):
    _,groups=np.unique(g[t],return_inverse=True)
    dist_by_class=np.empty((len(q),2),int)
    group_counts=np.zeros(len(q),int)
    positives=np.empty(len(q))
    for k in range(0,len(q),32):
        query=q[k:k+32]
        dist=np.zeros((len(query),len(t)),dtype=np.uint8)
        for field in table.T:
            dist+=field[query,None]!=field[t]
        block=dist_by_class[k:k+len(query)]
        for label in (0,1):
            block[:,label]=dist[:,y[t]==label].min(1)
        for j,d in enumerate(dist):
            closest=d==d.min()
            grp=groups[closest]
            totals=np.bincount(grp)
            keep=totals>0
            group_counts[k+j]=keep.sum()
            positives[k+j]=(np.bincount(grp,weights=y[t][closest],minlength=len(totals))[keep]/totals[keep]).mean()
    return dist_by_class,group_counts,positives


def run():
    raw,y,g,f,parent,same,count,_,_,wrong,_,paid=population()
    blind=same&(count==0)
    awake=blind & ~parent & paid & raw.CryoSleep.eq(False).to_numpy()
    report={}
    for level in ('recorded','coarse'):
        x=profile(raw,g,level=='coarse')
        assert x.shape==(len(y),14)
        dist=np.full((len(y),2),999,int)
        size=np.zeros(len(y),int)
        p=np.full(len(y),np.nan)
        for k in range(8):
            t,q=np.flatnonzero(f!=k),np.flatnonzero(same&(f==k))
            assert not np.isin(g[t],g[q]).any()
            dist[q],size[q],p[q]=neighbors(x,t,q,g,y)
        match=dist[np.arange(len(y)),y.astype(int)]
        opposite=dist[np.arange(len(y)),(~y).astype(int)]
        vote=(p>=.5)!=parent
        agreement=(p==0)|(p==1)
        report[level]={}
        for label,scope in dict(all_unanimous=same,all_127_agree=blind,awake_paid_negative_all_agree=awake).items():
            miss=scope&wrong
            r=dict(rows=int(scope.sum()),errors=int(miss.sum()),same_outcome_closer=int((miss&(match<opposite)).sum()),
                   equal_distance=int((miss&(match==opposite)).sum()),opposite_closer=int((miss&(match>opposite)).sum()),
                   exact_opposite=int((miss&(opposite==0)).sum()),exact_same=int((miss&(match==0)).sum()),
                   any_exact=int((miss&(dist.min(1)==0)).sum()),
                   all_tied_order_vote=ticker(scope,wrong,vote&(p!=.5)),
                   at_least_two_unanimous_bookings=ticker(scope,wrong,vote&agreement&(size>=2)),
                   closest_mismatched_fields_for_errors={str(i):int((miss&(dist.min(1)==i)).sum()) for i in np.unique(dist.min(1)[miss])})
            assert r['same_outcome_closer']+r['equal_distance']+r['opposite_closer']==r['errors']
            report[level][label]=r
        print(level,report[level]['all_127_agree'],flush=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
