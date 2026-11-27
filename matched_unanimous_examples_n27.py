"""Describe common wrong decisions alongside comparably registered passengers with the opposite outcome."""
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import cohort
from full_roster_booking_validation_n23 import ROOT
from unanimous_error_reach_n27 import inputs

REPORT=ROOT/'reports/nov23_matched_learning_curve/matched_unanimous_examples.json'


def run():
    raw,frame,y,g,fold,first,random,ordered,wide,_=inputs()
    _,_,known,_=cohort(raw,y,fold)
    n=len(y)
    pred=(first^random)[:n]
    agree=known&((wide[:n]>=.5)==pred[:,None]).all(1)
    paid=raw[FEES].gt(0).any(axis=1).to_numpy()
    asleep=raw.CryoSleep.eq(True).to_numpy()
    awake=raw.CryoSleep.eq(False).to_numpy()
    party=raw.groupby(g).PassengerId.transform('size').to_numpy()
    cab=raw.Cabin.str.split('/',expand=True)
    berth=pd.to_numeric(cab[1],errors='coerce').to_numpy(float)
    band=pd.DataFrame(dict(planet=raw.HomePlanet,dest=raw.Destination,sleep=raw.CryoSleep,deck=cab[0],side=cab[2],vip=raw.VIP,
                           group_size=party,decision=pred,paid=paid))
    strata=band.groupby(band.columns.tolist(),dropna=False).ngroup().to_numpy()
    bills=raw[FEES].to_numpy(float)
    x=np.column_stack([np.nan_to_num(raw.Age.to_numpy(float),nan=-25)/15,np.nan_to_num(berth,nan=-300)/200,
                       np.nan_to_num(np.log1p(bills),nan=-2)/3])
    err=agree&(y!=pred)
    good=agree&(y==pred)
    def sample(i):
        d=dict(id=str(raw.PassengerId.iloc[i]),order=str(g[i]),age=None if pd.isna(raw.Age.iloc[i]) else int(raw.Age.iloc[i]),
               cabin=None if pd.isna(raw.Cabin.iloc[i]) else str(raw.Cabin.iloc[i]),true=bool(y[i]),pred=bool(pred[i]))
        d['bills']=[None if np.isnan(v) else int(v) for v in bills[i]]
        return d
    output={}
    for scope,mask in [('all',np.ones(n,bool)),('sleep_zero',asleep&~paid),('awake_paid',awake&paid),('solo_sleep_zero',asleep&~paid&(party==1))]:
        dists=[]; pairs=[]
        for group in np.unique(strata[mask&err]):
            q=np.flatnonzero(err&mask&(strata==group))
            others=np.flatnonzero(good&mask&(strata==group))
            if not len(others):continue
            distance=cdist(x[q],x[others])
            distance[g[q,None]==g[others][None,:]]=np.inf
            ix=np.argmin(distance,axis=1)
            for a,b,c in zip(q,others[ix],distance[np.arange(len(q)),ix]):
                if np.isfinite(c):
                    pairs.append((float(c),int(a),int(b)))
                    dists.append(c)
        pairs.sort()
        examples=[]
        if pairs:
            for j in [0,len(pairs)//2]:
                dist,a,b=pairs[j]
                examples.append(dict(error=sample(a),matching_correct=sample(b),normalized_distance=dist,
                                     shared_fields=band.iloc[a].astype(str).to_dict()))
        output[scope]=dict(unanimous=int((mask&agree).sum()),errors=int((err&mask).sum()),
            false_positive=int((err&mask&pred).sum()),false_negative=int((err&mask&~pred).sum()),
            matched_different_booking=len(pairs),exact=int(np.sum(np.array(dists)==0)),
            nearest_distance_quartiles=np.quantile(dists,[.25,.5,.75]).tolist() if dists else [],examples=examples)
    REPORT.write_text(json.dumps(output,indent=2)+'\n')
    for k,row in output.items():print(k,{key:row[key] for key in ('errors','false_positive','false_negative','matched_different_booking','exact','nearest_distance_quartiles')},row['examples'][0]['error'],row['examples'][0]['matching_correct'])


if __name__=='__main__':
    run()
