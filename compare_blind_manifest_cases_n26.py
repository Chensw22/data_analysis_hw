"""Inspect the shared mistakes against physically neighboring passengers from separate held-out bookings."""
import json

import numpy as np
import pandas as pd

from exhaustive_family_quorums_n26 import summary
from full_roster_booking_validation_n23 import ROOT,data
from nori_booking_regression_n26 import FEES

REPORT=ROOT/'reports/nov23_matched_learning_curve/blind_manifest_cases.json'


def match(raw,y,g,f,omit=None):
    cabin=raw.Cabin.str.split('/',expand=True)
    size=raw.groupby(g).PassengerId.transform('size')
    frame=raw[['HomePlanet','CryoSleep','Destination','VIP']].copy()
    frame['deck'],frame['side'],frame['size']=cabin[0],cabin[2],size
    # Missing, zero and positive are kept separate in each service.
    for c in FEES:
        frame[c]=np.where(raw[c].isna(),-1,raw[c].gt(0).astype(int))
    frame['paid']=raw[FEES].gt(0).any(axis=1)
    drop={'registration':['HomePlanet','Destination','VIP'], 'cabin':['deck','side'],
          'bill':FEES, 'size':['size']}
    if omit:
        frame=frame.drop(columns=drop[omit])
    age=raw.Age.to_numpy(float)
    berth=pd.to_numeric(cabin[1],errors='coerce').to_numpy(float)
    fees=raw[FEES].fillna(0).to_numpy()
    levels=['close','broad']
    support={k:np.zeros(len(raw),int) for k in levels}
    fraction={k:np.zeros(len(raw)) for k in levels}
    nearest=np.full(len(raw),-1,int)
    distances=np.full(len(raw),np.inf)
    for rows in frame.groupby(list(frame),sort=False,dropna=False).indices.values():
        rows=np.asarray(rows)
        valid=rows[((omit=='registration')|np.isfinite(age[rows]))&((omit=='cabin')|np.isfinite(berth[rows]))]
        for i in valid:
            ref=valid[(f[valid]!=f[i])&(f[valid]>=0)]
            if not len(ref):
                continue
            assert not np.any(g[ref]==g[i])
            d=np.zeros(len(ref))
            if omit!='bill':
                price_gap=np.max(abs(fees[ref]-fees[i])/np.maximum(50.,.15*np.maximum(fees[ref],fees[i])),axis=1)
                d=np.maximum(d,price_gap)
            if omit!='registration':
                d=np.maximum(d,abs(age[ref]-age[i])/3.)
            if omit!='cabin':
                d=np.maximum(d,abs(berth[ref]-berth[i])/100.)
            # Keep one representative per reference booking, closest under the predeclared distance.
            order=np.argsort(d,kind='stable')
            _,ix=np.unique(g[ref[order]],return_index=True)
            keep=order[np.sort(ix)]
            ref,d=ref[keep],d[keep]
            nearest[i],distances[i]=ref[0],d[0]
            for name,max_gap in zip(levels,[1.,2.]):
                samples=ref[d<=max_gap]
                support[name][i]=len(samples)
                if len(samples):
                    fraction[name][i]=y[samples].mean()
    return support,fraction,nearest,distances


def run():
    raw,y,g,f=data()
    with np.load(ROOT/'artifacts/nov26_retained_order_mechanisms/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId)
        np.testing.assert_array_equal(z['fold'],f)
        base=z['reference161']
        known=z['known']
        blind=z['eligible']&((z['pairs']>=.5)==base[:,None,None]).all(axis=(1,2))&((z['original_panel']>=.5)==base[:,None]).all(1)
    wrong=base!=y
    assert blind.sum()==3861 and np.count_nonzero(wrong&blind)==266
    paid=raw[FEES].gt(0).any(axis=1).to_numpy()
    scopes=dict(all_known=known,unanimous_31=blind,awake_paid_negative=blind&raw.CryoSleep.eq(False).to_numpy()&paid&~base,
                sleeping_no_positive=blind&raw.CryoSleep.eq(True).to_numpy()&~paid&base)
    support,fraction,nearest,distance=match(raw,y,g,f)
    report=dict(distance='Require equal origin, destination, sleep, VIP, deck, side, exact booking size and each bill missing/zero/positive state. In the close match: known age within 3 years, known berth number within 100, each positive bill differs by at most max(50,15% of the larger bill). The broad match doubles all continuous tolerances. One closest passenger per external training booking, with no use of truth in choosing neighbors.')
    for name,area in scopes.items():
        info={}
        for level,nbr in support.items():
            prob=fraction[level]
            mixed=(prob>0)&(prob<1)
            action=(nbr>=3)&np.where(base,prob<.5,prob>.5)
            evidence=summary(action,wrong,area,f)
            for label,m in [('wrong',area&wrong),('right',area&~wrong)]:
                evidence[label]=dict(rows=int(m.sum()),with_match=int((m&(nbr>0)).sum()),three_bookings=int((m&(nbr>=3)).sum()),
                    mixed_labels=int((m&mixed).sum()),all_neighbors_support_original=int((m&(nbr>0)&(prob==base.astype(float))).sum()))
            info[level]=evidence
        if name in ('awake_paid_negative','sleeping_no_positive'):
            # An illustrative case, selected for proximity to another booking with the opposite truth.
            ix=np.flatnonzero(area&wrong&(nearest>=0)&(y[nearest]!=y)&(distance<=1))
            if len(ix):
                i=ix[np.argmin(distance[ix])]
                fields=['PassengerId','HomePlanet','CryoSleep','Cabin','Destination','Age','VIP',*FEES]
                examples=raw.loc[[i,nearest[i]],fields].copy()
                examples['Transported']=y[[i,nearest[i]]]
                info['closest_opposite_truth_example']=dict(distance=float(distance[i]),passengers=json.loads(examples.to_json(orient='records')))
        report[name]=info
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for k in scopes:
        a=report[k]['close']
        print(k,a['wrong'],a['fixed'],a['harmed'])


if __name__=='__main__':
    run()
