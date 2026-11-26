"""Cross-check own-versus-peer Nori evidence with same-booking versus substitute-fare specialists."""
import json

import numpy as np
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import champions
from exhaustive_family_quorums_n26 import inputs, summary
from full_roster_booking_validation_n23 import ROOT,data
from nori_fare_party_n26 import ART as NORI
from unpaid_companion_fare_n26 import OUT as SPECIALISTS,design


def check(truth, check_groups, folds, passenger, q, t, packet):
    for key,val in [('t',t),('q',q),('passenger',passenger[q])]:
        np.testing.assert_array_equal(packet[key],val)
    assert not np.isin(check_groups[q],check_groups[t]).any()
    assert len(np.unique(folds[q]))==1
    label_key='labels' if 'labels' in packet else 'train_y'
    np.testing.assert_array_equal(packet[label_key],truth[t])


def run():
    raw,y,g,f=data()
    _,cohort=design(raw,g)
    ids=raw.PassengerId.to_numpy(str)
    parent,known,_,hi=champions()
    _,cur,_=inputs()
    sign=np.where(parent,-1.,1.)
    results={key:np.zeros(len(y),bool) for key in ['absolute_two_families','confirmed_price_contrasts','all_three_price_contrasts']}
    for k in np.unique(f):
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        st,sq=t[cohort[t]],q[cohort[q]]
        nori={}
        for mode in ['own','peer']:
            with np.load(NORI/f'{mode}{k}.npz') as z:
                check(y,g,f,ids,q,t,z)
                nori[mode]=z['p'][cohort[q]]
        opposes=(nori['peer']>=.5)!=parent[sq]
        moved=(nori['peer']-nori['own'])*sign[sq]>0
        count=np.zeros(len(sq),int)
        absolute=np.zeros(len(sq),int)
        for arch in ['tree','tabicl']:
            with np.load(SPECIALISTS/f'{arch}{k}.npz') as z:
                check(y,g,f,ids,sq,st,z)
                actual=z[arch+'_linked']
                other=z[arch+'_matched_other']
                against=(actual>=.5)!=parent[sq]
                absolute+=against
                count+=against&((actual-other)*sign[sq]>0)
        results['absolute_two_families'][sq]=opposes&(absolute>=1)
        results['confirmed_price_contrasts'][sq]=opposes&moved&(count>=1)
        results['all_three_price_contrasts'][sq]=opposes&moved&(count==2)
    scopes=dict(full=known&cohort,common=cur['common']&cohort,fresh=known&cohort&~np.isin(g,g[hi]))
    out={key:{n:summary(a,parent!=y,m,f) for n,m in scopes.items()} for key,a in results.items()}
    path=ROOT/'reports/nov23_matched_learning_curve/corroborate_unpaid_fare.json'
    path.write_text(json.dumps(out,indent=2)+'\n')
    print({k:(v['full']['fixed'],v['full']['harmed'],v['common']['net']) for k,v in out.items()})


if __name__=='__main__':
    with threadpool_limits(1):
        run()
