"""Compare the direction of the fare-feature effect across three disjoint labeled booking banks."""
import argparse
import hashlib

import numpy as np
import pandas as pd

from full_roster_booking_validation_n23 import ROOT, data
from price_implied_deck_n26 import OUT as DECK, build_features

ART = ROOT/'artifacts'
OUT = ART/'nov26_independent_fare_bookings'
REPORT = ROOT/'reports/nov23_matched_learning_curve/independent_fare_bookings.json'


def roster():
    raw,y,g,f = data()
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    people = pd.concat([raw,test],ignore_index=True)
    tg=test.PassengerId.str.split('_').str[0].to_numpy()
    assert not np.isin(g,tg).any()
    assign=np.fromiter((int.from_bytes(hashlib.sha256(('fare-banks:'+s).encode()).digest()[:4],'big')%4 for s in tg),int,len(tg))
    return people,y,np.r_[g,tg],np.r_[f%4,assign],f


def train():
    import torch
    from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    raw,y,g,bank,_=roster()
    design=build_features(raw,g)
    OUT.mkdir(exist_ok=True)
    for i in range(4):
        t=np.flatnonzero(bank[:len(y)]==i)
        q=np.flatnonzero(bank!=i)
        assert not np.isin(g[t],g[q]).any()
        path=OUT/f'bank{i}.npz'
        if path.exists():
            continue
        probs={k:infer(design[k],y,t,q,7181+i) for k in ['raw','fine']}
        assert all(np.isfinite(p).all() and (p>=0).all() and (p<=1).all() for p in probs.values())
        tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,t=t,q=q,train_y=y[t],ids=raw.PassengerId.iloc[q].to_numpy(str),weight=DIGEST,**probs)
        tmp.replace(path)
        print('bank',i,len(t),len(q),flush=True)


def report():
    import json
    from arbitration_against_scored_nine_n26 import champions
    from exhaustive_family_quorums_n26 import summary
    from held_expert_portfolios_n26 import SIX,panel_predictions,sources
    from rebase_fare_six_n26 import NAME, predict
    from six_way_against_scored_nine_n26 import test_decisions
    from forecast_frozen_queue_n26 import NAMES
    people,y,g,bank,f=roster()
    views=sources()[-1]
    v=views['full']
    p=panel_predictions(v,'full')
    a,known,_,hi=champions()
    with np.load(DECK/'deck_estimates.npz') as z:
        np.testing.assert_array_equal(z['passenger'],people.PassengerId)
        eligible=z['eligible']
    parent=predict(a,p['neural_deck_raw'],p['neural_deck_fine'],np.column_stack([p[n] for n in SIX]),v['votes'][:,:18],eligible[:len(y)])['joint']>=.5
    test_ids=people.PassengerId.iloc[len(y):]
    incumbent=np.r_[a,test_decisions(NAMES['nine'],test_ids)[0]]
    base=np.r_[parent,test_decisions(NAME,test_ids)[0]]
    original=np.full((len(base),4),np.nan)
    fine=original.copy()
    for i in range(4):
        t,q=np.flatnonzero(bank[:len(y)]==i),np.flatnonzero(bank!=i)
        with np.load(OUT/f'bank{i}.npz') as z:
            for key,val in [('t',t),('q',q),('train_y',y[t]),('ids',people.PassengerId.iloc[q].to_numpy(str))]:
                np.testing.assert_array_equal(z[key],val)
            assert not np.isin(g[t],g[q]).any()
            original[q,i]=z['raw']
            fine[q,i]=z['fine']
    assert (np.isfinite(original).sum(1)==3).all() and np.array_equal(np.isfinite(original),np.isfinite(fine))
    sign=np.where(base,-1,1)[:,None]
    raw_oppose=((original>=.5)!=base[:,None])&np.isfinite(original)
    oppose=((fine>=.5)!=base[:,None])&np.isfinite(fine)
    support=oppose&((fine-original)*sign>0)
    paid=eligible & (base==incumbent)
    raw_majority=paid&(raw_oppose.sum(1)>=2)
    votes=dict(raw_majority=raw_majority, fine_majority=paid&(oppose.sum(1)>=2), directional_majority=paid&(support.sum(1)>=2),
               direction_unanimous=paid&(support.sum(1)==3))
    fresh=~np.isin(g[:len(y)],g[hi])
    masks=dict(all=known,new_orders=known&fresh,earlier_agreement=known&v['common']&(v['anchor']==a))
    out={}
    for key,act in votes.items():
        r={scope:summary(act[:len(y)],parent!=y,m,f) for scope,m in masks.items()}
        r['extra_over_raw_majority']=summary((act&~raw_majority)[:len(y)],parent!=y,known,f)
        r['vs_raw_majority']=summary((act^raw_majority)[:len(y)],(parent^raw_majority[:len(y)])!=y,known,f)
        decisions=base[len(y):]^act[len(y):]
        r['test_added_changes']=int(act[len(y):].sum())
        r['test_extra_over_raw_majority']=int((act&~raw_majority)[len(y):].sum())
        r['total_test_changes_from_champion']=int((decisions!=incumbent[len(y):]).sum())
        out[key]=r
    REPORT.write_text(json.dumps(dict(results=out,base=NAME,
        scope='Only paid passengers unchanged by the frozen 111-edit policy. Each query uses three equally weighted disjoint reference banks; the held group never appears in any of those three. Test bookings use fixed SHA256 assignment to match the training support sizes.'),indent=2)+'\n')
    print([(k,v['all']['fixed'],v['all']['harmed'],v['new_orders']['net'],v['test_added_changes']) for k,v in out.items()],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','report'])
    if parser.parse_args().action=='train':
        train()
    else:
        report()
