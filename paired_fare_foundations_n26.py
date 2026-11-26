"""Check whether two distinct foundation models support the same additional fare corrections."""
import json
import numpy as np
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import champions
from exhaustive_family_quorums_n26 import summary
from full_roster_booking_validation_n23 import data,ROOT
from held_expert_portfolios_n26 import DECK,SIX,panel_predictions,sources
from nori_booking_regression_n26 import packet_dir
from nori_fare_party_n26 import ART as NORI
from rebase_fare_six_n26 import predict


def policies(anchor,enabled,raw,fine):
    sign=np.where(anchor,-1.,1.)[:,None]
    oppose=(fine>=.5)!=anchor[:,None]
    raw_oppose=(raw>=.5)!=anchor[:,None]
    shifted=(fine-raw)*sign>0
    both=enabled & oppose.all(1)
    return dict(raw_pair=enabled & raw_oppose.all(1),fare_pair=both,
                both_raised=both & shifted.all(1),at_least_one_crossed=both & ~raw_oppose.all(1))


def evaluate():
    people,y,g,f=data()
    v=sources()[-1]['full']
    p=panel_predictions(v,'full')
    a,known,_,hi=champions()
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible'][:len(y)]
    anchor=predict(a,p['neural_deck_raw'],p['neural_deck_fine'],np.column_stack([p[n] for n in SIX]),v['votes'][:,:18],enabled)['joint']>=.5
    raw_nori=np.empty(len(y))
    own_nori=np.empty(len(y))
    for k in np.unique(f):
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        assert not np.isin(g[t],g[q]).any()
        for path,key,dest in [(packet_dir(0)/f'fold{k}.npz','ids',raw_nori),(NORI/f'own{k}.npz','passenger',own_nori)]:
            with np.load(path) as z:
                for col,expected in [('t',t),('q',q),(key,people.PassengerId.iloc[q].to_numpy(str)),('labels',y[t])]:
                    np.testing.assert_array_equal(z[col],expected)
                dest[q]=z['p']
    raw=np.column_stack([p['neural_deck_raw'],raw_nori])
    fine=np.column_stack([p['neural_deck_fine'],own_nori])
    assert np.isfinite(raw).all() and np.isfinite(fine).all()
    acts=policies(anchor,enabled&(a==anchor),raw,fine)
    masks=dict(all=known,fresh=known&~np.isin(g,g[hi]),earlier_agreement=known&v['common']&(v['anchor']==a))
    out={}
    for key,act in acts.items():
        out[key]={n:summary(act,y!=anchor,m,f) for n,m in masks.items()}
        out[key]['additional_vs_raw_pair']=summary(act&~acts['raw_pair'],y!=anchor,known,f)
    (ROOT/'reports/nov23_matched_learning_curve/paired_fare_foundations.json').write_text(json.dumps(out,indent=2)+'\n')
    print({k:[v['all']['fixed'],v['all']['harmed'],v['fresh']['net']] for k,v in out.items()})


if __name__=='__main__':
    with threadpool_limits(1):
        evaluate()
