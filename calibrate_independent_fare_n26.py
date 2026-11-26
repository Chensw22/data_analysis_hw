"""Fit the fare signal on held bookings never used by any of the three reference models."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import champions
from exhaustive_family_quorums_n26 import summary
from forecast_frozen_queue_n26 import NAMES
from held_expert_portfolios_n26 import SIX, panel_predictions, sources
from independent_fare_bookings_n26 import ART, DECK, OUT, REPORT, roster
from rebase_fare_six_n26 import NAME, predict
from six_way_against_scored_nine_n26 import test_decisions


def features(people,y,g,bank):
    raw=np.full((len(people),4),np.nan)
    fare=raw.copy()
    for i in range(4):
        t,q=np.flatnonzero(bank[:len(y)]==i),np.flatnonzero(bank!=i)
        with np.load(OUT/f'bank{i}.npz') as z:
            for key,truth in [('t',t),('q',q),('ids',people.PassengerId.iloc[q].to_numpy(str)),('train_y',y[t])]:
                np.testing.assert_array_equal(z[key],truth)
            assert not np.isin(g[q],g[t]).any()
            raw[q,i]=z['raw']
            fare[q,i]=z['fine']
    keep=np.isfinite(raw)
    assert np.array_equal(keep,np.isfinite(fare)) and (keep.sum(1)==3).all()
    r=logit(raw[keep].reshape(-1,3).clip(.003,.997))
    f=logit(fare[keep].reshape(-1,3).clip(.003,.997))
    step=f-r
    basic=np.column_stack([r.mean(1),r.std(1),np.median(r,axis=1)])
    extra=np.column_stack([step.mean(1),step.std(1),np.median(step,axis=1)])
    return basic,np.column_stack([basic,extra])


def calibrate(y,g,bank,eligible,basic,full,ordered):
    out=np.full((len(bank),2),np.nan)
    for i in range(4):
        patients=np.flatnonzero((bank[:len(y)]==i)&eligible[:len(y)])
        target=np.flatnonzero((bank==i)&eligible&(np.arange(len(bank))>=len(y)))
        if ordered:
            blocks=np.array_split(np.unique(g[patients]),4)
            splits=[(patients[~np.isin(g[patients],b)], patients[np.isin(g[patients],b)]) for b in blocks]
        else:
            parts=GroupKFold(4,shuffle=True,random_state=9137+i).split(patients,groups=g[patients])
            splits=[(patients[a],patients[b]) for a,b in parts]
        for t,q in [*splits,(patients,target)]:
            # Every model supplying a feature here used only a different bank for fitting.
            assert not np.isin(g[t],g[q]).any() and np.all(bank[t]==i) and np.all(bank[q]==i)
            for k,x in enumerate([basic,full]):
                assert np.isnan(out[q,k]).all()
                out[q,k]=LogisticRegression(C=.2,max_iter=400).fit(x[t],y[t]).predict_proba(x[q])[:,1]
    assert np.array_equal(np.isfinite(out).all(1),eligible)
    return out


def run():
    people,y,g,bank,fold=roster()
    basic,full=features(people,y,g,bank)
    v=sources()[-1]['full']
    p=panel_predictions(v,'full')
    a,known,_,hi=champions()
    ids=people.PassengerId.iloc[len(y):]
    with np.load(DECK/'deck_estimates.npz') as z:
        np.testing.assert_array_equal(z['passenger'],people.PassengerId)
        eligible=z['eligible']
    initial=predict(a,p['neural_deck_raw'],p['neural_deck_fine'],np.column_stack([p[n] for n in SIX]),v['votes'][:,:18],eligible[:len(y)])['joint']>=.5
    anchor=np.r_[initial,test_decisions(NAME,ids)[0]]
    incumbent=np.r_[a,test_decisions(NAMES['nine'],ids)[0]]
    scopes=dict(all=known,fresh=known&~np.isin(g[:len(y)],g[hi]),earlier_agreement=known&v['common']&(v['anchor']==a))
    result={}
    for order in (False,True):
        signals=calibrate(y,g,bank,eligible,basic,full,order)
        raw,fine=(signals>=.5).T
        unique=eligible&(anchor==incumbent)&(raw==anchor)&(fine!=anchor)
        candidate=anchor^unique
        info={scope:summary(unique[:len(y)],initial!=y,m,fold) for scope,m in scopes.items()}
        info['test_additions']=int(unique[len(y):].sum())
        info['test_changes_from_incumbent']=int((candidate[len(y):]!=incumbent[len(y):]).sum())
        info['held_paid_comparison']=summary(fine[:len(y)]!=raw[:len(y)],raw[:len(y)]!=y,eligible[:len(y)],fold)
        result['ordered' if order else 'random']=info
    (REPORT.parent/'calibrate_independent_fare.json').write_text(json.dumps(result,indent=2)+'\n')
    print({n:(d['all']['fixed'],d['all']['harmed'],d['fresh']['net'],d['test_additions']) for n,d in result.items()})


if __name__=='__main__':
    with threadpool_limits(1):
        run()
