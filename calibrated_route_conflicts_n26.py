"""Learn whether an unexpected registered itinerary alters the outcome after accounting for current model confidence."""
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from audit_fare_routes_n26 import held_fare
from export_fare_deck_n26 import combine
from fare_route_inference_n26 import DECK, OUT, REPORT
from full_roster_booking_validation_n23 import data
from price_implied_deck_n26 import FEES
from solo_archive_recovery_n26 import ticker
from tabicl_consensus_review_n26 import OUT as REVIEW, sources, splits


def bounded(p):
    return logit(np.clip(p,.01,.99))


def descriptors(raw):
    with np.load(DECK/'deck_estimates.npz') as z:
        np.testing.assert_array_equal(z['passenger'][:len(raw)],raw.PassengerId)
        eligible=z['eligible'][:len(raw)]
        registered=z['label'][:len(raw)]
        deck_matches=z['fine'][:len(raw)].argmax(1)==registered
        deck={k:z[k][:len(raw)] for k in ('coarse','fine','displaced')}
    with np.load(OUT/'routes.npz') as z:
        np.testing.assert_array_equal(z['passenger'][:len(raw)],raw.PassengerId)
        route={k:z[k][:len(raw)] for k in ('broad','exact','displaced')}
    side=raw.Cabin.str.split('/').str[0].fillna('?')
    categories=pd.get_dummies(pd.DataFrame({'deck':side,'dest':raw.Destination.fillna('?'),'home':raw.HomePlanet.fillna('?')}),dtype=float)
    total=raw[FEES].fillna(0).sum(axis=1).to_numpy()
    shares=raw[FEES].fillna(0).to_numpy()/np.maximum(total,1)[:,None]
    basics=np.column_stack([categories,shares,np.log1p(total)/8,raw.Age.fillna(28)/30])
    extras={}
    for label,d,k in [('exact','fine','exact'),('displaced','displaced','displaced')]:
        odds=bounded(deck[d])-bounded(deck['coarse'])
        actual=odds[np.arange(len(raw)),np.maximum(0,registered)]
        actual[registered<0]=0
        change=np.nan_to_num(bounded(route[k][:,:-1])-bounded(route['broad'][:,:-1]))
        lead=deck[d].argmax(1)
        discordance=(lead!=registered)&(registered>=0)
        # The same discrepancy can have different implications for two registered decks.
        paired=np.column_stack([odds[:,j]*(registered==i) for i in np.unique(registered) for j in range(odds.shape[1])])
        features=np.column_stack([odds,actual,discordance,change,paired,
                                  categories.to_numpy()*actual[:,None],
                                  categories.to_numpy()*change[:,-2,None]])
        extras[label]=np.where(eligible[:,None],features,0).clip(-8,8)
    return basics,extras,eligible,deck_matches,registered>=0


def run():
    raw,y,g,_=data()
    panels=sources()[-1]
    basics,extras,eligible,matches,known_deck=descriptors(raw)
    preds={tag:held_fare(v,tag,with_routes=False) for tag,v in panels.items()}
    decisions={}
    for key,src,dst,t,q in splits(panels,g):
        if not (key.startswith('bookingmix') or key=='historical_to_new'):
            continue
        ix=src['ids'][t]
        query=dst['ids'][q]
        assert not np.isin(g[ix],g[query]).any()
        def matrix(v,idx,bank):
            return np.column_stack([basics[v['ids'][idx]],v['off'][idx]*np.where(v['anchor'][idx],-1,1),
                                bounded(bank['neural_deck_raw'][idx]),bounded(bank['neural_deck_fine'][idx]),
                                bounded(bank['tree_deck_fine'][idx]),v['common'][idx]])
        xt=matrix(src,t,preds['historical'])
        xq=matrix(dst,q,preds['full'])
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],query)
            review=dst['anchor'][q]^((z['identities_error']>.5)&(z['identities_outcome']>.5))
        parent=combine(review,preds['full']['neural_deck_raw'][q],preds['full']['neural_deck_fine'][q],eligible[query])>=.5
        votes={}
        for mode in ('base','exact','displaced'):
            a=xt if mode=='base' else np.column_stack([xt,extras[mode][ix]])
            b=xq if mode=='base' else np.column_stack([xq,extras[mode][query]])
            # All source and query rows are paid subjects and have a manifest-derived fare estimate.
            fit=LogisticRegression(C=.12,max_iter=700).fit(a[eligible[ix]],y[ix][eligible[ix]])
            votes[mode]=fit.predict(b)
        actions={m:eligible[query]&(votes[m]!=parent)&(votes['base']==parent) for m in ('exact','displaced')}
        name='bookingmix' if key.startswith('bookingmix') else key
        decisions.setdefault(name,[]).append((query,parent!=y[query],dst['common'][q],actions))
    report={}
    for name,items in decisions.items():
        ids,wrong,common=[np.concatenate([item[i] for item in items]) for i in range(3)]
        assert len(ids)==len(np.unique(ids))
        masks={'all':np.ones(len(ids),bool),'unanimous':common,
               'unanimous_deck_agrees':common&eligible[ids]&matches[ids],
               'unanimous_deck_differs':common&eligible[ids]&known_deck[ids]&~matches[ids]}
        report[name]={k:{s:ticker(mask,wrong,np.concatenate([v[3][k] for v in items])) for s,mask in masks.items()}
                       for k in ('exact','displaced')}
    REPORT.with_name('calibrated_route_conflicts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__':
    run()
