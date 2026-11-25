"""Re-evaluate transport after integrating over fare-implied alternatives to one recorded field."""
import json
import numpy as np
import pandas as pd
from audit_fare_routes_n26 import DECK, OUT as ROUTE, held_fare, reviews
from export_fare_deck_n26 import combine
from full_roster_booking_validation_n23 import data, ROOT
from price_implied_deck_n26 import DECKS, build_features
from solo_archive_recovery_n26 import ticker
from solo_selection_likelihood_n26 import fit_model
from tabicl_consensus_review_n26 import sources


def run():
    raw,y,g,_=data()
    x=build_features(raw,g)['raw']
    panels=sources()[-1]
    saved={}
    with np.load(DECK/'deck_estimates.npz') as deck, np.load(ROUTE/'routes.npz') as route:
        levels=sorted(raw.Destination.dropna().unique())
        assert list(route['columns'][:len(levels)])==['route_to_'+d for d in levels]
        np.testing.assert_array_equal(deck['passenger'],route['passenger'])
        eligible=deck['eligible']
        known=deck['label']>=0
        agrees=(deck['fine'].argmax(1)==deck['label'])&known
        for tag,v in panels.items():
            p=held_fare(v,tag,with_routes=False)
            reference=p['tree_deck_raw']
            modes={}
            for k in np.unique(v['fold']):
                location=np.flatnonzero(v['fold']==k)
                with np.load(DECK/f'tree_{tag}_{k}.npz') as z:
                    t,q=z['t'],z['q']
                    np.testing.assert_array_equal(v['ids'][location],q)
                assert not np.isin(g[t],g[q]).any()
                model=fit_model(x.iloc[t],y[t])
                frame=x.iloc[q]
                np.testing.assert_allclose(model.predict_proba(frame)[:,1],reference[location],atol=1e-12,rtol=0)
                for name,values,posterior in [
                    ('Deck',DECKS,{'exact':deck['fine'][q], 'wide':deck['coarse'][q], 'displaced':deck['displaced'][q]}),
                    ('Destination',levels,{n:route[key][q,:len(levels)] for n,key in [('exact','exact'),('wide','broad'),('displaced','displaced')]})
                ]:
                    probabilities=[]
                    for field in values:
                        alt=frame.assign(**{name:field})
                        probabilities.append(model.predict_proba(alt)[:,1])
                    grid=np.column_stack(probabilities)
                    for version,weights in posterior.items():
                        np.testing.assert_allclose(weights[eligible[q]].sum(1),1)
                        column=modes.setdefault(name+'_'+version,np.zeros(len(v['ids'])))
                        column[location]=np.sum(weights*grid,axis=1)
            target=ROOT/'artifacts/nov26_fare_counterfactuals'
            target.mkdir(exist_ok=True)
            np.savez_compressed(target/f'{tag}.npz',ids=v['ids'],**modes)
            saved[tag]=p,modes
    report={}
    for tag,ix,r in reviews(panels,g):
        panel='historical' if tag=='historical' else 'full'
        p,modes=saved[panel]
        v=panels[panel]
        ids=v['ids'][ix]
        fare=combine(r,p['neural_deck_raw'][ix],p['neural_deck_fine'][ix],eligible[ids])>=.5
        baseline=p['tree_deck_raw'][ix]>=.5
        masks={'all':np.ones(len(ix),bool),'consensus':v['common'][ix],
               'consensus_agreeing_deck':v['common'][ix]&agrees[ids]&eligible[ids]}
        acts={n:(base[ix]>=.5)!=fare for n,base in modes.items()}
        acts['both_exact']=acts['Deck_exact']&acts['Destination_exact']
        report[tag]={n:{key:ticker(mask,fare!=y[ids],a&(baseline==fare)&eligible[ids])
                         for key,mask in masks.items()} for n,a in acts.items()}
    path=ROOT/'reports/nov23_matched_learning_curve/fare_counterfactual_registration.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print({k:v['all'] for k,v in report['bookingmix'].items()},flush=True)


if __name__=='__main__':
    run()
