"""Measure coverage and agreement from three distinct receipt interventions against the queued release."""
import json
import numpy as np

from audit_fare_routes_n26 import DECK, held_fare, reviews
from export_fare_deck_n26 import combine
from full_roster_booking_validation_n23 import ROOT
from receipt_charge_swaps_n26 import OUT as SWAPS
from solo_archive_recovery_n26 import ticker
from tabicl_consensus_review_n26 import sources


def run():
    _,y,g,panels=sources()
    store={}
    with np.load(DECK/'deck_estimates.npz') as z:
        eligible=z['eligible']
        agrees=(z['label']>=0)&(z['label']==z['fine'].argmax(1))
    with np.load(SWAPS/'scores.npz') as z:
        replaceable=np.isfinite(z['exact']).any(axis=1)
    for tag,v in panels.items():
        p=held_fare(v,tag)
        swaps=held_fare(v,tag,directory=SWAPS,architectures=('tree',))
        with np.load(ROOT/f'artifacts/nov26_fare_counterfactuals/{tag}.npz') as cf:
            np.testing.assert_array_equal(v['ids'],cf['ids'])
            p['replaced_deck']=cf['Deck_exact']
            p['replaced_destination']=cf['Destination_exact']
        p['swap']=swaps['tree_route_exact']
        store[tag]=p
    report={}
    for tag,q,review in reviews(panels,g):
        name='historical' if tag=='historical' else 'full'
        p,v=store[name],panels[name]
        ids=v['ids'][q]
        fare=combine(review,p['neural_deck_raw'][q],p['neural_deck_fine'][q],eligible[ids])>=.5
        def moves(before,after):
            return ((p[before][q]>=.5)==fare)&((p[after][q]>=.5)!=fare)&eligible[ids]
        route=moves('neural_deck_fine','neural_route_exact')|moves('tree_deck_fine','tree_route_exact')
        swap=moves('tree_deck_fine','swap')&replaceable[ids]
        replace=moves('tree_deck_raw','replaced_deck')|moves('tree_deck_raw','replaced_destination')
        count=np.stack([route,swap,replace]).sum(0)
        masks={'all':np.ones(len(ids),bool), 'consensus':v['common'][q],
               'consensus_paid_agree':v['common'][q]&eligible[ids]&agrees[ids],
               'remaining_original_consensus_errors':v['common'][q]&(review!=y[ids])&(fare!=y[ids])}
        actions={'one_or_more':count>=1,'two_or_more':count>=2,'all_three':count==3}
        # Direct votes expose correct predictions that incremental feature-change rules can hide.
        named=['neural_deck_raw','tree_deck_raw','neural_deck_fine','tree_deck_fine',
               'neural_route_exact','tree_route_exact','swap']
        external=np.column_stack([(p[col][q]>=.5)!=fare for col in named])
        any_paid=eligible[ids]
        actions['any_direct_vote']=external.any(1)&any_paid
        actions['all_direct_votes']=external.all(1)&any_paid
        actions['direct_majority']=(external.sum(1)>=4)&any_paid
        actions['all_three_new_direct']=external[:,-3:].all(1)&replaceable[ids]&any_paid
        report[tag]={k:{subset:ticker(m,fare!=y[ids],a) for subset,m in masks.items()} for k,a in actions.items()}
    (ROOT/'reports/nov23_matched_learning_curve/fare_error_committee.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['bookingmix'],flush=True)


if __name__=='__main__':
    run()
