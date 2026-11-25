"""Count who benefits from the frozen fare-deck model, including people outside its coverage."""
import json

import numpy as np

from audit_fare_routes_n26 import DECK, held_fare, reviews
from export_fare_deck_n26 import combine
from full_roster_booking_validation_n23 import ROOT, data
from price_implied_deck_n26 import FEES
from solo_archive_recovery_n26 import ticker
from tabicl_consensus_review_n26 import sources


def run():
    raw,y,g,_=data()
    views=sources()[-1]
    bank={name:held_fare(v,name,with_routes=False) for name,v in views.items()}
    with np.load(DECK/'deck_estimates.npz') as z:
        eligible=z['eligible']
        deck=z['label']
        supported=z['fine'].argmax(1)==deck
    total=raw[FEES].fillna(0).sum(axis=1).to_numpy()
    report={}
    for name,q,parent in reviews(views,g):
        tag='historical' if name=='historical' else 'full'
        ids=views[tag]['ids'][q]
        p=bank[tag]
        wrong=parent!=y[ids]
        new=combine(parent,p['neural_deck_raw'][q],p['neural_deck_fine'][q],eligible[ids])>=.5
        common=views[tag]['common'][q]
        base=ticker(common,wrong,new!=parent)
        details={}
        for key,mask in [('complete_positive',eligible[ids]),('some_positive_incomplete',(total[ids]>0)&~eligible[ids]),
                         ('no_visible_positive',total[ids]==0),('original_true',parent),('original_false',~parent),
                         ('paid_low',eligible[ids]&(total[ids]<650)),('paid_650_999',eligible[ids]&(total[ids]>=650)&(total[ids]<1000)),
                         ('paid_1000_plus',eligible[ids]&(total[ids]>=1000)),
                         ('paid_registered_deck_leads',eligible[ids]&(deck[ids]>=0)&supported[ids]),
                         ('paid_different_deck_leads',eligible[ids]&(deck[ids]>=0)&~supported[ids]),
                         ('paid_missing_deck',eligible[ids]&(deck[ids]<0))]:
            part=ticker(common&mask,wrong,new!=parent)
            part['errors_after']=part['errors']-part['fixed']+part['harmed']
            details[key]=part
        for keys in [('complete_positive','some_positive_incomplete','no_visible_positive'),('original_true','original_false')]:
            assert all(sum(details[k][n] for k in keys)==base[n] for n in base)
        report[name]=dict(total=base,by_group=details)
    destination=ROOT/'reports/nov23_matched_learning_curve/fare_deck_rescue_profile.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(report['bookingmix'])


if __name__=='__main__':
    run()
