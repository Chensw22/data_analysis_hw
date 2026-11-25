"""Locate remaining errors of the queued fare model by family dissent and independent challengers."""
import json
import math
import numpy as np
from audit_fare_routes_n26 import DECK,reviews
from export_fare_deck_n26 import combine
from fare_registration_disputes_n26 import panel_predictions
from full_roster_booking_validation_n23 import ROOT
from solo_archive_recovery_n26 import ticker
from tabicl_consensus_review_n26 import sources


def run():
    _,y,g,panels=sources()
    saved={tag:panel_predictions(v,tag) for tag,v in panels.items()}
    with np.load(DECK/'deck_estimates.npz') as z:
        eligible=z['eligible']
    report={}
    for tag,q,review in reviews(panels,g):
        base='historical' if tag=='historical' else 'full'
        v,p=panels[base],saved[base]
        ids=v['ids'][q]
        fare=combine(review,p['neural_deck_raw'][q],p['neural_deck_fine'][q],eligible[ids])>=.5
        wrong=fare!=y[ids]
        opposed=(v['votes'][q,:18]!=fare[:,None]).sum(1)
        original_common=v['common'][q]
        def count(names):
            return np.column_stack([((p[n][q]>=.5)!=fare) for n in names]).sum(1)
        science=count(['neural_route_exact','tree_route_exact','swap'])
        reg=count(['register_all','register_paid','sequence_paid'])
        many=eligible[ids]&(reg>=2)&(science>=2)
        all_six=eligible[ids]&(reg==3)&(science==3)
        masks={'previous_unanimity':original_common,'other_no_family_dissent':~original_common&(opposed==0),
               'other_one_to_four':~original_common&(opposed>=1)&(opposed<=4),
               'other_five_to_nine':~original_common&(opposed>=5)&(opposed<=9),
               'other_ten_or_more':~original_common&(opposed>=10)}
        assert np.column_stack(list(masks.values())).sum(1).min()==1
        assert sum(m.sum() for m in masks.values())==len(q)
        actions={'two_majorities':many,'six_votes':all_six,'family_majority':opposed>=10,
                 'family_and_two_majorities':many&(opposed>=10),'family_and_six_votes':all_six&(opposed>=10)}
        r={k:{n:ticker(m,wrong,a) for n,a in actions.items()} for k,m in dict(all=np.ones(len(q),bool),**masks).items()}
        chosen=actions['family_and_six_votes']
        gain=np.where(chosen,np.where(wrong,1,-1),0)
        r['selected_evaluation']=dict(before=float((~wrong).mean()),
            after=float((~wrong).mean()+gain.mean()),undo_fare=ticker(fare!=review,wrong,chosen),
            additional=ticker(fare==review,wrong,chosen),
            fold_net=[int(gain[v['fold'][q]==k].sum()) for k in np.unique(v['fold'][q])])
        orders,inv,size=np.unique(g[ids],return_inverse=True,return_counts=True)
        group_net=np.bincount(inv,weights=gain)
        rng=np.random.default_rng(8194)
        draws=[]
        for _ in range(2000):
            sample=rng.integers(0,len(orders),len(orders))
            draws.append(group_net[sample].sum()/size[sample].sum())
        r['selected_evaluation']['booking_bootstrap_95']=np.quantile(draws,[.025,.975]).tolist()
        r['extra_correct_needed_to_exceed_85']=max(0,math.floor(len(ids)*.85)+1-int((~wrong).sum()))
        report[tag]=r
        if tag=='bookingmix':
            print([(k,v['two_majorities']) for k,v in r.items() if isinstance(v,dict) and 'two_majorities' in v],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/surviving_loss_by_vote.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
