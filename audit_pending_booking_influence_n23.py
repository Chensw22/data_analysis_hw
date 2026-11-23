"""Reconstruct eight pending decisions and locate their paired net gains by whole booking."""
import json
import numpy as np
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write
from pending_submission_diversity_n23 import DONOR,PAIR,PAIRS
from service_pair_change_budget_n23 import changed
from service_pair_portfolio_n23 import NAMES,actions,replay_votes
from zero_donor_pair_overlay_n23 import additional

REPORT=ROOT/'reports/nov23_booking_influence'


def votes_for(frame,groups,row,y,seed,weight):
    ref,views=replay_votes(frame,groups,row,y,seed,weight)
    results=dict(zip(NAMES,views.T))
    rules=actions(views!=ref[:,None])
    with np.load(ART/f'nov23_service_pairs/{seed}.npz') as p, np.load(ART/f'nov23_zero_spend_analogues/{seed}.npz') as d:
        for k in ('row','y','base','ref','fold'):
            np.testing.assert_array_equal(p[k],d[k])
        np.testing.assert_array_equal(d['row'],row)
        np.testing.assert_array_equal(d['y'],y)
        np.testing.assert_array_equal(d['ref'],ref)
        extra=additional(d['base'],ref,views[:,-1],d['matched'],row,.10)
        results[DONOR]=views[:,-1]|extra
        results[PAIR+'quarter']=changed(p['same_passenger_both'],p['base'],ref,.25)
    results[PAIR+'corroborated']=ref^rules['pair_plus_corroborated']
    return ref,results


def main():
    row,y,frame,g,_=load()
    _,index=np.unique(g,return_inverse=True)
    size=frame.BookingSize.to_numpy()
    previous=json.loads((ROOT/'reports/nov23_recent_family_transfer/pending_diversity.json').read_text())
    weight=json.loads((PAIRS/'test_budget.json').read_text())['test']['change_150']['weight']
    plans,history={},{}
    for k,seed in enumerate([*map(str,SEEDS),'v131']):
        ref,all_votes=votes_for(frame,g,row,y,seed,weight)
        reference=ref==y
        donor=all_votes[DONOR]==y
        out={}
        for name,v in all_votes.items():
            hit=v==y
            fixed=hit&~reference
            harmed=~hit&reference
            delta=fixed.astype(int)-harmed
            booked=np.bincount(index,weights=delta).astype(int)
            positive=np.sort(booked[booked>0])[::-1]
            net=int(booked.sum())
            assert net==previous['files'][name]['net_correct'][k]
            assert net>0 and positive.size
            out[name]=dict(fixed=int(fixed.sum()),harmed=int(harmed.sum()),net=net,
                           positive_bookings=len(positive),negative_bookings=int((booked<0).sum()),
                           largest_gain=int(positive[0]),worst_leave_one_net=net-int(positive[0]),
                           min_beneficial_bookings_to_erase_net=int(np.searchsorted(np.cumsum(positive),net)+1),
                           net_by_party_size=[int(delta[size==1].sum()),int(delta[size==2].sum()),int(delta[size>=3].sum())],
                           paired_versus_160=dict(only_this_correct=int((hit&~donor).sum()),only_160_correct=int((~hit&donor).sum())),
                           new_fixes_outside_160=int((fixed&~donor).sum()))
            if seed!='v131':history.setdefault(name,[]).append(delta)
        plans[seed]=out
    stable={}
    for name,h in history.items():
        a=np.stack(h)
        fixed=(a==1).all(0)
        harmed=(a==-1).all(0)
        donor=np.stack(history[DONOR])
        stable[name]=dict(same_passenger_fixed=int(fixed.sum()),same_passenger_harmed=int(harmed.sum()),
                          stable_fixes_160_never_fixes=int((fixed&~(donor==1).any(0)).sum()),
                          per_size_net_across_three=[int(a[:,scope].sum()) for scope in (size==1,size==2,size>=3)])
    write(REPORT/'influence.json',dict(bookings=int(index.max()+1),plans=plans,stable=stable))
    for name,s in stable.items(): print(name,s,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        main()
