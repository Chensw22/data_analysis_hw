"""Audit a fixed dispute-band edit under paired training regimes and leave-family-out voter panels."""
import json

import numpy as np

from broad_booking_policy_replay_n23 import interval
from family_identity_transfer_n26 import panels_and_reference
from disjoint_tree_fields_n25 import ART,DEST


def gate(p,weights=None):
    weights=np.ones(p.shape[1]) if weights is None else weights
    votes=(p>=.5)@weights
    minority=np.minimum(votes,weights.sum()-votes)
    return (minority>=4*weights.sum()/68-1e-10)&(minority<=16*weights.sum()/68+1e-10)


def compare(challenger,use,lo,hi,y):
    outcomes=[]
    for anchor in (lo,hi):
        edited=np.where(use,challenger,anchor)
        outcomes.append((edited==y).astype(int)-(anchor==y))
    return np.min(outcomes,axis=0),np.max(outcomes,axis=0)


def run():
    y,g,f,rows,names,p,h,_,_,r=panels_and_reference()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        old=z['nine'];hf=z['fold']
    prior,lo,hi=r
    known=prior>=0
    lo=np.where(known,np.where(prior.astype(bool),hi,lo),lo)
    hi=np.where(known,lo,hi)
    assert np.array_equal(lo[known],hi[known])
    tags=np.array([name.split('/')[0] for name in names])
    source=tags=='permutation_selection'
    unseen=~np.isin(g,g[rows])
    by_action={}
    deltas={}
    for tag,ix,seq,low,high,mask,fold in [('historical',rows,h,old,old,np.ones(len(rows),bool),hf),
        ('all',np.arange(len(y)),p,lo,hi,known,f)]:
        challenger=seq[:,source].mean(1)>=.5
        all_gates={'exact':gate(seq),'no_tree_family':gate(seq[:,~source])}
        families=np.unique(tags)
        weights=np.array([1/np.sum(tags==name) for name in tags])
        all_gates['equal_family_band']=gate(seq,weights)
        leave=np.stack([gate(seq[:,tags!=family]) for family in families])
        all_gates['every_leave_one_family']=leave.all(0)
        all_gates['most_leave_one_family']=leave.mean(0)>=.75
        for name,use in all_gates.items():
            a,b=compare(challenger,use,low,high,y[ix])
            deltas[tag,name]=(a,b)
            for key,scope in [(tag,np.ones(len(ix),bool))]+([('new',unseen)] if tag=='all' else []):
                usable=scope&mask
                by_action.setdefault(name,{})[key]=dict(rows=int(usable.sum()),fixed=int((a[usable]>0).sum()),
                    harmed=int((a[usable]<0).sum()),whole_net_bounds=[int(a[scope].sum()),int(b[scope].sum())],
                    net_by_fold=[int(a[usable&(fold==k)].sum()) for k in np.unique(fold)],
                    delta_pp_interval=interval(a[usable],g[ix[usable]]))
    paired={}
    for name in by_action:
        first=deltas['historical',name][0]
        second=deltas['all',name][0][rows]
        m=known[rows]
        paired[name]={'fixed_twice':int(np.sum(m&(first>0)&(second>0))),
                      'harmed_twice':int(np.sum(m&(first<0)&(second<0))),
                      'fixed_either':int(np.sum(m&((first>0)|(second>0)))),
                      'opposite_impact':int(np.sum(m&(first*second<0)))}
    for k,v in by_action.items():
        print(k,[(tag,v[tag]['fixed'],v[tag]['harmed'],v[tag]['whole_net_bounds']) for tag in ('all','historical','new')],paired[k])
    DEST.with_name('dispute_band_stability.json').write_text(json.dumps(dict(by_action=by_action,same_passengers=paired),indent=2)+'\n')


if __name__=='__main__':
    run()
