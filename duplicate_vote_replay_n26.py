"""Reproduce the identical deployed basket/pooled votes in the historical and current dispute gate."""
import json
import numpy as np
from disjoint_tree_fields_n25 import ART,DEST
from family_identity_transfer_n26 import panels_and_reference
from dispute_band_stability_n26 import gate


def run():
    y,g,f,rows,names,p,h,_,_,(parent,lo,hi)=panels_and_reference()
    a,b=[names.index('spending_shape/'+n) for n in ('pooled_direction','basket_direction')]
    tree=[i for i,n in enumerate(names) if n.startswith('permutation_selection/')]
    with np.load(ART/'proposal_n26_dispute_band/frozen_votes.npz') as z:
        maxdiff=float(np.max(np.abs(z['p'][:,a]-z['p'][:,b])))
    assert maxdiff==0
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        old=z['nine']
    known=parent>=0
    new=~np.isin(g,g[rows])
    reference=np.where(parent,hi,lo)
    result={}
    for tag,scores,anchor,target,mask in [('historical',h,old,y[rows],np.ones(len(rows),bool)),
        ('all_known',p,reference,y,known),('unseen',p,reference,y,known&new)]:
        original=gate(scores)
        out={'identical_prob_fraction':float(np.mean(np.abs(scores[mask,a]-scores[mask,b])<1e-8))}
        candidate=scores[:,tree].mean(1)>=.5
        for policy,src,dest in [('original',a,a),('copy_pooled',a,b),('copy_basket',b,a)]:
            v=scores.copy()
            v[:,dest]=v[:,src]
            support=gate(v)
            flip=support & mask & (candidate!=anchor)
            out[policy]=dict(changed_band=int(np.sum(mask&(support!=original))),
                changed_decisions=int(np.sum(mask&(support!=original)&(candidate!=anchor))),
                fixed=int(np.sum(flip&(target==candidate))),harmed=int(np.sum(flip&(target!=candidate))))
        result[tag]=out
    DEST.with_name('duplicate_vote_replay.json').write_text(json.dumps(dict(test_max_abs_diff=maxdiff,results=result),indent=2)+'\n')
    print(result)


if __name__=='__main__':
    run()
