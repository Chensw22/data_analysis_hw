"""Measure the ceiling for lookup by hard votes, then try to recover its gains on unseen bookings."""
import json

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART, DEST
from solo_archive_recovery_n26 import archive,ticker
from solo_selection_likelihood_n26 import regimes


def ceiling(bits, truth, anchor, mask):
    # The majority of the evaluated labels is an optimistic diagnostic, never a deployable lookup.
    signature=np.packbits(np.column_stack([anchor,bits])[mask],axis=1)
    _,index,n=np.unique(signature,axis=0,return_inverse=True,return_counts=True)
    yes=np.bincount(index,weights=truth[mask],minlength=len(n)).astype(int)
    mixed=(yes>0)&(yes<n)
    errors=int(np.minimum(yes,n-yes).sum())
    return dict(rows=int(mask.sum()),signatures=len(n),singletons=int((n==1).sum()),
                passengers_in_mixed_cells=int(n[mixed].sum()),unavoidable_with_these_bits=errors,
                reference_errors=int((anchor[mask]!=truth[mask]).sum()),oracle_accuracy=1-errors/len(index))


def transfer(train,query,weight):
    tb,ta,tp,ty,tg=train
    qb,qa,qp=query
    wrong=ta!=ty
    output=np.zeros((len(qb),3),bool)
    for direction in (False,True):
        bank=np.flatnonzero(ta==direction)
        ids=np.flatnonzero(qa==direction)
        if not len(bank) or not len(ids):
            continue
        # Bound leverage from a single labeled booking when learning an error rate.
        _,codes,counts=np.unique(tg[bank],return_inverse=True,return_counts=True)
        booking_weight=1/counts[codes]
        odds=logit(np.where(ta[bank],1-tp[bank],tp[bank]).clip(.002,.998))
        for block in np.array_split(ids,int(np.ceil(len(ids)/64))):
            distance=cdist(qb[block],tb[bank],metric='cityblock',w=weight)
            limit=np.partition(distance,min(63,len(bank)-1),axis=1)[:,min(63,len(bank)-1)]
            kernel=2**(-(distance-distance.min(1)[:,None]))*(distance<=limit[:,None])*booking_weight
            exact=(distance<1e-10)*booking_weight
            native=logit(np.where(qa[block],1-qp[block],qp[block]).clip(.002,.998))
            for j,w in enumerate((exact,kernel)):
                shift=np.zeros(len(block))
                # One local intercept per query, shrunk toward its native probability.
                for _ in range(5):
                    p=expit(odds+shift[:,None])
                    gradient=(w*(wrong[bank]-p)).sum(1)-4*shift
                    curvature=(w*p*(1-p)).sum(1)+4
                    shift+=(gradient/curvature).clip(-1.,1.)
                output[block,j]=(native+shift>0)&(native<=0)&(w.sum(1)>0)
            output[block,2]=(kernel@wrong[bank]+2)/(kernel.sum(1)+4)>.5
    return output


def run():
    raw,y,g,views=regimes()
    past=views[0][1]
    history=None
    history_names=None
    results={}
    for tag,ids,fold,prob,view in views:
        anchor=view['anchor']
        truth=y[ids]
        known=view['known']
        names,old=archive(tag,ids,fold,g,y,ART/'nov24_remaining_cases/archive_scores.npz')
        bits=np.column_stack([view['votes'],old])
        weight=np.ones(bits.shape[1])
        families=np.array([n.split('/')[0] for n in names])
        for i,name in enumerate(families):
            weight[view['votes'].shape[1]+i]=1/(families==name).sum()
        solo=~raw.PassengerId.str.split('_').str[0].duplicated(keep=False).to_numpy()[ids]
        common=view['eligible']&(view['votes']==anchor[:,None]).all(1)
        unanimous=common&(old==anchor[:,None]).all(1)
        scopes={'all':known,'solo':known&solo,'parties':known&~solo,'common':common,'shared_with_archive':unanimous}
        bounds={n:{c:ceiling(b,truth,anchor,m) for c,m in scopes.items()} for n,b in
                [('original_families',view['votes']),('families_and_archive',bits)]}
        def pool(t):
            return bits[t],anchor[t],prob[t],truth[t],g[ids[t]]
        decisions=np.zeros((len(ids),3),bool)
        for k in np.unique(fold):
            s=np.flatnonzero(known&(fold==k))
            for a,b in GroupKFold(3,shuffle=True,random_state=3498).split(s,groups=g[ids[s]]):
                t,q=s[a],s[b]
                assert not np.isin(g[ids[t]],g[ids[q]]).any()
                decisions[q]=transfer(pool(t),(bits[q],anchor[q],prob[q]),weight)
        def assess(actions,masks):
            return {n:{c:ticker(m,truth!=anchor,v) for c,m in masks.items()} for n,v in
                    zip(['exact_offset','neighbors_offset','neighbors_rate'],actions.T,strict=True)}
        results[tag]=dict(oracle=bounds,local=assess(decisions,scopes))
        if tag=='historical':
            history=pool(np.flatnonzero(known))
            history_names=[*view['cohort'],*names]
        else:
            assert history_names==[*view['cohort'],*names]
            fresh=~np.isin(g[ids],g[past])&known
            correction=np.zeros_like(decisions)
            correction[fresh]=transfer(history,(bits[fresh],anchor[fresh],prob[fresh]),weight)
            results[tag]['on_new_orders']=assess(correction,{n:m&fresh for n,m in scopes.items()})
        print(tag,bounds['families_and_archive']['all'],results[tag]['local']['neighbors_offset']['all'],flush=True)
    DEST.with_name('vote_signature_transfer.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
