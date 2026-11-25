"""Conservative whole-order replay of four composite November 23 exports."""
import numpy as np
from scipy.special import expit

from broad_geographic_transfer_n23 import BILLS
from broad_service_bag_n23 import design,fit as fit_bag
from broad_two_source_manifest_n23 import fit_model
from broad_within_booking_bill_matching_n23 import encode
from broad_zero_spend_analogues_n23 import simulate
from registration_bill_links_n23 import fields,level_map,score
from service_pair_change_budget_n23 import changed


def components(frame,y,groups,fold,p108,cache):
    person,bills=encode(frame)
    personal=np.column_stack([person,bills,np.log(frame.BookingSize)])
    reg,fees,cat=fields(frame,level_map(frame))
    registration=np.column_stack([reg,fees])
    categorical=np.r_[cat,np.zeros(fees.shape[1],bool)]
    design_args=design(frame)
    paid=frame[BILLS].gt(0).any(axis=1).to_numpy()
    scores=np.full((len(y),4),np.nan)
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.isin(groups[t],groups[q]).any()
        path=cache/f'composite{k}.npz'
        if not path.exists():
            bag=fit_bag(*design_args,y,t,q,'fixed')
            individual=fit_model(personal[t],y[t].astype(int)).predict_proba(personal[q])[:,1]
            direct=expit(score(registration[t],y[t],registration[q],categorical))
            candidates=(~paid[q])&(p108[q]<.5)
            donor=np.full(len(q),np.nan)
            donor[candidates]=simulate(frame,groups,y,t,q[candidates],paid)['matched']
            np.savez_compressed(path,t=t,q=q,truth=y[t],p108=p108[q],bag=bag,individual=individual,direct=direct,donor=donor)
        with np.load(path) as z:
            for key,expected in [('t',t),('q',q),('truth',y[t]),('p108',p108[q])]:
                np.testing.assert_array_equal(z[key],expected)
            scores[q]=np.column_stack([z[n] for n in ('bag','individual','direct','donor')])
        print('verified grouped composite models',k,flush=True)
    return scores.T


def donor_bounds(donor,base,ref,fold):
    # Within each query fold the deployed rule takes floor(eligible / 10). Rows with
    # unresolved parent decisions can change both a passenger's rank and the quota.
    low=np.zeros(len(base),bool)
    high=low.copy()
    for k in np.unique(fold):
        q=np.flatnonzero((fold==k)&np.isfinite(donor)&(ref!=1))
        ranking=.75*base[q]+.25*donor[q]
        ranked=q[np.lexsort((q,-ranking))]
        certain=ref[ranked]==0
        low[ranked]=np.arange(1,len(ranked)+1)<=certain.sum()//10
        high[ranked]=np.cumsum(certain)<=len(ranked)//10
    assert not (low&~high).any()
    return low,high


def compose(old,profile,tree,linear,donor,pair_quarter,pair,reg,size):
    corrob=pair|((profile.astype(int)+linear+tree)>=2)
    families=profile.astype(int)+linear+tree+(donor|pair_quarter|pair)+reg
    return {old[0]:donor|linear,old[2]:donor|((size>=3)&linear),
            old[3]:donor|(families>=2),old[6]:corrob}


def composite_ranges(frame,y,g,fold,p108,prior,grouped,old,cache):
    bag,individual,direct,donor=components(frame,y,g,fold,p108,cache)
    ref=prior==1
    pair=grouped[old[5]]!=ref
    tree=grouped[old[4]]!=ref
    linear=changed(bag,p108,ref,.25)!=ref
    personal=changed(individual,p108,ref,.25)!=ref
    reg=changed(direct,p108,ref,.25)!=ref
    pair_quarter=changed(grouped['pair_probability'],p108,ref,.25)!=ref
    extra_low,extra_high=donor_bounds(donor,p108,prior,fold)
    outputs={}
    size=frame.BookingSize.to_numpy()[:len(y)]
    # A three-way vote requires a nested partition teacher and the historical risk
    # calibrator. Permit either decision here, only publishing rows invariant to both.
    for endpoint,ex,possible_three in [('low',extra_low,False),('high',extra_high,True)]:
        donor_flip=pair|ex
        profile=personal | possible_three
        decisions=compose(old,profile,tree,linear,donor_flip,pair_quarter,pair,reg,size)
        for n,a in decisions.items():
            outputs.setdefault(n,{})[endpoint]=a
    for v in outputs.values():
        assert not (v['low']&~v['high']).any()
    return outputs
