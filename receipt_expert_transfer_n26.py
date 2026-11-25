"""Test complementary bill evidence jointly and transfer historical coefficients to unseen bookings."""
import json

import numpy as np
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART,DEST,stats
from family_identity_transfer_n26 import fit,panels_and_reference
from full_roster_booking_validation_n23 import data
from latent_bill_rounding_n26 import OUT
from outstanding_error_families_n26 import observed


def odds(p):
    return logit(np.clip(p,.002,.998))


def read(tag,ids,fold):
    before=np.empty((len(ids),5))
    after=np.empty_like(before)
    for k in np.unique(fold):
        loc=fold==k
        q=ids[loc]
        paths=[ART/f'nov26_conditional_bill_deviations/{tag}_{k}.npz',
               ART/f'nov26_latent_receipt_channel/{tag}_{k}.npz',
               OUT/f'outcome_{tag}_{k}.npz',OUT/f'charge_classifier_{tag}_{k}.npz',
               OUT/f'charge_tabicl_{tag}_{k}.npz']
        for j,path in enumerate(paths):
            with np.load(path) as z:
                np.testing.assert_array_equal(z['q'],q)
                if j==0:
                    a,b=z['raw'],z['deviation']
                elif j==1:
                    a,b=z['prior'],z['prior']+z['channel']
                elif j==2:
                    a,b=z['direct'],z['floor_features']
                elif j==3:
                    a,b=odds(z['raw']),odds(z['spectrum'])
                else:
                    a,b=odds(z['ordinary']),odds(z['evidence'])
                before[loc,j],after[loc,j]=a,b
    return before,after


def run():
    raw,y,g,f=data()
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    current=observed(np.arange(len(raw)),f,p,names,native,*ref,awake,'full')
    old_fit={}
    report={}
    for tag,ids,fold,nat,view in [('historical',rows,hf,old_native,hist),('full',np.arange(len(y)),f,native,current)]:
        before,after=read(tag,ids,fold)
        anchor=view['anchor']
        known=view['known']
        wrong=anchor!=y[ids]
        sign=np.where(anchor,-1,1)
        off=sign*odds(nat)
        pre=sign[:,None]*before
        post=sign[:,None]*after
        revisions=(post>0)&(pre<=0)
        oppose=(post>0).sum(1)
        flags={'two_revisions':revisions.sum(1)>=2,'three_revisions':revisions.sum(1)>=3,
               'one_revision_majority':revisions.any(1)&(oppose>=3),'all_oppose':oppose==5}
        fresh=~np.isin(g[ids],g[rows])
        if tag=='historical':
            for k in np.unique(fold):
                loc=fold==k
                fresh[loc]=~np.isin(g[ids[loc]],g[ids[~loc]])
        basic=np.column_stack([np.ones(len(ids)),sign,off])
        designs={'base':basic,'ordinary':np.column_stack([basic,pre-off[:,None]]),
                 'all_bill_evidence':np.column_stack([basic,pre-off[:,None],post-pre])}
        for key,x in designs.items():
            if tag=='historical':
                old_fit[key]=fit(x[known],off[known],wrong[known])
            else:
                flags['history_to_new_'+key]=off+x@old_fit[key]>0
            cross=np.zeros(len(ids),bool)
            for k in np.unique(fold):
                pool=np.flatnonzero((fold==k)&known)
                for tr,te in GroupKFold(4,shuffle=True,random_state=9435).split(pool,groups=g[ids[pool]]):
                    a,b=pool[tr],pool[te]
                    assert not np.isin(g[ids[a]],g[ids[b]]).any()
                    coef=fit(x[a],off[a],wrong[a])
                    cross[b]=off[b]+x[b]@coef>0
            flags[key]=cross
        family=view['eligible']&(view['votes']==anchor[:,None]).all(1)
        res={}
        for key,edit in flags.items():
            sets={}
            for cohort,mask in [('all',known),('new',known&fresh),('families',family),('new_families',family&fresh)]:
                if key.startswith('history_to_new') and not cohort.startswith('new'):
                    continue
                delta=edit*np.where(wrong,1,-1)
                sets[cohort]=stats(delta,mask,fold,g[ids])
            res[key]=sets
        edits_any=(post>0).any(1)&known
        report[tag]=dict(results=res,upper=dict(original_errors=int((known&wrong).sum()),reachable=int((edits_any&wrong).sum()),
            exposed_correct=int((edits_any&~wrong).sum()),family_errors=int((family&wrong).sum()),
            family_reachable=int((edits_any&family&wrong).sum()),family_exposed=int((edits_any&family&~wrong).sum())))
        print(tag,[(m,s.get('all',s['new'])['fixed'],s.get('all',s['new'])['harmed']) for m,s in res.items()],flush=True)
    DEST.with_name('receipt_expert_transfer.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
