"""Compare cabin reconstruction, bill reconstruction and co-traveller evidence across two architectures."""
import json

import numpy as np
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART,DEST,stats
from family_identity_transfer_n26 import fit,panels_and_reference
from full_roster_booking_validation_n23 import data
from outstanding_error_families_n26 import observed
from receipt_expert_transfer_n26 import read as receipt_scores


def predictions(tag,ids,fold):
    before=np.empty((len(ids),3,2))
    after=np.empty_like(before)
    with np.load(ART/f'nov26_global_cabin_assignment/geometry_{tag}.npz') as z:
        np.testing.assert_array_equal(z['q'],ids)
        before[:,0,0],after[:,0,0]=z['plain'],z['geometry']
    for k in np.unique(fold):
        loc=fold==k
        q=ids[loc]
        files=[('nov26_global_cabin_assignment','tabicl_',0,1,'base','p'),
               ('nov26_latent_bill_rounding','charge_classifier_',1,0,'raw','spectrum'),
               ('nov26_latent_bill_rounding','charge_tabicl_',1,1,'ordinary','evidence'),
               ('nov26_confidence_ordered_booking','',2,0,'independent','conditional'),
               ('nov26_joint_party_tabicl','',2,1,'direct','joint')]
        for folder,prefix,view,arch,plain,extra in files:
            with np.load(ART/f'{folder}/{prefix}{tag}_{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                before[loc,view,arch],after[loc,view,arch]=z[plain],z[extra]
    return before,after


def run():
    raw,y,g,f=data()
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0]);ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    cur=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    old_fit={};decisions={};reports={}
    for tag,ids,fold,base,view in [('historical',rows,hf,old_native,hist),('full',np.arange(len(y)),f,native,cur)]:
        known=view['known'];anchor=view['anchor'];wrong=anchor!=y[ids]
        before,after=predictions(tag,ids,fold)
        pre=(before>=.5)!=anchor[:,None,None];post=(after>=.5)!=anchor[:,None,None]
        changed=post&~pre
        flags={}
        # Count information sources and architectures independently to avoid double-counting replicas.
        for name,mask in [('opinion',post),('revision',changed)]:
            both=mask.all(2)
            flags[name+'_two_sources_both_models']=both.sum(1)>=2
            flags[name+'_all_sources_both_models']=both.all(1)
            flags[name+'_two_sources_each_architecture']=(mask.sum(1)>=2).all(1)
        flags['majority_new_vs_plain']=((post.sum((1,2))>=4)&(pre.sum((1,2))<4))
        flags['quarter_mean']=((.75*base+.25*after.mean((1,2))>=.5)!=anchor)&(anchor==(base>=.5))
        sign=np.where(anchor,-1,1)
        off=sign*logit(base.clip(.002,.998))
        original=sign[:,None,None]*logit(before.clip(.002,.998))
        step=sign[:,None,None]*(logit(after.clip(.002,.998))-logit(before.clip(.002,.998)))
        simple=np.column_stack([np.ones(len(ids)),sign,off,original.mean(1)-off[:,None]])
        main=np.column_stack([simple,original.reshape(len(ids),6)-off[:,None]])
        designs=dict(ordinary=main,mean_delta=np.column_stack([main,step.mean(2)]),
                     both_architectures=np.column_stack([main,step.mean(2),step.min(2),step.max(2)]))
        fresh=~np.isin(g[ids],g[rows])
        for k in np.unique(fold):
            if tag=='historical':
                loc=fold==k;fresh[loc]=~np.isin(g[ids[loc]],g[ids[~loc]])
            pool=np.flatnonzero((fold==k)&known)
            for t,q in GroupKFold(4,shuffle=True,random_state=21473).split(pool,groups=g[ids[pool]]):
                tr,held=pool[t],pool[q]
                assert not np.isin(g[ids[tr]],g[ids[held]]).any()
                for n,x in designs.items():
                    coef=fit(x[tr],off[tr],wrong[tr])
                    flags.setdefault('local_'+n,np.zeros(len(ids),bool))[held]=off[held]+x[held]@coef>0
        for n,x in designs.items():
            if tag=='historical':
                old_fit[n]=fit(x[known],off[known],wrong[known])
            else:
                flags['transfer_'+n]=fresh&(off+x@old_fit[n]>0)
        family=view['eligible']&(view['votes']==anchor[:,None]).all(1)
        table={}
        for n,flag in flags.items():
            d=flag*np.where(wrong,1,-1)
            table[n]={s:stats(d,m,fold,g[ids]) for s,m in [('all',known),('new',known&fresh),('families',family),('new_families',family&fresh)]
                      if not n.startswith('transfer_') or s.startswith('new')}
        decisions[tag]=flags
        any_new=post.any((1,2));any_old=pre.any((1,2))
        _,receipt=receipt_scores(tag,ids,fold)
        bill_dispute=((receipt>=0)!=anchor[:,None]).any(1)
        new_geography_or_party=post[:,[0,2]].any((1,2))
        coverage={}
        for n,m in [('all',known),('families',family)]:
            coverage[n]=dict(errors=int((m&wrong).sum()),reachable=int((m&wrong&any_new).sum()),
                             exposed=int((m&~wrong&any_new).sum()),new_rescue=int((m&wrong&any_new&~any_old).sum()),
                             new_harm=int((m&~wrong&any_new&~any_old).sum()),
                             beyond_receipt_rescue=int((m&wrong&~bill_dispute&new_geography_or_party).sum()),
                             beyond_receipt_harm=int((m&~wrong&~bill_dispute&new_geography_or_party).sum()),
                             still_unreached=int((m&wrong&~bill_dispute&~new_geography_or_party).sum()))
        reports[tag]=dict(scores=table,coverage=coverage)
        print(tag,coverage,[(k,r.get('all',r['new'])['fixed'],r.get('all',r['new'])['harmed']) for k,r in table.items()],flush=True)
    same=cur['known'][rows]&hist['known']&(cur['anchor'][rows]==hist['anchor'])
    reports['repeated']={}
    for n in decisions['historical']:
        if n.startswith('local_'):
            continue
        flag=decisions['full'][n][rows]&decisions['historical'][n]
        d=flag*np.where(hist['anchor']!=y[rows],1,-1)
        reports['repeated'][n]=stats(d,same,hf,g[rows])
    DEST.with_name('cross_manifest_views.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
