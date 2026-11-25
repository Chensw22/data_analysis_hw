"""Compare absolute and newly corroborated decisions on genuinely incomplete receipts."""
import json

import numpy as np

from disjoint_tree_fields_n25 import ART,DEST,stats
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from latent_bill_rounding_n26 import OUT,BILLS
from outstanding_error_families_n26 import observed


def run():
    raw,y,g,f=data()
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    current=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    missing=raw[BILLS].isna().to_numpy()
    result={}; retained={}
    for tag,ids,fold,view in [('historical',rows,hf,hist),('full',np.arange(len(y)),f,current)]:
        single=missing[ids].sum(1)==1
        clean=~np.isin(g[ids],g[rows])
        flags={}
        for k in np.unique(fold):
            loc=np.flatnonzero(fold==k)
            q=ids[loc]
            if tag=='historical':
                clean[loc]=~np.isin(g[q],g[ids[fold!=k]])
            select=np.flatnonzero(single[loc])
            j=missing[q[select]].argmax(1)
            with np.load(OUT/f'synthetic_{tag}_{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                raw_tree,filled_tree,specialist=z['score'][select,j].T>=.5
            with np.load(OUT/f'charge_tabicl_{tag}_{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                raw_net=z['base'][select]>=.5
                filled_net=z['integrated'][select]>=.5
            a=view['anchor'][loc[select]]
            score=(filled_tree!=a).astype(int)+(filled_net!=a)+(specialist!=a)
            raw_score=(raw_tree!=a).astype(int)+(raw_net!=a)+(specialist!=a)
            affected=((raw_tree!=filled_tree)&(filled_tree!=a))|((raw_net!=filled_net)&(filled_net!=a))
            methods={'raw_majority':raw_score>=2,'imputed_majority':score>=2,'raw_unanimity':raw_score==3,
                     'imputed_unanimity':score==3,'new_majority':(score>=2)&(raw_score<2),
                     'new_unanimity':(score==3)&(raw_score<3),'majority_with_change':(score>=2)&affected}
            for name,selected in methods.items():
                flags.setdefault(name,np.zeros(len(ids),bool))[loc[select]]=selected
        wrong=view['anchor']!=y[ids]
        family=view['eligible']&(view['votes']==view['anchor'][:,None]).all(1)
        table={}
        for name,flag in flags.items():
            delta=flag*np.where(wrong,1,-1)
            table[name]={scope:stats(delta,mask,fold,g[ids]) for scope,mask in [('all',view['known']),('new',view['known']&clean),
                         ('missing',single&view['known']),('families',family)]}
        result[tag]=table
        retained[tag]=flags
        print(tag,[(n,s['all']['fixed'],s['all']['harmed']) for n,s in table.items()])
    paired=current['known'][rows]&hist['known']&(hist['anchor']==current['anchor'][rows])
    result['same_people']={}
    for name,bits in retained['historical'].items():
        changes=bits&retained['full'][name][rows]
        reward=changes*np.where(hist['anchor']!=y[rows],1,-1)
        result['same_people'][name]=stats(reward,paired,hf,g[rows])
    DEST.with_name('missing_bill_committee.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    run()
