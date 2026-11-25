"""Find repeated nonbilling rescues and matched counterexamples among shared residual errors."""
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from cross_manifest_views_n26 import predictions
from disjoint_tree_fields_n25 import ART,DEST
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from outstanding_error_families_n26 import observed
from receipt_expert_transfer_n26 import read as receipt_scores

LABELS=['cabin_tree','cabin_transformer','party_tree','party_transformer']


def run():
    raw,y,g,f=data()
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0]);ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    full=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    models={}
    for tag,ids,fold,view in [('full',np.arange(len(y)),f,full),('historical',rows,hf,hist)]:
        before,after=predictions(tag,ids,fold)
        _,receipt=receipt_scores(tag,ids,fold)
        known=view['eligible']&(view['votes']==view['anchor'][:,None]).all(1)
        bills=((receipt>=0)!=view['anchor'][:,None]).any(1)
        orig=before[:,[0,2]].reshape(len(ids),4)
        new=after[:,[0,2]].reshape(len(ids),4)
        dissent=(new>=.5)!=view['anchor'][:,None]
        induced=dissent&((orig>=.5)==view['anchor'][:,None])
        take=rows if tag=='full' else np.arange(len(ids))
        models[tag]=[v[take] for v in (known,bills,dissent,induced)]
    known,bills,oppose,induced=models['full']
    hk,hb,hop,hi=models['historical']
    agree=known&hk&(full['anchor'][rows]==hist['anchor'])
    same=agree&~bills&~hb
    err=hist['anchor']!=y[rows]
    cold=np.zeros(len(rows),bool)
    for k in range(3):
        ix=hf==k;cold[ix]=~np.isin(g[rows[ix]],g[rows[~ix]])
    output={}
    for title,mask in [('matching_consensus',agree),('no_billing_rescue',same),('unseen_no_billing_rescue',same&cold)]:
        counts={}
        for n,flag in [('either_fit',oppose.any(1)|hop.any(1)),('both_fits_any',oppose.any(1)&hop.any(1)),
                       ('same_model_both_fits',(oppose&hop).any(1)),('induced_same_model_both_fits',(induced&hi).any(1)),
                       *[(name,oppose[:,j]&hop[:,j]) for j,name in enumerate(LABELS)]]:
            counts[n]=dict(fixed=int((mask&flag&err).sum()),harmed=int((mask&flag&~err).sum()))
        output[title]=dict(rows=int(mask.sum()),errors=int((mask&err).sum()),counts=counts)
    flagged=same&(oppose&hop).any(1)
    fixes=np.flatnonzero(flagged&err);harms=np.flatnonzero(flagged&~err)
    example_pairs=[]
    if len(fixes) and len(harms):
        features=np.column_stack([5*hist['anchor'],2*awake[rows],raw.Age.iloc[rows].fillna(30)/30,old_native])
        cost=np.linalg.norm(features[fixes,None]-features[harms],axis=2)
        a,b=linear_sum_assignment(cost)
        def describe(j):
            r=raw.iloc[rows[j]]
            return dict(passenger=str(r.PassengerId),planet=None if r.isna().HomePlanet else str(r.HomePlanet),
                        age=None if r.isna().Age else int(r.Age),anchor=bool(hist['anchor'][j]),actual=bool(y[rows[j]]),
                        repeated_dissent=[LABELS[i] for i in np.flatnonzero(oppose[j]&hop[j])])
        for ix in np.argsort(cost[a,b])[:3]:
            example_pairs.append(dict(rescued=describe(fixes[a[ix]]),misclassified_if_changed=describe(harms[b[ix]])))
    output['examples']=example_pairs
    DEST.with_name('repeated_nonbill_rescues.json').write_text(json.dumps(output,indent=2)+'\n')
    print(output['no_billing_rescue'],output['unseen_no_billing_rescue'],example_pairs,flush=True)


if __name__=='__main__':
    run()
