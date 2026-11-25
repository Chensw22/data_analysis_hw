"""Evaluate whether inverse-field evidence ranks common errors above comparable correct passengers."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART,DEST
from evaluate_inverse_tabicl_n26 import OUT,VIEWS,odds,ratio
from family_identity_transfer_n26 import fit,panels_and_reference
from full_roster_booking_validation_n23 import data


def run():
    raw,y,g,f=data()
    _,_,_,rows,_,panel,_,native,_,_=panels_and_reference()
    votes=(panel>=.5).sum(1)
    common=(votes==0)|(votes==68)
    side=votes==68
    target=side!=y
    fresh=~np.isin(g,g[rows])
    solo=~raw.PassengerId.str[:4].duplicated(keep=False).to_numpy()
    hard=common&~side&raw.CryoSleep.eq(False).to_numpy()&solo
    assert (hard&target).sum()==143
    opposition=np.where(side,-1.,1.)
    prior=opposition*odds(native)
    # Use observable odds ratios from all three tasks plus the restricted positive-basket fit.
    signals={name:np.zeros((len(y),4)) for name in ('observed','shuffled')}
    for k in range(8):
        q=np.flatnonzero(f==k)
        with np.load(OUT/f'{k}_paid_identity.npz') as z:
            paid=z['q']
            assert np.isin(paid,q).all()
            for key,array in signals.items():
                array[paid,3]=ratio(z[key])*opposition[paid]
        for j,view in enumerate(VIEWS):
            with np.load(OUT/f'{k}_{view}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                for key,array in signals.items():
                    value=ratio(z[key])
                    if j==0:
                        value[~np.isin(q,paid)]=0
                    array[q,j]=value*opposition[q]
    reports={}
    for protocol in ('all','new'):
        predictions={}
        for k in range(8):
            held=np.flatnonzero(common&(f==k))
            if protocol=='new':
                splits=[(held[~fresh[held]],held[fresh[held]])]
            else:
                splits=[(held[a],held[b]) for a,b in GroupKFold(4,shuffle=True,random_state=3249).split(held,groups=g[held])]
            for t,q in splits:
                assert not np.isin(g[t],g[q]).any()
                basic=np.column_stack([np.ones(len(y)),side,prior])
                # Constrain the evaluation quota independently for each original class.
                # Estimate its upper-risk 10% from other bookings only.
                for key,x in [('prior',basic),*[(key,np.column_stack([basic,array])) for key,array in signals.items()]]:
                    model=fit(x[t],prior[t],target[t])
                    pred=prior[q]+x[q]@model
                    tr=prior[t]+x[t]@model
                    accept=np.zeros(len(q),bool)
                    for direction in (False,True):
                        limit=np.quantile(tr[side[t]==direction],.90)
                        accept[side[q]==direction]=pred[side[q]==direction]>=limit
                    predictions.setdefault(key,[]).append((q,pred,accept))
        for key,batches in predictions.items():
            q,score,selected=(np.concatenate([b[j] for b in batches]) for j in range(3))
            expected=np.flatnonzero(common&(fresh if protocol=='new' else True))
            np.testing.assert_array_equal(np.sort(q),expected)
            result={}
            for name,mask in [('unanimous_false',~side[q]),('unanimous_true',side[q]),('awake_solo_false',hard[q])]:
                labels=target[q[mask]]
                result[name]=dict(rows=int(mask.sum()),errors=int(labels.sum()),
                    auc=float(roc_auc_score(labels,score[mask])),average_precision=float(average_precision_score(labels,score[mask])),
                    flagged_wrong=int((selected[mask]&labels).sum()),flagged_correct=int((selected[mask]&~labels).sum()))
            reports.setdefault(protocol,{})[key]=result
            print(protocol,key,result['awake_solo_false'])
    DEST.with_name('inverse_signal_ranking.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
