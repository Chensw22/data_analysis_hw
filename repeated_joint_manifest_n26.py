"""Test replicated conditional-field evidence across two genuinely different training rosters."""
import json

import numpy as np
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART, DEST
from family_identity_transfer_n26 import fit, panels_and_reference
from full_roster_booking_validation_n23 import data
from joint_manifest_forest_n26 import OUT
from outstanding_error_families_n26 import observed


def read(series, rows, folds):
    values = np.empty((len(rows),4,2))
    for k in np.unique(folds):
        mask = folds==k
        with np.load(OUT/f'{series}_{k}.npz') as z:
            np.testing.assert_array_equal(rows[mask],z['q'])
            values[mask] = np.stack([z[name] for name in ['service_pair','route','sleep_side','origin_berth']],axis=1)
    return values


def run():
    raw,y,g,f = data()
    _,_,_,rows,names,p,h,native,old_native,ref = panels_and_reference()
    awake = raw.CryoSleep.eq(False).to_numpy()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0] = z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par = z['fold'],z['nine'],z['v131']
    a = observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    b = observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    agree = a['anchor'][rows]==b['anchor']
    eligible = a['known'][rows]&b['known']&agree
    error = a['anchor'][rows]!=y[rows]
    fresh = np.zeros(len(rows),bool)
    for k in range(3):
        ix = hf==k
        fresh[ix] = ~np.isin(g[rows[ix]],g[rows[~ix]])
    common = (a['votes'][rows]==a['anchor'][rows,None]).all(1)&(b['votes']==b['anchor'][:,None]).all(1)
    full = read('full',np.arange(len(y)),f)[rows]
    hist = read('historical',rows,hf)
    direction = np.where(a['anchor'][rows],-1,1)
    def opposition(prob):
        return direction[:,None,None]*logit(prob.clip(.002,.998))
    v,w = opposition(full),opposition(hist)
    # A conditional update is selective if the held-out field changes the sign
    # of the masked prediction in both training regimes.
    both = np.minimum(v,w)
    incremental = np.minimum(v[:,:,1]-v[:,:,0],w[:,:,1]-w[:,:,0])
    change = (both[:,:,1]>0)&(np.maximum(v[:,:,0],w[:,:,0])<=0)
    flags = dict(any_repeated=(both[:,:,1]>0).any(1),majority_repeated=(both[:,:,1]>0).sum(1)>=3,
                 any_conditional=change.any(1),two_conditional=change.sum(1)>=2)
    for j,field in enumerate(['service_pair','route','sleep_side','origin_berth']):
        flags[field+'_repeated']=both[:,j,1]>0
        flags[field+'_conditional']=change[:,j]
    native_odds = direction*logit(native[rows].clip(.002,.998))
    historical_odds = direction*logit(old_native.clip(.002,.998))
    controls = np.column_stack([np.ones(len(rows)),direction,native_odds,historical_odds])
    xsets = {'base':controls,'masked':np.column_stack([controls,both[:,:,0]]),
             'joint':np.column_stack([controls,both[:,:,0],both[:,:,1], incremental])}
    for fi in range(8):
        for hi in range(3):
            pool = np.flatnonzero(eligible&(f[rows]==fi)&(hf==hi))
            for t,u in GroupKFold(3,shuffle=True,random_state=3861).split(pool,groups=g[rows[pool]]):
                tr,q = pool[t],pool[u]
                assert not np.isin(g[rows[tr]],g[rows[q]]).any()
                for name,x in xsets.items():
                    beta = fit(x[tr],native_odds[tr],error[tr])
                    flags.setdefault('fit_'+name,np.zeros(len(rows),bool))[q] = native_odds[q]+x[q]@beta>0
    report={}
    for name,mask in [('all',eligible),('unseen',eligible&fresh),('both_agreeing_families',eligible&common),
                      ('unseen_agreeing_families',eligible&fresh&common)]:
        counts={}
        for method,edits in flags.items():
            marked=mask&edits
            counts[method] = dict(fixed=int((marked&error).sum()),harmed=int((marked&~error).sum()))
        report[name]=dict(rows=int(mask.sum()),errors=int((mask&error).sum()),results=counts)
        print(name,report[name])
    DEST.with_name('repeated_joint_manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
