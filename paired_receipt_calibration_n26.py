"""Train on the weaker of two paired receipt margins, then predict entirely new bookings."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART, DEST, stats
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from outstanding_error_families_n26 import observed
from receipt_expert_transfer_n26 import odds,read


def representation(scores,anchor,native):
    before,after=scores
    sign=np.where(anchor,-1,1)
    off=sign*odds(native)
    base=np.column_stack([np.ones(len(off)),sign,off])
    controls=np.column_stack([base,sign[:,None]*before-off[:,None]])
    return off,{'base':base,'controls':controls,'receipt':np.column_stack([controls,sign[:,None]*(after-before)])}


def paired_fit(x1,x2,o1,o2,wrong,robust):
    z=np.where(wrong,1.,-1.)
    def objective(beta):
        m1=z*(o1+x1@beta)
        m2=z*(o2+x2@beta)
        if robust:
            # Soft minimum, normalized so identical predictions keep their original margin.
            margin=-np.logaddexp(-m1,-m2)+np.log(2.)
            w1=expit(m2-m1)
            factor=-z*expit(-margin)
            loss=np.logaddexp(0.,-margin).sum()
            grad=x1.T@(factor*w1)+x2.T@(factor*(1-w1))
        else:
            loss=.5*(np.logaddexp(0.,-m1).sum()+np.logaddexp(0.,-m2).sum())
            grad=-.5*(x1.T@(z*expit(-m1))+x2.T@(z*expit(-m2)))
        return loss+8*(beta@beta),grad+16*beta
    opt=minimize(objective,np.zeros(x1.shape[1]),jac=True,method='L-BFGS-B',options={'ftol':1e-10,'maxiter':400})
    assert opt.success,opt.message
    return opt.x


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
    fo,fv=representation(read('full',np.arange(len(y)),f),current['anchor'],native)
    ho,hv=representation(read('historical',rows,hf),hist['anchor'],old_native)
    comparable=hist['known']&current['known'][rows]&(hist['anchor']==current['anchor'][rows])
    fresh=~np.isin(g,g[rows])&current['known']
    def families(view):
        return view['eligible']&(view['votes']==view['anchor'][:,None]).all(1)
    common=families(current); paired_common=common[rows]&families(hist)
    flags={}
    training_counts={}
    for k in range(8):
        held=f==k
        # All current-model scores used to train these coefficients come from
        # this same excluded fold. No new booking was in any historical fit.
        for scope,paired,eligible in [('all',comparable,fresh),('families',comparable&paired_common,fresh&common)]:
            t=np.flatnonzero(paired&held[rows])
            q=np.flatnonzero(eligible&held)
            assert not np.isin(g[q],g[rows[t]]).any()
            wrong=hist['anchor'][t]!=y[rows[t]]
            training_counts.setdefault(scope,[]).append([len(t),int(wrong.sum())])
            for name in fv:
                for mode in ('average','weaker'):
                    coef=paired_fit(hv[name][t],fv[name][rows[t]],ho[t],fo[rows[t]],wrong,mode=='weaker')
                    pred=fo[q]+fv[name][q]@coef>0
                    flags.setdefault((scope,name,mode),np.zeros(len(y),bool))[q]=pred
    results={}
    wrong=current['anchor']!=y
    for (scope,name,mode),edit in flags.items():
        delta=edit*np.where(wrong,1,-1)
        mask=fresh if scope=='all' else fresh&common
        results['_'.join((scope,name,mode))]=stats(delta,mask,f,g)
    print([(key,r['fixed'],r['harmed']) for key,r in results.items()])
    DEST.with_name('paired_receipt_calibration.json').write_text(json.dumps(dict(training_counts=training_counts,results=results),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
