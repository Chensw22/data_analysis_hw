"""Compare fixed and target-normalized confidence ranks for the 68 release-ready predictors."""
import json

import numpy as np
from scipy.special import expit, ndtri
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import assignments
from family_identity_transfer_n26 import fit, panels_and_reference
from full_roster_booking_validation_n23 import ROOT
from solo_archive_recovery_n26 import ticker
from tabicl_consensus_review_n26 import sources


def percentiles(reference, query):
    # Midranks preserve ties at one half and other repeated values.
    arr=np.empty_like(query)
    for j in range(query.shape[1]):
        a=np.sort(reference[:,j])
        left=np.searchsorted(a,query[:,j],side='left')
        right=np.searchsorted(a,query[:,j],side='right')
        arr[:,j]=ndtri((left+right+1)/(2*(len(a)+1)))
    return arr


def scaled(train,query,train_domain,query_domain):
    # Estimate quantiles separately in each prediction direction and booking type.
    out=np.empty_like(query)
    for domain in np.unique(query_domain):
        ask=query_domain==domain
        use=train_domain==domain
        if use.sum()<12:
            use=train_domain//2==domain//2
        if use.sum()<12:
            use=np.ones(len(train),bool)
        out[ask]=percentiles(train[use],query[ask])
    return out


def inputs(v,ranks,ids):
    # Normalize within each passenger; a separate absolute-margin term preserves overall confidence.
    level=np.median(ranks,axis=1)
    x=(ranks-level[:,None])/2
    return np.column_stack([v['basic'][ids,:3],level,np.quantile(x,[.15,.85],axis=1).T,
                            x,x*np.sign(v['basic'][ids,1,None])])


def run():
    raw,y,g,views=sources()
    _,_,_,rows,names,curp,oldp,*_=panels_and_reference()
    old,cur=views['historical'],views['full']
    np.testing.assert_array_equal(rows,old['ids'])
    # Track release-time feature availability and name alignment directly against the frozen artifact.
    with np.load(ROOT/'artifacts/proposal_n26_dispute_band/frozen_votes.npz') as z:
        np.testing.assert_array_equal(z['names'],names)
    size=raw.booking_size.to_numpy()
    def oriented(v,probs):
        side=v['anchor'].astype(int)
        domain=side*2+(size[v['ids']]==1)
        return (1-2*side[:,None])*probs,domain
    historical,hd=oriented(old,oldp)
    current,cd=oriented(cur,curp)
    flip={};scores={};tested=np.zeros(len(y),bool)
    for f,k,t,q in assignments(old,cur,g):
        tr=scaled(historical[t],historical[t],hd[t],hd[t])
        x=inputs(old,tr,t)
        coef=fit(x,old['off'][t],old['wrong'][t])
        for mode in ('source_scale','roster_scale'):
            block=np.empty((len(q),len(names)))
            for j in np.unique(cur['fold'][q]):
                loc=cur['fold'][q]==j
                if mode=='source_scale':
                    block[loc]=scaled(historical[t],current[q[loc]],hd[t],cd[q[loc]])
                else:
                    # All unlabeled passengers used for normalization share the very same fitted base models.
                    available=cur['known']&(cur['fold']==j)
                    block[loc]=scaled(current[available],current[q[loc]],cd[available],cd[q[loc]])
            score=cur['off'][q]+inputs(cur,block,q)@coef
            scores.setdefault(mode,np.zeros(len(y)))[q]=score
            flip.setdefault(mode,np.zeros(len(y),bool))[q]=score>0
        tested[q]=True
    assert tested.sum()==6055
    flip['both_scales']=flip['source_scale']&flip['roster_scale']
    fresh=tested&~np.isin(g,g[rows])
    scopes={'all':tested,'unanimous':tested&cur['common'],'fresh':fresh,'fresh_unanimous':fresh&cur['common']}
    out={};wrong=cur['wrong']
    mix=json.loads((ROOT/'reports/nov23_matched_learning_curve/booking_mix_confidence_transfer.json').read_text())
    weights=np.array(mix['target_test_share'])
    for n,flags in flip.items():
        tab={s:ticker(m,wrong,flags) for s,m in scopes.items()}
        delta=flags*np.where(wrong,1,-1)
        tab['test_mix_gain_pp']=float(100*weights@np.array([delta[tested&(size.clip(max=4)==b)].mean() for b in range(1,5)]))
        if n in scores:
            tab['auc']={s:float(roc_auc_score(wrong[m],scores[n][m])) for s,m in scopes.items()}
        out[n]=tab
    (ROOT/'reports/nov23_matched_learning_curve/transported_confidence_ranks.json').write_text(json.dumps(out,indent=2)+'\n')
    print([(k,v['all'],v['unanimous'],v['test_mix_gain_pp']) for k,v in out.items()],flush=True)


if __name__=='__main__':
    with threadpool_limits(2):
        run()
