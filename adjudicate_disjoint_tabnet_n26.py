"""Test countervotes replicated on disjoint training labels beyond the already queued TabNet release."""
import json
import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import fit
from disjoint_booking_tabnet_n26 import OUT as HALVES
from evaluate_nested_release_n26 import DECK,REVIEW,panel_predictions,protocols,sources,stages
from frozen_queue_shared_errors_n26 import paired
from tabnet_attention_reliability_n26 import signals
from tabnet_attention_transfer_n26 import decide,design,historical_outputs
from target_mix_expert_routing_n26 import ROOT,batches,ticker

REPORT=ROOT/'reports/nov23_matched_learning_curve/adjudicate_disjoint_tabnet.json'


def half_signals(x,ref,p):
    signed=logit(p.clip(.002,.998))*np.where(ref,-1,1)[:,None]
    avg=signed[:,:2].mean(1)
    gap=np.abs(signed[:,0]-signed[:,1])
    contrary=(p[:,:2]>=.5)!=ref[:,None]
    return np.column_stack([x,avg,gap,avg-signed[:,2],gap*avg,contrary.sum(1)]),contrary


def run():
    raw,y,g,f,known,versions=paired()
    panels=sources()[-1]
    model=panel_predictions(panels['full'],'full')
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
    net={'full':signals(raw,y,g,f),'historical':historical_outputs(raw,y,g,panels['historical'])}
    split={}
    for tag,v in panels.items():
        with np.load(HALVES/tag/'combined.npz') as z:
            ids=v['ids']
            np.testing.assert_array_equal(ids,z['ids'])
            np.testing.assert_array_equal(z['fold'],v['fold'])
            np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[ids])
            np.testing.assert_array_equal(z['p_full'],net[tag][0][ids,0])
            split[tag]=np.full((len(y),3),np.nan)
            split[tag][ids]=np.column_stack([z[n] for n in ('p0','p1','p_full')])
    results={}
    for detail,b in zip(protocols(panels,g,True),batches(True),strict=True):
        key,src,dst,it,iq=detail
        t,q=b.train_ids,b.ids
        np.testing.assert_array_equal(src['ids'][it],t)
        np.testing.assert_array_equal(dst['ids'][iq],q)
        assert known[q].all() and not np.isin(g[t],g[q]).any()
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            risk=np.column_stack([z['identities_error'],z['identities_outcome']])
        anchor,rev,fare,now=stages(dst,model,iq,enabled[q],risk)
        np.testing.assert_array_equal(anchor,versions[q,6])
        np.testing.assert_array_equal(now,b.query_cell%2)
        original=(versions[q,:7]==now[:,None]).all(1)&(rev==now)&(fare==now)
        old=b.train_cell%2!=0
        tag='full' if b.tag=='full' else 'historical'
        p,a=net[tag]
        full,attention=net['full']
        prior=decide(b.train_fields,b.query_fields,old,now,b.train_error,p[t],full[q],a[t],attention[q])
        common=original&~prior['actual_and_scores']
        xt=design(b.train_fields,p[t],old)
        xq=design(b.query_fields,full[q],now)
        xt,vt=half_signals(xt,old,split[tag][t])
        xq,vq=half_signals(xq,now,split['full'][q])
        w=np.where(vt.any(1),1.,.2)
        primary=fit(xt,b.train_error,xq,w)>.5
        with_masks=fit(np.column_stack([xt,a[t]]),b.train_error,np.column_stack([xq,attention[q]]),w)>.5
        edits={'one_or_more':vq.any(1)&primary,'two':vq.all(1)&primary,
               'two_and_attention':vq.all(1)&primary&with_masks,'any_and_attention':vq.any(1)&primary&with_masks}
        results.setdefault(b.tag,[]).append((q,original,common,now,prior['actual_and_scores'],edits))
    out={}
    previous=json.loads((REPORT.parent/'tabnet_attention_transfer.json').read_text())
    for tag,stack in results.items():
        q,orig,same,now,prior=(np.concatenate([v[i] for v in stack]) for i in range(5))
        assert len(q)==len(np.unique(q))
        wrong=y[q]!=now
        prior_count=ticker(orig,wrong,prior)
        for k in ('fixed','harmed'):
            assert prior_count[k]==previous[tag]['actual_and_scores'][k]
        out[tag]={}
        for key in stack[0][-1]:
            act=np.concatenate([v[-1][key] for v in stack])
            r=ticker(same,wrong,act)
            r['net_by_fold']=[int((np.where(wrong,1,-1)*act*same*(f[q]==k)).sum()) for k in range(8)]
            r['combined_accuracy']=float(((now^(orig&prior)^(same&act))==y[q]).mean())
            out[tag][key]=r
        print(tag,{k:(r['fixed'],r['harmed']) for k,r in out[tag].items()},flush=True)
    REPORT.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
