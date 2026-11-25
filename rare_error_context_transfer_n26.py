"""Test whether oversampling observed errors teaches tabular in-context review to recognize rare failures."""
import argparse
import hashlib
import json

import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score

from full_roster_booking_validation_n23 import ROOT
from tabicl_consensus_review_n26 import matrices, sources

OUT=ROOT/'artifacts/nov26_rare_error_context'


def choose(error,cells,seed):
    rng=np.random.default_rng(seed)
    picked=[]; control=[]; rates=np.ones(12)
    for c in np.unique(cells):
        ids=np.flatnonzero(cells==c)
        case=ids[error[ids]]
        ordinary=ids[~error[ids]]
        sample=min(len(ordinary),max(18,round(1.5*len(case))))
        selected=np.r_[case,rng.choice(ordinary,sample,replace=False)]
        picked.extend(selected)
        control.extend(rng.choice(ids,len(selected),replace=False))
        if len(ordinary):
            rates[c]=sample/len(ordinary)
    return rng.permutation(picked),rng.permutation(control),rates


def train():
    import torch
    from booking_mix_confidence_transfer_n26 import assignments
    from joint_party_tabicl_n26 import model
    from tabicl_routed_context_n23 import DIGEST,WEIGHT
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,g,v=sources()
    old,cur=v['historical'],v['full']
    def strata(z):
        prior=expit(z['off'])
        return (6*z['common']+3*z['anchor']+np.digitize(prior,[.12,.3])).astype(int)
    os,cs=strata(old),strata(cur)
    OUT.mkdir(exist_ok=True)
    for f,k,t,q in assignments(old,cur,g):
        path=OUT/f'{f}_{k}.npz'
        if not path.exists():
            xt,xq=matrices(raw,old,cur,t,q)
            xt=xt[1].assign(all_agree=old['common'][t].astype(int))
            xq=xq[1].assign(all_agree=cur['common'][q].astype(int))
            wrong=old['wrong'][t]
            enriched,sham,rates=choose(wrong,os[t],8674+f*5+k)
            assert len(set(enriched))==len(set(sham))==len(enriched)==len(sham)
            assert wrong[enriched].sum()==wrong.sum()
            p={}
            for n,subset in [('natural',np.arange(len(t))),('sham',sham),('enriched',enriched)]:
                est=model(6003).fit(xt.iloc[subset],wrong[subset])
                np.testing.assert_array_equal(est.classes_,[False,True])
                p[n]=est.predict_proba(xq)[:,1]
            p['prior_restored']=expit(logit(p['enriched'].clip(.0001,.9999))+np.log(rates[cs[q]]))
            with path.with_suffix('.tmp').open('wb') as file:
                np.savez_compressed(file,t=old['ids'][t],q=cur['ids'][q],train_y=y[old['ids'][t]],
                                    sample=enriched,sham_sample=sham,checkpoint=DIGEST,**p)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for n,val in [('t',old['ids'][t]),('q',cur['ids'][q]),('train_y',y[old['ids'][t]]),('checkpoint',DIGEST)]:
                np.testing.assert_array_equal(z[n],val)
        print(f,k,flush=True)


def score():
    from booking_mix_confidence_transfer_n26 import assignments
    from solo_archive_recovery_n26 import ticker
    raw,y,g,v=sources()
    old,cur=v['historical'],v['full']
    preds={};queries=[]
    for f,k,t,q in assignments(old,cur,g):
        with np.load(OUT/f'{f}_{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            for n in ('natural','sham','enriched','prior_restored'):
                preds.setdefault(n,[]).append(z[n])
        queries.append(q)
    q=np.concatenate(queries)
    assert len(q)==len(np.unique(q))==6055
    p={n:np.concatenate(blocks) for n,blocks in preds.items()}
    edits={n:prob>.5 for n,prob in p.items()}
    edits['both']=edits['natural']&edits['prior_restored']
    edits['new_after_restoration']=edits['prior_restored']&~edits['natural']
    fresh=~np.isin(g[q],g[old['ids']])
    wrong=cur['wrong'][q]
    scopes={'all':np.ones(len(q),bool),'unanimous':cur['common'][q],'new_orders':fresh,
            'new_unanimous':fresh&cur['common'][q]}
    mix=json.loads((ROOT/'reports/nov23_matched_learning_curve/booking_mix_confidence_transfer.json').read_text())
    weights=np.array(mix['target_test_share'])
    band=raw.booking_size.to_numpy()[q].clip(max=4)
    results={}
    for name,edit in edits.items():
        delta=edit*np.where(wrong,1,-1)
        result={s:ticker(m,wrong,edit) for s,m in scopes.items()}
        result['weighted_gain_pp']=float(100*weights@np.array([delta[band==b].mean() for b in range(1,5)]))
        results[name]=result
    metrics={}
    for n,prob in p.items():
        metrics[n]={}
        for s,m in scopes.items():
            ids=np.flatnonzero(m)
            top=ids[np.argsort(-prob[ids],kind='stable')[:int(np.ceil(len(ids)*.1))]]
            metrics[n][s]=dict(auc=float(roc_auc_score(wrong[m],prob[m])),nll=float(log_loss(wrong[m],prob[m])),
                               top_tenth=len(top),top_tenth_wrong=int(wrong[top].sum()))
    path=ROOT/'reports/nov23_matched_learning_curve/rare_error_context_transfer.json'
    path.write_text(json.dumps(dict(methods=results,metrics=metrics),indent=2)+'\n')
    print([(n,r['all']['fixed'],r['all']['harmed'],round(r['weighted_gain_pp'],3),r['unanimous']['fixed'],r['unanimous']['harmed'])
           for n,r in results.items()],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('train','score'))
    (train if parser.parse_args().action=='train' else score)()
