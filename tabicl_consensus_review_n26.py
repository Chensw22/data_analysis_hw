"""Evaluate a second opinion using passenger records and within-agreement confidence gradients."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT, data

OUT=ROOT/'artifacts/nov26_tabicl_consensus_review'
RESULT=ROOT/'reports/nov23_matched_learning_curve/tabicl_consensus_review.json'


def prepare():
    from family_identity_transfer_n26 import panels_and_reference
    from solo_selection_likelihood_n26 import regimes
    from stable_confidence_axes_n26 import panel
    raw,y,g,views=regimes()
    _,_,_,_,names,p,h,*_=panels_and_reference()
    OUT.mkdir(exist_ok=True)
    for v,a in zip(views,[h,p],strict=True):
        tag=v[0]
        block=panel(*v,g,y,a,names)
        np.savez_compressed(OUT/f'{tag}.npz',**{n:block[n] for n in
            ['basic','relative','off','known','common','anchor','wrong','ids','fold','votes']},
            passenger=raw.PassengerId.iloc[block['ids']].to_numpy(str),names=np.asarray(block['names']))


def sources():
    raw,y,g,_=data()
    frame=compact_features(raw)
    frame['booking_size']=raw.PassengerId.groupby(g).transform('size').to_numpy()
    selected={}
    for name in ('historical','full'):
        with np.load(OUT/f'{name}.npz') as z:
            v={n:z[n] for n in z.files}
        np.testing.assert_array_equal(v['passenger'],raw.PassengerId.iloc[v['ids']])
        np.testing.assert_array_equal(v['wrong'],v['anchor']!=y[v['ids']])
        selected[name]=v
    np.testing.assert_array_equal(selected['historical']['names'],selected['full']['names'])
    return frame,y,g,selected


def splits(v,g):
    hist,cur=v['historical'],v['full']
    for tag,b in v.items():
        for k in np.unique(b['fold']):
            rows=np.flatnonzero(b['known']&(b['fold']==k))
            for sub,(t,q) in enumerate(GroupKFold(3,shuffle=True,random_state=1871).split(rows,groups=g[b['ids'][rows]])):
                yield f'{tag}_{k}_{sub}',b,b,rows[t],rows[q]
    # The source models and every labeled meta-training row exclude these orders.
    fresh=cur['known']&~np.isin(g[cur['ids']],g[hist['ids']])
    yield 'historical_to_new',hist,cur,np.flatnonzero(hist['known']),np.flatnonzero(fresh)
    from booking_mix_confidence_transfer_n26 import assignments
    for f,k,t,q in assignments(hist,cur,g):
        yield f'bookingmix_{f}_{k}',hist,cur,t,q


def matrices(raw,src,dst,t,q):
    base=src['basic'][t]
    residual=src['relative'][t]
    coeff=np.linalg.solve(base.T@base+4*np.eye(base.shape[1]),base.T@residual)
    clean=residual-base@coeff
    center=clean.mean(0)
    scale=clean.std(0).clip(.15)
    axes=np.linalg.svd(((clean-center)/scale).clip(-7,7),full_matrices=False)[2][:12]
    fields={}
    for name,block,idx in [('t',src,t),('q',dst,q)]:
        x=raw.iloc[block['ids'][idx]].reset_index(drop=True)
        basic=block['basic'][idx]
        original=x.assign(original_decision=block['anchor'][idx].astype(int),
                          original_error_log_odds=block['off'][idx])
        quantiles=pd.DataFrame(basic[:,3:8],columns=[f'confidence_distribution_{i}' for i in range(5)])
        ordinary=pd.concat([original,quantiles],axis=1)
        resid=block['relative'][idx]-basic@coeff
        pcs=((resid-center)/scale).clip(-7,7)@axes.T
        named=pd.DataFrame(pcs,columns=[f'confidence_axis_{i}' for i in range(pcs.shape[1])])
        fields[name]=[ordinary,pd.concat([ordinary,named],axis=1)]
    return fields['t'],fields['q']


def train():
    import torch
    from joint_party_tabicl_n26 import model
    from tabicl_routed_context_n23 import DIGEST,WEIGHT
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,g,v=sources()
    for key,src,dst,t,q in splits(v,g):
        assert not np.isin(g[src['ids'][t]],g[dst['ids'][q]]).any()
        path=OUT/f'pred_{key}.npz'
        if not path.exists():
            support,query=matrices(raw,src,dst,t,q)
            scores={}
            for j,label in enumerate(['distribution','identities']):
                for n,target in [('outcome',y[src['ids'][t]]),('error',src['wrong'][t])]:
                    fit=model(7601).fit(support[j],target)
                    np.testing.assert_array_equal(fit.classes_,[False,True])
                    p=fit.predict_proba(query[j])[:,1]
                    if n=='outcome':
                        p=np.where(dst['anchor'][q],1-p,p)
                    scores[f'{label}_{n}']=p
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,t=src['ids'][t],q=dst['ids'][q],train_y=y[src['ids'][t]],checkpoint=DIGEST,**scores)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for n,val in [('t',src['ids'][t]),('q',dst['ids'][q]),('train_y',y[src['ids'][t]]),('checkpoint',DIGEST)]:
                np.testing.assert_array_equal(z[n],val)
        print(key,flush=True)


def score():
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    from solo_archive_recovery_n26 import ticker
    from stable_confidence_axes_n26 import calibrated,standardized
    raw,y,g,v=sources()
    result={}
    for key,src,dst,t,q in splits(v,g):
        tag=key.split('_')[0] if key!='historical_to_new' else key
        with np.load(OUT/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],dst['ids'][q])
            p={n:z[n] for n in ('distribution_error','distribution_outcome','identities_error','identities_outcome')}
        # Fit the named-confidence risk selector and its scales on the source rows alone.
        joined={n:np.concatenate([src[n][t],dst[n][q]]) for n in ('off','basic','relative','wrong')}
        s=np.arange(len(t)); zq=len(t)+np.arange(len(q))
        x=standardized(joined,s)
        risk=expit(calibrated(joined,joined,s,zq,x,x))
        flip={n:prob>.5 for n,prob in p.items()}
        flip['both_targets']=flip['identities_error']&flip['identities_outcome']
        flip['both_targets_new_vs_distribution']=flip['both_targets']&~(flip['distribution_outcome']&flip['distribution_error'])
        # Odds are for overturning the original combined decision, in both target formulations.
        flip['linear_and_neural']=(risk>.5)&flip['both_targets']
        result.setdefault(tag,[]).append((q,flip,p,risk))
    report={}
    for tag,batches in result.items():
        dst=v['historical' if tag=='historical' else 'full']
        q=np.concatenate([r[0] for r in batches])
        assert len(np.unique(q))==len(q)
        risk=np.concatenate([r[3] for r in batches])
        truth=dst['wrong'][q]
        common=dst['common'][q]
        highest=np.zeros(len(q),bool)
        members=np.flatnonzero(common)
        order=np.argsort(-risk[members],kind='stable')
        highest[members[order[:int(np.ceil(len(members)*.1))]]]=True
        scopes={'all':np.ones(len(q),bool),'unanimous':common,'highest_risk_tenth_of_unanimous':highest}
        if tag=='bookingmix':
            size=raw.booking_size.to_numpy()[q].clip(max=4)
            test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
            test_group=test.PassengerId.str.split('_').str[0]
            weights=np.bincount(test_group.map(test_group.value_counts()).clip(upper=4),minlength=5)[1:]/len(test)
        counts={}
        for n in batches[0][1]:
            change=np.concatenate([r[1][n] for r in batches])
            counts[n]={s:ticker(m,truth,change) for s,m in scopes.items()}
            if tag=='bookingmix':
                delta=change*np.where(truth,1,-1)
                counts[n]['test_mix_gain_pp']=float(100*weights@np.array([delta[size==b].mean() for b in range(1,5)]))
        ranks={n:float(roc_auc_score(truth[common],np.concatenate([r[2][n] for r in batches])[common])) for n in batches[0][2]}
        report[tag]=dict(methods=counts,auc_in_unanimous=ranks)
        print(tag,[(n,v['all']['fixed'],v['all']['harmed'],v['unanimous']['fixed'],v['unanimous']['harmed']) for n,v in counts.items()],flush=True)
    RESULT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','train','score'])
    {'prepare':prepare,'train':train,'score':score}[p.parse_args().action]()
