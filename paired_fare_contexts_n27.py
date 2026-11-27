"""Compare two overlapping, out-of-booking TabICL fits for every passenger."""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from booking_mix_confidence_transfer_n26 import partition
from price_implied_deck_n26 import build_features
from leave_booking_out_fare_n27 import roster
from repeated_archive_misses_n26 import ART, cohort, count
from full_roster_booking_validation_n23 import ROOT, data

OUT=ART/'nov27_paired_fare_contexts'
REPORT=ROOT/'reports/nov23_matched_learning_curve/paired_fare_contexts.json'


def train():
    import torch
    from hashlib import sha256
    from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer
    torch.set_num_threads(2)
    assert sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    frame,y,groups,f=roster()
    views=build_features(frame,groups)
    n=len(y)
    OUT.mkdir(exist_ok=True)
    for k in range(8):
        hold=(f==k)|(f==(k+1)%8)
        t=np.flatnonzero(~hold)
        q=np.r_[np.flatnonzero(hold),np.arange(n,len(frame))]
        assert not np.isin(groups[t],groups[q]).any()
        for name in ('raw','fine'):
            path=OUT/f'{name}_{k}.npz'
            if not path.exists():
                p=infer(views[name],y,t,q,6220+k)
                tmp=path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,passenger=frame.PassengerId.iloc[q].to_numpy(str),t=t,q=q,labels=y[t],p=p,weight=DIGEST)
                tmp.replace(path)
            print(k,name,flush=True)


def assess():
    frame,y,groups,f=roster()
    raw,truth,grp,fold=data()
    np.testing.assert_array_equal(y,truth)
    np.testing.assert_array_equal(f,fold)
    _,blind,known,fresh=cohort(raw,y,f)
    ids=frame.PassengerId.to_numpy(str)
    n=len(y)
    h=np.stack([np.r_[f,partition(groups[n:],'matched_pair_fare_contexts',8)]]*2)
    h[1]=(h[0]-1)%8
    probs={}
    for name in ('raw','fine'):
        p=np.full((len(frame),8),np.nan)
        for k in range(8):
            with np.load(OUT/f'{name}_{k}.npz') as z:
                tr=np.flatnonzero((f!=k)&(f!=(k+1)%8))
                q=np.r_[np.flatnonzero((f==k)|(f==(k+1)%8)),np.arange(n,len(frame))]
                for field,val in [('t',tr),('q',q),('labels',y[tr]),('passenger',ids[q])]:
                    np.testing.assert_array_equal(z[field],val)
                p[q,k]=z['p']
        probs[name]=p
    with np.load(ART/'nov27_dissent_likelihood_transfer/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        first=z['first']
        prior=z['random_median']
    anchor=first^prior
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        native=np.median(z['scores'][:,-3:],axis=1)
        oldmodel=z['scores'][:,-2]
    odds=lambda v:logit(v.clip(.002,.998))
    sign=np.where(anchor,-1.,1.)
    raw_votes=(probs['raw']>=.5)!=anchor[:,None]
    fine_votes=(probs['fine']>=.5)!=anchor[:,None]
    increments=(odds(probs['fine'])-odds(probs['raw']))*sign[:,None]
    signals=dict(fine=fine_votes,changed=fine_votes&~raw_votes,
                 strengthened=fine_votes&(increments>0),
                 incremental=fine_votes&(increments>abs(odds(native))[:,None]))
    report={}
    out={}
    rows=np.arange(len(frame))[:,None]
    for name,grid in signals.items():
        matched=grid[rows,h.T].all(1)
        for tag,more in [('all',np.ones(len(frame),bool)),('original_support',(oldmodel>=.5)!=anchor),
                         ('retained',~prior)]:
            a=matched & more
            err=anchor[:n]!=y
            delta=a[:n]*np.where(err,1,-1)
            extra=np.mean(grid[n:],axis=1)>=.75
            label=name+'_'+tag
            out[label]=a
            out[label+'_six_of_eight']=extra&more[n:]
            report[label]=dict(all=count(a[:n],known,err),fresh=count(a[:n],fresh&known,err),old=count(a[:n],blind,err),
                by_fold=[int(delta[known&(f==j)].sum()) for j in range(8)],test_pair=int(a[n:].sum()),
                test_six_of_eight=int((extra&more[n:]).sum()))
    np.savez_compressed(OUT/'aligned.npz',passenger=ids,y=y,fold=f,**out)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['train','assess'])
    globals()[parser.parse_args().stage]()
