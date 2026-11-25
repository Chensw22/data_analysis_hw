"""Retest the full union on a second TabICL seed with each fold's training bookings unchanged."""
import argparse
import json
import numpy as np
import pandas as pd

from full_roster_booking_validation_n23 import ROOT,data

OUT=ROOT/'artifacts/nov26_historical_awake'


def generate():
    import torch
    import hashlib
    from exaone_validation_data import compact_features
    from tabicl_routed_context_n23 import WEIGHT,DIGEST,infer
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    raw,y,g,f=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    x=compact_features(pd.concat([raw,test],ignore_index=True))
    awake=raw.CryoSleep.eq(False).to_numpy()
    for k in range(8):
        q,t=np.flatnonzero((f==k)&awake),np.flatnonzero((f!=k)&awake)
        assert not np.isin(g[t],g[q]).any()
        path=OUT/f'full_{k}_2991.npz'
        if not path.exists():
            p=infer(x,y,t,q,2991+k)
            assert np.isfinite(p).all() and len(p)==len(q)
            tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as buf:
                np.savez_compressed(buf,p=p,q=q,t=t,labels=y[t],passenger=raw.PassengerId.iloc[q].to_numpy(str),
                                    checkpoint=DIGEST,seed=2991+k)
            tmp.replace(path)
        print('full same orders, second seed',k,flush=True)


def evaluate():
    from awake_context_ranks_n25 import OUT as ORIGINAL
    from disjoint_tree_fields_n25 import ART,DEST,stats
    from family_identity_transfer_n26 import panels_and_reference
    from joint_candidate_coverage_n26 import branch_decisions
    raw,y,g,f=data()
    awake=raw.CryoSleep.eq(False).to_numpy()
    _,_,_,rows,names,p,h,native,old_native,reference=panels_and_reference()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        old=z['nine'];hf=z['fold'];old_prior=z['v131']
    new=~np.isin(g,g[rows])
    report={}
    for tag,row,fold,seq,anchor,prior,lo,hi in [('full',np.arange(len(y)),f,p,native,*reference),
                                               ('historical',rows,hf,h,old_native,old_prior,old,old)]:
        pred=branch_decisions(row,fold,seq,names,anchor,prior,lo,hi,awake,tag,891)
        prob=np.column_stack([anchor,anchor])
        for k in np.unique(fold):
            m=(fold==k)&awake[row]
            paths=([ORIGINAL/f'{k}_personal.npz',OUT/f'full_{k}_2991.npz'] if tag=='full'
                   else [OUT/f'{k}_{seed+k}.npz' for seed in (891,2991)])
            for j,path in enumerate(paths):
                with np.load(path) as z:
                    np.testing.assert_array_equal(z['q'],row[m])
                    np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[row[m]])
                    prob[m,j]=z['p']
        side=anchor>=.5
        modes={'first':prob[:,0]>=.5,'second':prob[:,1]>=.5,
               'mean':prob.mean(1)>=.5,'both':np.where((prob[:,0]>=.5)==(prob[:,1]>=.5),prob[:,0]>=.5,side)}
        known=prior>=0
        cold=np.ones(len(row),bool)
        for k in np.unique(fold):
            cold[fold==k]=~np.isin(g[row[fold==k]],g[row[fold!=k]])
        scopes={'all':np.ones(len(row),bool)}
        scopes['new_orders' if tag=='full' else 'complete_order_holdout']=new[row] if tag=='full' else cold
        sums=(seq>=.5).sum(1)
        scopes['unanimous']=(sums==0)|(sums==68)
        for name,net in modes.items():
            outcomes=[]
            for b in (0,1):
                v=np.where(known,prior,b)
                nine=pred['nine'][b]
                free=(v==nine)&(v==side)
                change=(pred['band'][b]!=nine)|(pred['fields'][b]!=nine)|(free&(net!=nine))
                outcomes.append(nine^change)
            out=np.stack(outcomes)
            if name=='first':
                np.testing.assert_array_equal(out,pred['all_three'])
            checks={}
            for key,comparison in [('nine',pred['nine']),('first',pred['all_three'])]:
                delta=(out==y[row]).astype(int)-(comparison==y[row])
                np.testing.assert_array_equal(delta[0,known],delta[1,known])
                checks[key]={}
                for scope,mask in scopes.items():
                    v=stats(delta[0],known&mask,fold,g[row])
                    v['bounds']=[int(fun(delta,axis=0)[mask].sum()) for fun in (np.min,np.max)]
                    checks[key][scope]=v
            report.setdefault(tag,{})[name]=checks
            print(tag,name,[(s,v['fixed'],v['harmed']) for s,v in checks['nine'].items()],flush=True)
    DEST.with_name('replicate_awake_ordering.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['generate','evaluate'])
    generate() if parser.parse_args().action=='generate' else evaluate()
