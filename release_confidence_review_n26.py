"""Review the release-ready confidence panel, with all three historical source folds on genuinely new orders."""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.special import logit

from booking_mix_confidence_transfer_n26 import assignments, partition
from full_roster_booking_validation_n23 import ROOT
from tabicl_consensus_review_n26 import matrices, sources

OUT=ROOT/'artifacts/nov26_release_confidence_review'


def summarize(p, native, anchor):
    sign=np.where(anchor,-1,1)
    odds=sign[:,None]*logit(p.clip(.002,.998))
    off=sign*logit(native.clip(.002,.998))
    qt=np.quantile(odds,[.1,.25,.5,.75,.9],axis=1).T
    center=qt[:,2]
    basic=np.column_stack([np.ones(len(p)),sign,off,center-off,qt[:,0]-center,qt[:,1]-center,
                           qt[:,3]-center,qt[:,4]-center])
    return dict(basic=np.column_stack([basic,basic[:,3:]*sign[:,None]]),relative=odds-center[:,None],
                off=off,anchor=anchor,agreement=((p>=.5)==anchor[:,None]).all(1))


def prepare():
    from family_identity_transfer_n26 import panels_and_reference
    from kaggle_workflow import sha256
    from export_budget_quarter_n23 import source
    from full_roster_booking_validation_n23 import data
    _,y,_,views=sources()
    _,_,_,rows,columns,p,h,native,old_native,_=panels_and_reference()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    frozen=ROOT/'artifacts/proposal_n26_dispute_band/frozen_votes.npz'
    record=json.loads((ROOT/'reports/proposal_n26_dispute_band/manifest.json').read_text())
    assert sha256(frozen)==record['model_sha256']
    with np.load(frozen) as z:
        np.testing.assert_array_equal(z['names'],columns)
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        bank=z['p']
    test_native=source('proposal_online_first_counts15',test)[0]
    artifact=ROOT/'artifacts/proposal_n26_joint_dispute_awake/submission.csv'
    receipt=json.loads((ROOT/'reports/proposal_n26_joint_dispute_awake/manifest.json').read_text())
    assert sha256(artifact)==receipt['sha256']
    base=pd.read_csv(artifact)
    np.testing.assert_array_equal(base.PassengerId,test.PassengerId)
    train=data()[0]
    assert rows.tolist()==views['historical']['ids'].tolist()
    OUT.mkdir(exist_ok=True)
    for tag,prob,inc,anchor,people in [
            ('historical',h,old_native,views['historical']['anchor'],train.PassengerId.iloc[rows]),
            ('full',p,native,views['full']['anchor'],train.PassengerId),
            ('test',bank,test_native,base.Transported.to_numpy(bool),test.PassengerId)]:
        np.savez_compressed(OUT/f'{tag}.npz',**summarize(prob,inc,anchor),names=columns,passenger=people.to_numpy(str))


def train():
    import hashlib
    import torch
    from joint_party_tabicl_n26 import model
    from tabicl_routed_context_n23 import DIGEST,WEIGHT
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,g,v=sources()
    old,cur=v['historical'],v['full']
    for tag,b in v.items():
        with np.load(OUT/f'{tag}.npz') as z:
            np.testing.assert_array_equal(z['passenger'],b['passenger'])
            np.testing.assert_array_equal(z['anchor'],b['anchor'])
            b.update({n:z[n] for n in ['basic','off','relative','agreement']})
    fresh=cur['known']&~np.isin(g,g[old['ids']])
    shard=partition(g,'query:',5)
    for f,k,t,eligible in assignments(old,cur,g):
        q=np.union1d(eligible,np.flatnonzero(fresh&(shard==k)))
        assert not np.isin(g[q],g[old['ids'][t]]).any()
        assert not np.isin(g[q],g[old['ids'][old['fold']!=f]]).any()
        path=OUT/f'fit_{f}_{k}.npz'
        if not path.exists():
            xt,xq=matrices(raw,old,cur,t,q)
            xt=xt[1].assign(all_agree=old['agreement'][t].astype(int))
            xq=xq[1].assign(all_agree=cur['agreement'][q].astype(int))
            result={}
            for n,labels in [('error',old['wrong'][t]),('outcome',y[old['ids'][t]])]:
                est=model(6003).fit(xt,labels)
                np.testing.assert_array_equal(est.classes_,[False,True])
                prob=est.predict_proba(xq)[:,1]
                result[n]=prob if n=='error' else np.where(cur['anchor'][q],1-prob,prob)
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,t=old['ids'][t],q=q,eligible=eligible,train_y=y[old['ids'][t]],weights=DIGEST,**result)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for n,val in [('t',old['ids'][t]),('q',q),('eligible',eligible),('train_y',y[old['ids'][t]]),('weights',DIGEST)]:
                np.testing.assert_array_equal(z[n],val)
        print(f,k,flush=True)


def score():
    from solo_archive_recovery_n26 import ticker
    raw,y,g,v=sources()
    old,cur=v['historical'],v['full']
    total=np.zeros((len(y),2));count=np.zeros(len(y),int)
    single=np.zeros_like(total)
    for f,k,t,eligible in assignments(old,cur,g):
        with np.load(OUT/f'fit_{f}_{k}.npz') as z:
            q=z['q']
            estimates=np.column_stack([z['error'],z['outcome']])
            total[q]+=estimates
            count[q]+=1
            loc=np.searchsorted(q,eligible)
            np.testing.assert_array_equal(q[loc],eligible)
            single[eligible]=estimates[loc]
    fresh=cur['known']&~np.isin(g,g[old['ids']])
    valid=count>0
    assert valid.sum()==6055 and np.all(count[fresh]==3) and np.all(count[valid&~fresh]==1)
    wrong=cur['wrong']
    size=raw.booking_size.to_numpy().clip(max=4)
    mix=json.loads((ROOT/'reports/nov23_matched_learning_curve/booking_mix_confidence_transfer.json').read_text())
    weights=np.array(mix['target_test_share'])
    result={}
    for batch,estimates in [('assigned',single),('all_eligible',total/count.clip(1)[:,None])]:
        outcomes={}
        for n,flags in [('error',estimates[:,0]>.5),('outcome',estimates[:,1]>.5),('both',(estimates>.5).all(1))]:
            delta=flags*np.where(wrong,1,-1)
            item={'all':ticker(valid,wrong,flags),'fresh':ticker(fresh,wrong,flags),'unanimous':ticker(valid&cur['common'],wrong,flags),
                  'size_net':[int(delta[valid&(size==b)].sum()) for b in range(1,5)],
                  'weighted_gain_pp':float(100*weights@np.array([delta[valid&(size==b)].mean() for b in range(1,5)]))}
            outcomes[n]=item
        result[batch]=outcomes
        print(batch,[(n,v['all']['fixed'],v['all']['harmed'],v['fresh']['fixed'],v['fresh']['harmed'],v['size_net']) for n,v in outcomes.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/release_confidence_review.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','train','score'])
    {'prepare':prepare,'train':train,'score':score}[parser.parse_args().action]()
