"""Predict two passengers' four joint outcomes, then reconcile both forecasts for each passenger."""
import hashlib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_joint_observed_state_n23 import predict
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from tabicl_routed_context_n23 import DIGEST, WEIGHT


def link(index, groups, frame, shuffled):
    """Pair companions first, then remaining passengers by recorded cabin location."""
    size=len(index)
    near=np.full(size,-1)
    if shuffled:
        waiting=np.random.default_rng(614).permutation(size)
    else:
        local=pd.DataFrame(dict(g=groups[index],j=np.arange(size),deck=frame.Deck.iloc[index].to_numpy(),
                                side=frame.Side.iloc[index].to_numpy(),number=frame.CabinNumber.iloc[index].to_numpy()))
        # Cabin registration order breaks ties between companions without reading their outcome.
        local=local.sort_values(['g','deck','side','number','j'],kind='stable')
        order=local.groupby('g',sort=False).cumcount().to_numpy()
        width=local.groupby('g',sort=False).g.transform('size').to_numpy()
        stay=order < (width//2)*2
        paired=local.j.to_numpy()[stay].reshape(-1,2)
        near[paired[:,0]],near[paired[:,1]]=paired[:,1],paired[:,0]
        waiting=local.loc[~stay].sort_values(['deck','side','number','g','j'],kind='stable').j.to_numpy()
    if len(waiting)==1:
        extra=np.flatnonzero(near>=0)[0]
        waiting=np.r_[waiting,extra,near[extra]]
    if len(waiting)%2:
        tri=waiting[-3:]
        near[tri]=np.roll(tri,-1)
        waiting=waiting[:-3]
    pairs=waiting.reshape(-1,2)
    near[pairs[:,0]],near[pairs[:,1]]=pairs[:,1],pairs[:,0]
    np.testing.assert_array_equal(np.sort(near),np.arange(size))
    assert not (near==np.arange(size)).any()
    return near


def pairing(frame,groups,y,t,q,shuffled,seed):
    matrices,partners=[],[]
    for ix in (t,q):
        partner=link(ix,groups,frame,shuffled)
        partners.append(partner)
        first=frame.iloc[ix].reset_index(drop=True)
        second=first.iloc[partner].reset_index(drop=True)
        mat=pd.concat([first.add_prefix('first_'),second.add_prefix('second_')],axis=1)
        mat['same_booking']=(groups[ix]==groups[ix[partner]]).astype(float)
        matrices.append(mat)
    table=pd.concat(matrices,ignore_index=True)
    paired,other=partners
    train=np.arange(len(t))
    query=np.arange(len(t),len(t)+len(q))
    # Only the training passengers and their training partners supply outcomes.
    jt=2*y[t].astype(int)+y[t[paired]].astype(int)
    bank=predict(table,jt,train,query,seed).reshape(len(q),2,2)
    first=bank[:,1,:].sum(1)
    second=bank[:,:,1].sum(1)[np.argsort(other)]
    result=dict(first=first,second=second,both=(first+second)/2)
    if not shuffled:
        result['binary']=predict(table,y[t],train,query,seed)[:,1]
    return result


def main():
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    row,y,frame,g,_=load()
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        common=(z['votes']!=y[None,:,None]).all((0,2))
        right=(z['votes']==y[None,:,None]).all((0,2))
    path=ART/'nov23_paired_manifest_states'
    path.mkdir(exist_ok=True)
    plans,hist={},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        bank={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            for arm,sham in [('near',False),('random',True)]:
                file=path/f'{seed}_{f}_{arm}.npz'
                if file.is_file():
                    with np.load(file) as z:
                        for key,val in [('t',row[t]),('q',row[q]),('label',y[t])]:np.testing.assert_array_equal(z[key],val)
                        probs={k:z[k] for k in z.files if k not in ('t','q','label')}
                else:
                    probs=pairing(frame,g,y,t,q,sham,f)
                    np.savez_compressed(file,t=row[t],q=row[q],label=y[t],**probs)
                for k,p in probs.items():bank.setdefault(arm+'_'+k,np.full(len(y),np.nan))[q]=p
        hit={'reference':ref==y}
        for k,p in bank.items():
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            hit[k]=(p>=.5)==y
            hit['quarter_'+k]=np.where((op[:,0]>=.5)==ref,.75*op[:,0]+.25*p>=.5,ref)==y
        plans[seed]={k:paired(v,ref==y) for k,v in hit.items()}
        if seed!='v131':
            for k,v in hit.items():hist.setdefault(k,[]).append(v)
        np.savez_compressed(path/f'{seed}.npz',row=row,y=y,base=op[:,0],fold=fold,ref=ref,**bank)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in plans[seed].items() if k.startswith('quarter_')},flush=True)
    base=np.stack(hist['reference'])
    metrics={k:dict(common_correct=int((np.all(v,axis=0)&common).sum()),common_right_missed=int((~np.any(v,axis=0)&right).sum()),
                    gain95_pp=interval((np.stack(v).astype(float)-base).mean(0),g)) for k,v in hist.items()}
    write(ROOT/'reports/nov23_adaptive_manifest_splines/paired_states.json',dict(plans=plans,stable=metrics))


if __name__=='__main__':
    with threadpool_limits(2):main()
