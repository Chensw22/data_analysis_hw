"""Share a target for unexpected transport among payers and unexpected nontransport among others."""
import hashlib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import CATEGORICAL, load
from broad_joint_observed_state_n23 import predict
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from tabicl_routed_context_n23 import DIGEST, WEIGHT


def state_table(frame):
    table=frame.drop(columns='BookingSize').copy()
    usual=~table[['VRDeck','Spa','RoomService']].gt(0).any(axis=1).to_numpy()
    table['usual_decision']=np.where(usual, 'transport', 'stay')
    return table,usual


def tree(table,y,t,q,f):
    model=CatBoostClassifier(iterations=350,depth=5,learning_rate=.055,l2_leaf_reg=4,thread_count=2,
                             random_seed=73+int(f),allow_writing_files=False,verbose=False)
    model.fit(table.iloc[t],y[t],cat_features=[*CATEGORICAL,'usual_decision'])
    return model.predict_proba(table.iloc[q])[:,1]


def run_fold(table,y,usual,t,q,f,framework):
    def fit(target,ti,qi):
        if framework=='tree':
            return tree(table,target,ti,qi,f)
        return predict(table,target,ti,qi,f)[:,1]
    direct=fit(y,t,q)
    unexpected=fit(y!=usual,t,q)
    separate=np.zeros(len(q))
    for state in (False,True):
        sub=usual[q]==state
        separate[sub]=fit(y,t[usual[t]==state],q[sub])
    return dict(direct=direct,exception=np.where(usual[q],1-unexpected,unexpected),separate=separate)


def main():
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    row,y,frame,groups,_=load()
    table,usual=state_table(frame)
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        common=(z['votes']!=y[None,:,None]).all((0,2))
        right=(z['votes']==y[None,:,None]).all((0,2))
    out=ART/'nov23_exception_state_coding'
    out.mkdir(exist_ok=True)
    plans,hist={},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        bank={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(groups,groups[q]))
            assert not np.intersect1d(groups[t],groups[q]).size
            for family in ('tree','tabicl'):
                file=out/f'{seed}_{f}_{family}.npz'
                if file.is_file():
                    with np.load(file) as z:
                        np.testing.assert_array_equal(row[t],z['train'])
                        np.testing.assert_array_equal(row[q],z['query'])
                        np.testing.assert_array_equal(y[t],z['label'])
                        scores={k:z[k] for k in ['direct','exception','separate']}
                else:
                    scores=run_fold(table,y,usual,t,q,f,family)
                    np.savez_compressed(file,train=row[t],query=row[q],label=y[t],**scores)
                for k,p in scores.items():bank.setdefault(family+'_'+k,np.full(len(y),np.nan))[q]=p
        for mode in ['direct','exception','separate']:
            bank['two_'+mode]=(bank['tree_'+mode]+bank['tabicl_'+mode])/2
        hits={'reference':ref==y}
        for k,p in bank.items():
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            hits[k]=(p>=.5)==y
            wins=.75*op[:,0]+.25*p>=.5
            hits['quarter_'+k]=np.where((op[:,0]>=.5)==ref,wins,ref)==y
        plans[seed]={k:paired(v,ref==y) for k,v in hits.items()}
        if seed!='v131':
            for k,v in hits.items():hist.setdefault(k,[]).append(v)
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,base=op[:,0],ref=ref,**bank)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in plans[seed].items() if k.startswith('quarter_')},flush=True)
    baseline=np.stack(hist['reference'])
    summary={k:dict(stable_common=int((np.all(v,axis=0)&common).sum()),stable_exception=int((np.all(v,axis=0)&common&(y!=usual)).sum()),
                    stable_misses=int((~np.any(v,axis=0)&right).sum()),gain95_pp=interval((np.stack(v).astype(float)-baseline).mean(0),groups)) for k,v in hist.items()}
    write(ROOT/'reports/nov23_adaptive_manifest_splines/exception_coding.json',dict(plans=plans,stable=summary))


if __name__=='__main__':
    with threadpool_limits(2):main()
