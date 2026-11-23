"""Condition a multiclass prediction on a passenger's observed sleep or spending state."""
import hashlib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from tabicl_routed_context_n23 import DIGEST, WEIGHT, TabICLClassifier


def views(frame):
    sleep = frame.CryoSleep.map({'True': 1, 'False': 0, '__MISSING__': 2})
    assert sleep.notna().all()
    bills = frame[BILLS]
    positive = bills.gt(0)
    any_risk = positive[['VRDeck', 'Spa', 'RoomService']].any(axis=1)
    state = np.where(any_risk, 2, np.where(positive.any(axis=1), 1, 0))
    budget = frame.drop(columns=BILLS).copy()
    budget['amount_recorded'] = np.log1p(bills.sum(axis=1, min_count=1))
    budget['n_unobserved'] = bills.isna().sum(axis=1)
    # The reduced budget view retains total spending but hides which services received it.
    return {'sleep': (frame.drop(columns='CryoSleep'), sleep.to_numpy(int)),
            'basket': (budget, state)}


def predict(table,target,t,q,f):
    model = TabICLClassifier(n_estimators=2, model_path=WEIGHT, allow_auto_download=False, device='cpu',
                             batch_size=1, n_jobs=2, random_state=73+int(f), kv_cache=False,
                             use_amp=False, use_fa3=False)
    model.fit(table.iloc[t],target[t])
    out = model.predict_proba(table.iloc[q])
    np.testing.assert_array_equal(model.classes_, np.arange(out.shape[1]))
    np.testing.assert_allclose(out.sum(1),1,atol=2e-5)
    return out


def main():
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    row,y,frame,g,_=load()
    datasets=views(frame.drop(columns='BookingSize'))
    with np.load(ART/'nov23_release_vote_information/panel.npz') as panel:
        np.testing.assert_array_equal(panel['row'],row)
        np.testing.assert_array_equal(panel['y'],y)
        common=(panel['votes']!=y[None,:,None]).all((0,2))
        right=(panel['votes']==y[None,:,None]).all((0,2))
    output=ART/'nov23_joint_observed_state'
    output.mkdir(exist_ok=True)
    plans, history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,old,_=read_opinions(seed,row,y)
        bank={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            for name,(table,state) in datasets.items():
                raw=output/f'{seed}_{f}_{name}.npz'
                if raw.exists():
                    with np.load(raw) as z:
                        for key,arr in [('training',row[t]),('query',row[q]),('label',y[t]),('state',state[q])]:
                            np.testing.assert_array_equal(z[key],arr)
                        joint,direct=z['joint'],z['direct']
                else:
                    joint=predict(table,2*state+y.astype(int),t,q,f).reshape(len(q),3,2)
                    direct_table=table.assign(observed_state=state.astype(str))
                    direct=predict(direct_table,y.astype(int),t,q,f)[:,1]
                    np.savez_compressed(raw,training=row[t],query=row[q],label=y[t],state=state[q],joint=joint,direct=direct)
                matching=joint[np.arange(len(q)),state[q]]
                conditioned=matching[:,1]/matching.sum(1).clip(1e-12)
                for key,score in [('conditional',conditioned),('marginal',joint[:,:,1].sum(1)),('direct',direct)]:
                    bank.setdefault(f'{name}_{key}',np.full(len(y),np.nan))[q]=score
        for key in ('conditional','direct','marginal'):
            bank['mean_'+key]=(bank['sleep_'+key]+bank['basket_'+key])/2
        hits={'reference':ref==y}
        for k,p in bank.items():
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            hits[k]=(p>=.5)==y
            hits['quarter_'+k]=np.where(ref==(old[:,0]>=.5),.75*old[:,0]+.25*p>=.5,ref)==y
        plans[seed]={k:paired(v,ref==y) for k,v in hits.items()}
        if seed!='v131':
            for k,v in hits.items():history.setdefault(k,[]).append(v)
        np.savez_compressed(output/f'{seed}.npz',row=row,y=y,fold=fold,base=old[:,0],ref=ref,**bank)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in plans[seed].items() if k.startswith('quarter_')},flush=True)
    baseline=np.stack(history['reference'])
    summary={k:dict(stable_common=int((np.all(v,axis=0)&common).sum()), consistently_harmed=int((~np.any(v,axis=0)&right).sum()),
                    gain95_pp=interval((np.stack(v).astype(float)-baseline).mean(0),g)) for k,v in history.items()}
    write(ROOT/'reports/nov23_adaptive_manifest_splines/joint_states.json',dict(plans=plans,stable=summary))


if __name__=='__main__':
    with threadpool_limits(2):main()
