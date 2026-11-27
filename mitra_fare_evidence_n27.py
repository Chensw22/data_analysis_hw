"""Compare a fixed Mitra context with and without cross-booking fare-implied deck evidence."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/nov27_mitra_fare'
REPORT=ROOT/'reports/nov23_matched_learning_curve/mitra_fare_evidence_n27.json'


def prepare():
    import pandas as pd
    from nori_booking_regression_n26 import data,design
    from nori_fare_party_n26 import features
    from sklearn.model_selection import train_test_split
    raw,y,g,f=data()
    full=pd.concat([raw,pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)],ignore_index=True)
    OUT.mkdir(exist_ok=True)
    for k in range(9):
        roster=full if k==8 else raw
        groups=roster.PassengerId.str.split('_').str[0].to_numpy()
        t=np.arange(len(y)) if k==8 else np.flatnonzero(f!=k)
        q=np.arange(len(y),len(full)) if k==8 else np.flatnonzero(f==k)
        assert not np.isin(groups[t],groups[q]).any()
        s=train_test_split(t,train_size=384,stratify=y[t],random_state=932+k)[0]
        x=design(roster,groups,t)
        own,_,_,_=features(roster,groups,t,q)
        posterior=own[:,:8]
        # The posterior mean position and its certainty plus the probability of the registered deck.
        fine=np.column_stack([posterior@np.arange(8),posterior.max(1),own[:,-1]])
        use=np.r_[s,q]
        m=np.column_stack([x,fine])[use]
        m=np.nan_to_num(m,nan=-1.).astype(np.float32)
        np.savez_compressed(OUT/f'input{k}.npz',s=s,t=t,q=q,labels=y[s],m=m,ids=roster.PassengerId.iloc[q].to_numpy(str),width=x.shape[1])


def train(control=False):
    import torch
    import mitra_candidate as engine
    assert torch.backends.mps.is_available()
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.67)
    model,_=engine.load_initial_model(engine.CHECKPOINT_DIR,'mps')
    for k in range(9):
        with np.load(OUT/f'input{k}.npz') as z:
            s,q,labels,m,width=[z[v] for v in ('s','q','labels','m','width')]
            modes=[('shuffled',m.shape[1])] if control else [('raw',int(width)),('fare',m.shape[1])]
            for mode,w in modes:
                path=OUT/f'{mode}{k}.npz'
                if path.exists():
                    with np.load(path) as old:
                        for name in ['s','q','labels','ids']:np.testing.assert_array_equal(old[name],z[name])
                    continue
                inputs=m[:,:w].copy()
                if mode=='shuffled':
                    rng=np.random.default_rng(36710+k)
                    # Independently permute available fare features within planet in support and held query.
                    for domain in (np.arange(len(s)),len(s)+np.arange(len(q))):
                        available=domain[m[domain,-1]>=0]
                        for planet in np.unique(m[available,0]):
                            rows=available[m[available,0]==planet]
                            inputs[rows,int(width):]=m[rng.permutation(rows),int(width):]
                x=torch.as_tensor(inputs,device='mps')
                p=engine.predict_arrays(model,x[:len(s)],torch.as_tensor(labels.astype(int),device='mps'),x[len(s):],chunk=384)[:,1]
                tmp=path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,s=s,q=q,labels=labels,ids=z['ids'],p=p,sha=engine.WEIGHT_SHA)
                tmp.replace(path)
                print(mode,k,len(p),flush=True)


def report():
    import pandas as pd
    from full_roster_booking_validation_n23 import data
    from repeated_archive_misses_n26 import cohort,count
    from audit_clean_fare_anchor_bounds_n26 import read_sources
    raw,y,g,f=data()
    n=len(y)
    b,blind,known,fresh=cohort(raw,y,f)
    mask,versions=read_sources(raw,y,g,f)
    np.testing.assert_array_equal(mask,known)
    ids=np.r_[raw.PassengerId.to_numpy(str),pd.read_csv(ROOT/'data/raw/test.csv').PassengerId.to_numpy(str)]
    results={}
    modes=['raw','fare']
    if (OUT/'shuffled8.npz').exists():modes.append('shuffled')
    for mode in modes:
        p=np.full(len(ids),np.nan)
        for k in range(9):
            with np.load(OUT/f'{mode}{k}.npz') as z:
                q=np.flatnonzero(f==k) if k<8 else np.arange(n,len(ids))
                np.testing.assert_array_equal(z['q'],q)
                np.testing.assert_array_equal(z['ids'],ids[q])
                np.testing.assert_array_equal(z['labels'],y[z['s']])
                p[q]=z['p']
        assert np.isfinite(p).all()
        results[mode]=p
    with np.load(ROOT/'artifacts/nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        foundations=z['scores'][:,-2:]
    out={'support_count':384,'checkpoint':'Mitra v2','new_features':['expected_deck_index','maximum_deck_probability','registered_deck_probability'],'accuracy':{k:float(((p[:n]>=.5)==y).mean()) for k,p in results.items()}}
    flags={}
    paid=pd.read_csv(ROOT/'data/raw/train.csv').drop(columns=['Transported'])
    paid=pd.concat([paid,pd.read_csv(ROOT/'data/raw/test.csv')],ignore_index=True)
    fees=paid[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]
    eligible=fees.notna().all(axis=1).to_numpy()&fees.gt(0).any(axis=1).to_numpy()
    for v,folder in [('111','proposal_n26_champion_fare_six'),('161','proposal_n26_joint_fare_foundations')]:
        test=pd.read_csv(ROOT/'artifacts'/folder/'submission.csv')
        np.testing.assert_array_equal(test.PassengerId,ids[n:])
        base=np.r_[versions[v][0],test.Transported.to_numpy()]
        if v=='161':np.testing.assert_array_equal(base[:n][known],b[known])
        a=(results['raw']>=.5)!=base
        h=(results['fare']>=.5)!=base
        shift=(results['fare']-results['raw'])*np.where(base,-1.,1.)>0
        other=np.sum((foundations>=.5)!=base[:,None],axis=1)
        actions=dict(raw=a,fare=h,added=eligible&h&~a,supported=eligible&h&shift&(other>=1),all_three=eligible&h&shift&(other==2),
                     with_tabicl=eligible&h&shift&((foundations[:,0]>=.5)!=base),with_nori=eligible&h&shift&((foundations[:,1]>=.5)!=base))
        opposite_version='161' if v=='111' else '111'
        other_folder='proposal_n26_joint_fare_foundations' if v=='111' else 'proposal_n26_champion_fare_six'
        other_test=pd.read_csv(ROOT/'artifacts'/other_folder/'submission.csv')
        np.testing.assert_array_equal(other_test.PassengerId,ids[n:])
        delta=base!=np.r_[versions[opposite_version][0],other_test.Transported.to_numpy()]
        actions['adjudicate_50']=delta&h
        actions['adjudicate_50_both']=delta&h&a
        if 'shuffled' in results:
            null=(results['shuffled']>=.5)!=base
            actions['specific']=eligible&h&~a&~null
            actions['specific_with_support']=actions['specific']&(other>=1)
        out[v]={}
        for key,act in actions.items():
            flags[v+'_'+key]=act
            diff=np.where(base[:n]!=y,1,-1)*act[:n]
            out[v][key]=dict(test_changes=int(act[n:].sum()),fold_net=[int(diff[known&(f==i)].sum()) for i in range(8)],
                   scopes={s:count(act[:n],m,base[:n]!=y) for s,m in [('all',known),('fresh',known&fresh),('blind31',blind)]})
    np.savez_compressed(OUT/'aligned.npz',ids=ids,y=y,fold=f,**results,**flags)
    REPORT.write_text(json.dumps(out,indent=2)+'\n')
    print(out['accuracy'],[(k,v['scopes']['all']) for k,v in out['161'].items()])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','train','control','report'])
    command=p.parse_args().command
    if command=='control':train(control=True)
    else:globals()[command]()
