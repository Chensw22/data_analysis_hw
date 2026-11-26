"""Predict held bookings with unordered category indicators and an isometric rotated control."""
import argparse
import json
from time import monotonic

import numpy as np
import pandas as pd

from nori_booking_regression_n26 import CACHE,ROOT,SHA,data,design
from nori_fare_party_n26 import FEES,features,predictor

OUT=ROOT/'artifacts/nov26_nori_nominal_geometry'
REPORT=ROOT/'reports/nov23_matched_learning_curve/nori_nominal_geometry.json'


def nominal_design(frame,groups,t,q,rotated):
    original=design(frame,groups,t)
    blocks=[]
    rng=np.random.default_rng(5504)
    for col in original[:,:6].T:
        levels=np.unique(col[t])
        # Unseen query categories occupy the all-zero vector in either representation.
        encoded=(col[:,None]==levels).astype(np.float32)
        if rotated:
            orthogonal,_=np.linalg.qr(rng.standard_normal((len(levels),len(levels))))
            np.testing.assert_allclose(orthogonal@orthogonal.T,np.eye(len(levels)),atol=1e-6)
            encoded=encoded@orthogonal
        blocks.append(encoded)
    own,_,_,_=features(frame,groups,t,q)
    return np.column_stack([*blocks, original[:,6:],own]).astype(np.float32)


def train(mode,folds):
    OUT.mkdir(exist_ok=True)
    raw,y,g,f=data()
    torch,regressor=predictor()
    torch.set_num_threads(2)
    for k in folds:
        if k==8:
            test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
            frame=pd.concat([raw,test],ignore_index=True)
            groups=frame.PassengerId.str.split('_').str[0].to_numpy()
            t=np.arange(len(y))
            q=np.arange(len(y),len(frame))
        else:
            frame,groups=raw,g
            t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        assert not np.isin(groups[t],groups[q]).any()
        path=OUT/f'{mode}_{k}.npz'
        seed=6117+(k if k!=8 else 0)
        ids=frame.PassengerId.iloc[q].to_numpy(str)
        recipe=(CACHE/'one_member.json').read_text()
        if path.exists():
            with np.load(path) as z:
                for name,v in [('t',t),('q',q),('labels',y[t]),('passenger',ids),('recipe',recipe),('sha',SHA)]:
                    np.testing.assert_array_equal(z[name],v)
                assert np.isfinite(z['p']).all()
            continue
        start=monotonic()
        x=nominal_design(frame,groups,t,q,mode=='rotated')
        torch.manual_seed(seed)
        np.random.seed(seed)
        model=regressor(model_path=str(CACHE/'nori.pt'), model='nori-6m',device='cpu',
                        inference_config=str(CACHE/'one_member.json'),categorical_columns=[],augmentations=())
        model.fit(x[t],y[t].astype(float))
        p=np.asarray(model.predict(x[q]),float).ravel()
        assert p.shape==(len(q),) and np.isfinite(p).all()
        tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,t=t,q=q,labels=y[t],passenger=ids,recipe=recipe,sha=SHA,p=p,width=x.shape[1])
        tmp.replace(path)
        print(mode,k,len(q),round(monotonic()-start,1),flush=True)


def report():
    from exhaustive_family_quorums_n26 import summary
    from latent_passenger_difficulty_n26 import OUT as PANEL
    from nori_fare_party_n26 import ART as FARE
    raw,y,g,f=data()
    with np.load(PANEL/'panel.npz') as z:
        n=len(y)
        all_ids=z['passenger']
        base=z['anchor'][:n]
        known,common,fresh=[z[k] for k in ['known','common','fresh']]
    pp={}
    for mode in ('ordinal','indicator','rotated'):
        cols=[]
        for k in range(9):
            path=(FARE/(f'own{k}.npz' if k<8 else 'own_test.npz')) if mode=='ordinal' else OUT/f'{mode}_{k}.npz'
            ix=np.flatnonzero(f==k) if k<8 else np.arange(n,len(all_ids))
            with np.load(path) as z:
                np.testing.assert_array_equal(z['passenger'],all_ids[ix])
                np.testing.assert_array_equal(z['q'],ix)
                t=np.flatnonzero(f!=k) if k<8 else np.arange(n)
                np.testing.assert_array_equal(z['t'],t)
                np.testing.assert_array_equal(z['labels'],y[t])
                assert str(z['sha'])==SHA
                cols.append((ix,z['p']))
        arr=np.full(len(all_ids),np.nan)
        for ids,p in cols: arr[ids]=p
        assert np.isfinite(arr).all()
        pp[mode]=arr
    with np.load(PANEL/'panel.npz') as z:
        anchor=z['anchor']
        # Independent fare TabICL is the penultimate column in the aligned panel.
        assert z['names'][-2]=='TabICL fare'
        tab=z['scores'][:,-2]
    votes={k:(v>=.5)!=anchor for k,v in pp.items()}
    actions=dict(**votes, both=votes['indicator']&votes['rotated'],
                 nominal_only=votes['indicator']&votes['rotated']&~votes['ordinal'],
                 with_fare_tabicl=votes['indicator']&votes['rotated']&((tab>=.5)!=anchor))
    awake=raw.CryoSleep.eq(False).to_numpy()
    paid=raw[FEES].gt(0).any(axis=1).to_numpy()
    scopes=dict(all=known,common=common,historical_unseen=known&fresh,awake_paid_negative=common&awake&paid&~base,
                sleeping_no_spend_positive=common&raw.CryoSleep.eq(True).to_numpy()&~paid&base)
    output={name:{k:summary(a[:n],base!=y,m,f) for k,m in scopes.items()} for name,a in actions.items()}
    result=dict(accuracy={k:float(((p[:n]>=.5)==y).mean()) for k,p in pp.items()},
                test_changes={k:int(a[n:].sum()) for k,a in actions.items()},against_161=output)
    for mode in ('indicator','rotated'):
        a=(pp[mode][:n]>=.5)!=(pp['ordinal'][:n]>=.5)
        result[mode+'_vs_ordinal']=summary(a,(pp['ordinal'][:n]>=.5)!=y,np.ones(n,bool),f)
    np.savez_compressed(OUT/'aligned.npz',passenger=all_ids,fold=f,y=y,**pp)
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print(result['accuracy'],[(k,v['all']['fixed'],v['all']['harmed']) for k,v in output.items()])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['indicator','rotated','report'])
    parser.add_argument('--folds',nargs='+',type=int,choices=range(9),default=range(9))
    args=parser.parse_args()
    if args.mode=='report':
        report()
    else:
        train(args.mode,args.folds)
