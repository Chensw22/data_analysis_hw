"""Test assignment uncertainty in a pretrained table transformer across two booking splits."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT, data

OUT=ROOT/'artifacts/nov26_global_cabin_assignment'
ATTRIBUTES=['choice_count','observed_surprise','expected_cabin','choice_entropy']


def roster():
    raw,y,g,f=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    return pd.concat([raw,test],ignore_index=True),y,g,f


def prepare():
    from evaluate_global_cabin_assignment_n26 import cabin_views
    frame,_,_,_=roster()
    views,_,_,_=cabin_views(frame)
    view=views['geometry']
    np.savez_compressed(OUT/'uncertainty.npz',passenger=frame.PassengerId.to_numpy(str),values=view[ATTRIBUTES].to_numpy(),
                        deck=view.Deck.to_numpy(str),side=view.Side.to_numpy(str),cabin=view.CabinNumber.to_numpy())


def train():
    import torch
    from tabicl_routed_context_n23 import DIGEST, WEIGHT, infer
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    frame,y,g,f=roster()
    plain=compact_features(frame)
    features=plain.copy()
    with np.load(OUT/'uncertainty.npz') as z:
        np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
        features['Deck']=z['deck']
        features['Side']=z['side']
        features['CabinNumber']=z['cabin']
        features[ATTRIBUTES]=z['values']
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    for tag,ids,fold in [('historical',rows,hf),('full',np.arange(len(y)),f)]:
        for k in np.unique(fold):
            q=ids[fold==k]
            t=ids[(fold!=k)&~np.isin(g[ids],g[q])]
            assert not np.isin(g[t],g[q]).any()
            path=OUT/f'tabicl_{tag}_{k}.npz'
            if not path.exists():
                if tag=='full':
                    with np.load(ROOT/f'artifacts/nov25_passenger_state_contexts/{k}_global.npz') as z:
                        for key,value in [('t',t),('q',q),('train_y',y[t]),('checkpoint',DIGEST)]:
                            np.testing.assert_array_equal(z[key],value)
                        base=z['p']
                else:
                    base=infer(plain,y,t,q,891+int(k))
                pred=infer(features,y,t,q,891+int(k))
                with path.with_suffix('.tmp').open('wb') as buf:
                    np.savez_compressed(buf,t=t,q=q,train_y=y[t],base=base,p=pred,checkpoint=DIGEST)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for key,value in [('t',t),('q',q),('train_y',y[t]),('checkpoint',DIGEST)]:
                    np.testing.assert_array_equal(z[key],value)
                assert np.isfinite(z['p']).all() and np.isfinite(z['base']).all()
            print(tag,k,flush=True)


def score():
    from family_identity_transfer_n26 import panels_and_reference
    from outstanding_error_families_n26 import observed
    raw,y,g,f=data()
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ROOT/'artifacts/nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0]=z['prior']
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    current=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    results={}
    for tag,ids,fold,nat,view in [('historical',rows,hf,old_native,hist),('full',np.arange(len(y)),f,native,current)]:
        probabilities=np.zeros((len(ids),2))
        clean=~np.isin(g[ids],g[rows])
        for k in np.unique(fold):
            loc=fold==k
            q=ids[loc]
            with np.load(OUT/f'tabicl_{tag}_{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                probabilities[loc]=np.column_stack([z['base'],z['p']])
            if tag=='historical':
                clean[loc]=~np.isin(g[q],g[ids[~loc]])
        old,new=probabilities.T
        anchor=view['anchor']
        common=view['eligible']&(view['votes']==anchor[:,None]).all(1)
        edits={'direct':(new>=.5)!=anchor,'plain':(old>=.5)!=anchor,
               'induced':((old>=.5)!=(new>=.5))&((new>=.5)!=anchor),
               'quarter':(anchor==(nat>=.5))&((.75*nat+.25*new>=.5)!=anchor)}
        report={}
        for method,edit in edits.items():
            sub={}
            for scope,mask in [('all',view['known']),('new',view['known']&clean),('families',common),('new_families',common&clean)]:
                wrong=anchor!=y[ids]
                change=edit&mask
                sub[scope]=dict(rows=int(mask.sum()),fixed=int((change&wrong).sum()),harmed=int((change&~wrong).sum()),
                                by_fold=[int((change&wrong&(fold==k)).sum())-int((change&~wrong&(fold==k)).sum()) for k in np.unique(fold)])
            report[method]=sub
        results[tag]=report
        print(tag,[(a,r['all']['fixed'],r['all']['harmed']) for a,r in report.items()])
    (ROOT/'reports/nov23_matched_learning_curve/global_cabin_tabicl.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','train','score'])
    {'prepare':prepare,'train':train,'score':score}[parser.parse_args().action]()
