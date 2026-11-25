"""Freeze the paired registration and receipt challenge when a majority of model families corroborate it."""
import argparse
import json
import numpy as np
import pandas as pd

from export_consensus_review_n26 import ART, BAND, dig, write
from fare_route_inference_n26 import OUT as ROUTE, route_features
from full_roster_booking_validation_n23 import ROOT, data
from kaggle_workflow import validate_submission
from price_implied_deck_n26 import FEES, OUT as DECK, build_features

NAME='proposal_n26_six_way_dispute'
PARENT='proposal_n26_fare_deck'
OUT=ART/NAME


def roster():
    raw,y,g,_=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    people=pd.concat([raw,test],ignore_index=True)
    group=people.PassengerId.str[:4].to_numpy()
    assert not np.isin(group[len(y):],g).any()
    OUT.mkdir(exist_ok=True)
    return people,group,y,np.arange(len(y),len(people))


def fit(arch):
    people,g,y,test=roster()
    with np.load(DECK/'tree_full_0.npz') as z:
        t,q=z['t'],z['q']
        np.testing.assert_array_equal(z['y'],y[t])
    tr=np.arange(len(y))
    route=route_features(people,g)['exact']
    models=[('neural_route' if arch=='neural' else 'tree_route',route,False,ROUTE/f'{arch}_full_0.npz','exact')]
    if arch=='neural':
        import torch
        from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer
        assert dig(WEIGHT)==DIGEST
        torch.set_num_threads(2)
    else:
        from booking_stream_challenge_n26 import OUT as STREAM, stream
        from receipt_charge_swaps_n26 import OUT as SWAPS
        from solo_selection_likelihood_n26 import fit_model
        reg,local,_=stream(people)
        deck=build_features(people,g)['fine']
        with np.load(SWAPS/'scores.npz') as z:
            np.testing.assert_array_equal(z['passenger'],people.PassengerId)
            swapped=pd.concat([deck,pd.DataFrame(z['exact'],columns=['swapped_'+f for f in FEES])],axis=1)
        models.extend([('swap',swapped,False,SWAPS/'tree_full_0.npz','exact')])
        for key,frame,paid_only in [('register_all',reg,False),('register_paid',reg,True),
                                    ('sequence_paid',pd.concat([reg,local],axis=1),True)]:
            models.append((key,frame,paid_only,STREAM/'tree_full_0.npz',key))
    paid=people[FEES].gt(0).any(axis=1).to_numpy()
    pred={}
    errors={}
    for key,x,paid_only,source,col in models:
        def infer_from(train,query):
            if arch=='neural':
                return infer(x,y,train,query,5583)
            rows=train[paid[train]] if paid_only else train
            return fit_model(x.iloc[rows],y[rows]).predict_proba(x.iloc[query])[:,1]
        with np.load(source) as z:
            np.testing.assert_array_equal(z['q'],q)
            replay=infer_from(t,q)
            errors[key]=float(abs(replay-z[col]).max())
            np.testing.assert_allclose(replay,z[col],atol=3e-5,rtol=0)
            np.testing.assert_array_equal(replay>=.5,z[col]>=.5)
        pred[key]=infer_from(tr,test)
        print(key,errors[key],flush=True)
    np.savez_compressed(OUT/f'{arch}.npz',passenger=people.PassengerId.iloc[test].to_numpy(str),**pred)
    write(OUT/f'{arch}_replay.json',errors)


def combine(parent,challengers,family_votes,eligible):
    prior=parent>=.5
    six=((challengers>=.5)!=prior[:,None]).all(1)
    majority=(family_votes!=prior[:,None]).sum(1)>=10
    return np.where(six&majority&eligible,challengers.mean(1),parent)


def export():
    people,g,y,test=roster()
    ids=people.PassengerId.iloc[test].reset_index(drop=True)
    parent_dir=ART/PARENT
    manifest=json.loads((ROOT/'reports'/PARENT/'manifest.json').read_text())
    assert dig(parent_dir/'submission.csv')==manifest['sha256']
    parent_frame=pd.read_csv(parent_dir/'test_probabilities.csv',float_precision='round_trip')
    labels=pd.read_csv(parent_dir/'submission.csv')
    for f in (parent_frame,labels):
        np.testing.assert_array_equal(f.PassengerId,ids)
    parent=parent_frame.probability.to_numpy()
    np.testing.assert_array_equal(parent>=.5,labels.Transported)
    bank=ART/BAND/'frozen_votes.npz'
    assert dig(bank)==json.loads((ROOT/'reports'/BAND/'manifest.json').read_text())['model_sha256']
    with np.load(ART/'nov26_tabicl_consensus_review/full.npz') as z:
        families=z['names'][:18]
    with np.load(bank) as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        cols=np.array([s.split('/')[0] for s in z['names']])
        masks=np.stack([cols==name for name in families])
        assert (masks.sum(0)==1).all() and masks.any(1).all()
        votes=np.column_stack([z['p'][:,m].mean(1)>=.5 for m in masks])
    scores=[]
    for arch,cols in [('tree',['register_all','register_paid','sequence_paid','tree_route','swap']),
                      ('neural',['neural_route'])]:
        with np.load(OUT/f'{arch}.npz') as z:
            np.testing.assert_array_equal(z['passenger'],ids)
            scores.extend([z[n] for n in cols])
    challengers=np.column_stack(scores)
    assert np.isfinite(challengers).all() and ((challengers>=0)&(challengers<=1)).all()
    with np.load(DECK/'deck_estimates.npz') as z:
        np.testing.assert_array_equal(z['passenger'][test],ids)
        eligible=z['eligible'][test]
    combined=combine(parent,challengers,votes,eligible)
    file=OUT/'predictions.npz'
    np.savez_compressed(file,parent=parent,challengers=challengers,family_votes=votes,eligible=eligible,passenger=ids.to_numpy(str))
    with np.load(file) as z:
        replay=combine(z['parent'],z['challengers'],z['family_votes'],z['eligible'])
    np.testing.assert_array_equal(combined,replay)
    edits=(combined>=.5)!=(parent>=.5)
    assert edits.any()
    pd.DataFrame(dict(PassengerId=ids,Transported=combined>=.5)).to_csv(OUT/'submission.csv',index=False)
    pd.DataFrame(dict(PassengerId=ids,probability=combined,reload_probability=replay)).to_csv(OUT/'test_probabilities.csv',index=False)
    check=validate_submission(OUT/'submission.csv',ROOT/'data/raw/test.csv')
    assert all(json.loads(f.read_text())['file_sha256']!=check['sha256'] for f in (ROOT/'reports').glob('*/online_attempts/*.json'))
    nine=pd.read_csv(ART/'proposal_n24_selected_dissent/submission.csv')
    np.testing.assert_array_equal(nine.PassengerId,ids)
    info=dict(**check,version=NAME.removeprefix('proposal_'),training_rows=len(y),parents={PARENT:manifest['sha256']},
              model_sha256=dig(file),family_input_sha256=dig(bank),reload_difference=0,parent_changes=int(edits.sum()),
              previous_best_changes=int(((combined>=.5)!=nine.Transported).sum()),
              replay_max_errors={k:v for arch in ('tree','neural') for k,v in json.loads((OUT/f'{arch}_replay.json').read_text()).items()})
    selection=dict(version=info['version'],rule='On a fully recorded positive bill, all six challengers and at least ten of eighteen previously frozen families must oppose the parent.',
                   evidence='reports/nov23_matched_learning_curve/surviving_loss_by_vote.json')
    for directory in (OUT,ROOT/'reports'/NAME):
        write(directory/'manifest.json',info)
        write(directory/'selection.json',selection)
    print(info,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['tree','neural','export'])
    mode=parser.parse_args().mode
    if mode=='export':
        export()
    else:
        fit(mode)
