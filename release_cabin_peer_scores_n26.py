"""Replay and refit the four cabin and party experts used by the confidence review."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from exaone_validation_data import CATEGORICAL, compact_features
from full_roster_booking_validation_n23 import ROOT, data
from global_cabin_tabicl_n26 import ATTRIBUTES

ART = ROOT / 'artifacts'
OUT = ART / 'nov26_review_release'
NAMES = ['cabin_tree', 'cabin_tabicl', 'peer_tree', 'peer_tabicl']


def inputs():
    raw, y, groups, fold = data()
    test = pd.read_csv(ROOT / 'data/raw/test.csv', usecols=raw.columns.tolist())
    roster = pd.concat([raw,test],ignore_index=True)
    group = roster.PassengerId.str.split('_').str[0].to_numpy()
    np.testing.assert_array_equal(group[:len(raw)],groups)
    assert not np.isin(group[len(raw):],groups).any()
    split = [(np.flatnonzero(fold!=0), np.flatnonzero(fold==0)),
             (np.arange(len(y)), np.arange(len(y),len(roster)))]
    OUT.mkdir(exist_ok=True)
    return roster, y, group, split


def conditional(raw, y, g, t, q):
    from confidence_ordered_booking_n26 import parties, person_scores, examples, integrate, classifier
    frame = compact_features(raw)
    x = pd.get_dummies(frame,dtype=float).fillna(-1).to_numpy()
    assert np.array_equal(q,np.flatnonzero(np.isin(g,g[q])))
    p = person_scores(frame,y,g,t,q)
    cabin = raw.Cabin.fillna('?').to_numpy()
    sleep = raw.CryoSleep.astype('string').fillna('?').to_numpy()
    fixed, states, rows, _ = examples(parties(t,g),p,x,cabin,sleep,y)
    test, history, _, desc = examples(parties(q,g),p,x,cabin,sleep)
    model = classifier(310).fit(np.column_stack([fixed,states]),y[rows])
    prob = model.predict_proba(np.column_stack([test,history]))[:,1]
    score, _ = integrate(prob,desc,len(raw))
    return np.where(np.isfinite(score[q]),score[q],p[q])


def trees():
    from catboost import CatBoostClassifier
    from evaluate_global_cabin_assignment_n26 import cabin_views
    raw,y,g,splits=inputs()
    geo=cabin_views(raw)[0]['geometry']
    for label,(t,q) in zip(('replay','test'),splits,strict=True):
        assert max(t)<len(y) and not np.isin(g[t],g[q]).any()
        model=CatBoostClassifier(iterations=440,depth=6,learning_rate=.055,l2_leaf_reg=5,
                                 random_seed=9570,thread_count=2,verbose=False,allow_writing_files=False)
        cabin=model.fit(geo.iloc[t],y[t],cat_features=CATEGORICAL).predict_proba(geo.iloc[q])[:,1]
        # The replay originally encoded categories over the labeled roster.
        n=len(y) if label=='replay' else len(raw)
        peer=conditional(raw.iloc[:n],y,g[:n],t,q)
        save('trees',label,t,q,cabin,peer)
        print('trees',label,flush=True)


def table_models():
    import torch
    from tabicl_routed_context_n23 import WEIGHT, DIGEST, infer
    from joint_party_tabicl_n26 import features, outcome_patterns, joint
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,g,splits=inputs()
    cabin=compact_features(raw)
    with np.load(ART/'nov26_global_cabin_assignment/uncertainty.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId)
        cabin['Deck']=z['deck'];cabin['Side']=z['side'];cabin['CabinNumber']=z['cabin']
        cabin[ATTRIBUTES]=z['values']
    for label,(t,q) in zip(('replay','test'),splits,strict=True):
        assert max(t)<len(y) and not np.isin(g[t],g[q]).any()
        n=len(y) if label=='replay' else len(raw)
        peer,size=features(raw.iloc[:n],g[:n])
        cp=infer(cabin,y,t,q,891)
        target=2*outcome_patterns(y,t,g,size)+y[t]
        pp=joint(peer,target,t,q,size,4605)
        save('tabicl',label,t,q,cp,pp)
        print('tabicl',label,flush=True)


def save(kind,label,t,q,cabin,peer):
    path=OUT/f'context_{kind}_{label}.npz'
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream,t=t,q=q,cabin=cabin,peer=peer)
    path.with_suffix('.tmp').replace(path)


def verify():
    from cross_manifest_views_n26 import predictions
    raw,y,_,splits=inputs()
    _,_,_,fold=data()
    _,after=predictions('full',np.arange(len(y)),fold)
    replay=after[:,[0,2]].reshape(len(y),4)
    report={}
    for label,(t,q) in zip(('replay','test'),splits,strict=True):
        blocks=[]
        for kind in ('trees','tabicl'):
            with np.load(OUT/f'context_{kind}_{label}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                np.testing.assert_array_equal(z['t'],t)
                blocks.append([z['cabin'],z['peer']])
        probs=np.array(blocks).transpose(2,1,0).reshape(len(q),4)
        assert np.isfinite(probs).all() and np.all((probs>=0)&(probs<=1))
        if label=='replay':
            err=np.abs(probs-replay[q]).max(0)
            flips=((probs>=.5)!=(replay[q]>=.5)).sum(0)
            print('replay',err,flips,flush=True)
            assert (err<2e-5).all() and not flips.any()
            report.update(replay_fold=0,replay_rows=len(q),replay_tolerance=2e-5,
                          maximum_absolute_difference=err.tolist(),changed_hard_votes=flips.tolist())
        else:
            path=OUT/'context.npz'
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,names=NAMES,p=probs,passenger=raw.PassengerId.iloc[q].to_numpy(str))
            path.with_suffix('.tmp').replace(path)
            report.update(names=NAMES,train_rows=len(t),test_rows=len(q),shared_bookings=0,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    dest=ROOT/'reports/nov23_matched_learning_curve/release_cabin_peer_scores.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['trees','tabicl','verify'])
    {'trees':trees,'tabicl':table_models,'verify':verify}[parser.parse_args().action]()
