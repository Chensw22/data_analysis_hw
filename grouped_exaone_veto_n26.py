"""Replay the historical EXAONE vote before applying the same recipe to eight booking folds."""
import argparse
import hashlib
import json
import os

if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None, 'FALSE'):
    raise RuntimeError('EXAONE requires KMP_DUPLICATE_LIB_OK=FALSE before numerical imports.')
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'

import numpy as np
import pandas as pd

import exaone_candidate as core
import exaone_validation_data as data
from exaone_validation import coverage
from full_roster_booking_validation_n23 import ROOT,data as booking_data

ART=ROOT/'artifacts'
ONLINE=ROOT/'reports/proposal_online_first_exaone_fixed/manifest.json'
HISTORY=ART/'run_054'
OUT=ART/'nov26_grouped_exaone_veto'


def inputs():
    manifest=json.loads(ONLINE.read_text())
    proof=json.loads((ROOT/data.PROOF).read_text())
    core.verify_checkpoint()
    assert manifest['checkpoint_sha256']==core.WEIGHT_SHA
    assert hashlib.sha256(core.environment_lock().encode()).hexdigest()==manifest['environment_sha256']
    for name,digest in manifest['source_hashes'].items():
        if name.startswith('third_party/exaone/') or name in ('exaone_candidate.py','exaone_mps_compat.py','exaone_validation_data.py'):
            assert core.sha(ROOT/name)==digest,name
    for name in ('data/raw/train.csv','data/raw/test.csv'):
        assert core.sha(ROOT/name)==proof['source_data_sha256'][name],name
    raw,y,groups,fold=booking_data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    frame=data.compact_features(pd.concat([raw,test],ignore_index=True))
    assert data.feature_hash(frame)==proof['compact_values_sha256']
    history=HISTORY/(core.NAME+'_seed20261112.npz')
    summary=json.loads((ROOT/'reports/run_054/summary.json').read_text())
    assert core.sha(history)==summary['result_files_sha256'][history.name]
    with np.load(history) as z:
        rows,hf,hp=z['dev_ids'],z['fold_id'],z['prob']
    with np.load(HISTORY/'fold0_rows.npz') as z:
        t,q=z['train_ids'],z['validation_ids']
        np.testing.assert_array_equal(t,rows[hf!=0])
        np.testing.assert_array_equal(q,rows[hf==0])
    return frame,y,groups,fold,t,q,hp[hf==0]


def predict(frame,y,t,q):
    model=core.fit_context(frame.iloc[t],y[t],t,seed=0,device='mps')
    try:
        prob=core.valid_probabilities(model.predict_proba(frame.iloc[q]),len(q))
        metadata=model.metadata
        assert metadata['training_ids_sha256']==core.array_hash(t)
        assert metadata['labels_sha256']==core.array_hash(y[t].astype(np.int64))
        assert metadata['settings']==core.SETTINGS
        coverage(metadata['last_prediction']['forward_records'],len(t),len(q))
        return prob[:,1].copy()
    finally:
        model.release()


def run(folds,replay):
    frame,y,groups,fold,original_train,original_query,expected=inputs()
    print('Compact feature, row order, checkpoint, environment and model source verified.',flush=True)
    if not replay and not folds:
        return
    if not core.torch.backends.mps.is_available():
        raise RuntimeError('Native MPS required for replay.')
    core.torch.set_num_threads(4)
    core.torch.mps.set_per_process_memory_fraction(.65)
    OUT.mkdir(exist_ok=True)
    evidence=OUT/'historical_fold0.npz'
    if replay or not evidence.exists():
        prob=predict(frame,y,original_train,original_query)
        np.testing.assert_array_equal(prob,expected)
        save(evidence,original_train,original_query,prob,y)
    with np.load(evidence) as z:
        for name,value in [('t',original_train),('q',original_query),('p',expected),('training_labels',y[original_train])]:
            np.testing.assert_array_equal(z[name],value)
    for k in folds:
        q,t=np.flatnonzero(fold==k),np.flatnonzero(fold!=k)
        assert not np.isin(groups[q],groups[t]).any()
        path=OUT/f'fold{k}.npz'
        if path.exists():
            with np.load(path) as z:
                for name,value in [('t',t),('q',q),('training_labels',y[t])]:
                    np.testing.assert_array_equal(z[name],value)
                assert z['p'].shape==(len(q),) and np.isfinite(z['p']).all()
            continue
        prob=predict(frame,y,t,q)
        save(path,t,q,prob,y)
        print('Grouped EXAONE fold',k,'completed',len(t),len(q),flush=True)


def save(path,t,q,prob,y):
    pending=path.with_suffix('.tmp')
    with pending.open('wb') as stream:
        np.savez_compressed(stream,t=t,q=q,p=prob,training_labels=y[t])
    pending.replace(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds',type=int,nargs='+',choices=range(8),default=[])
    parser.add_argument('--replay',action='store_true',help='re-run the original historical fold')
    args=parser.parse_args()
    run(args.folds,args.replay)
