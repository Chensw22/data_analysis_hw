"""Stability audit requested while considering a possible submission exception.

Does not alter the >83% gate, export a submission, or contact Kaggle.
"""
import hashlib
import json
import shutil
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import contrast_features
import iterate

CANDIDATE='v34_contrasts_spatial'
SOURCE=iterate.ROOT/'artifacts/run_012'
OUT=iterate.ROOT/'artifacts/audit_run012'


def paired(y,a,b):
    ac=(a>=.5)==y;bc=(b>=.5)==y
    return dict(baseline_accuracy=float(ac.mean()),candidate_accuracy=float(bc.mean()),
                gain=float(bc.mean()-ac.mean()),corrections=int((~ac&bc).sum()),regressions=int((ac&~bc).sum()))


def main():
    iterate.assert_frozen(SOURCE)
    if OUT.exists():
        raise RuntimeError('Audit already exists; do not overwrite.')
    OUT.mkdir()
    shutil.copy2(__file__,OUT/'audit_candidate_source.py')
    train,_,matrices,dev=iterate.dataset();y=train.Transported.iloc[dev].astype(int).to_numpy()
    rows=[]
    for seed in iterate.SEEDS[1:]:
        a,_,_=iterate.evaluate(iterate.BASELINE,matrices,train,dev,seed,OUT)
        b,_,_=iterate.evaluate(CANDIDATE,matrices,train,dev,seed,OUT)
        row=dict(seed=seed,**paired(y,a,b));rows.append(row)
        print('REPEATED_CV',json.dumps(row),flush=True)
    groups=train.PassengerId.iloc[dev].str.split('_').str[0]
    probabilities={name:np.zeros(len(dev)) for name in [iterate.BASELINE,CANDIDATE]}
    for fold,(a,b) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=iterate.SEEDS[0]).split(dev,y,groups)):
        assert not set(groups.iloc[a])&set(groups.iloc[b])
        for name in probabilities:
            probabilities[name][b]=iterate.fit(name,matrices,dev[a],y[a],dev[b],iterate.SEEDS[0]+fold)[0]
        print('GROUP_AUDIT fold',fold+1,'complete',flush=True)
    group=paired(y,probabilities[iterate.BASELINE],probabilities[CANDIDATE])
    np.savez(OUT/'group_probabilities.npz',**probabilities)
    result=dict(candidate=CANDIDATE,source_run='run_012',screen_accuracy=.8192407247627265,
        repeat_cv=rows,group_cv=group,
        stable_relative_improvement=bool(all(r['gain']>0 for r in rows) and np.mean([r['gain'] for r in rows])>=.001 and group['gain']>0),
        original_83_percent_gate_passed=False,submission_authorized=False,
        note='Same development rows, fresh folds: stability audit, not independent test. Original >83% submission gate remains in force unless user explicitly approves an exception.',
        source_protocol=json.loads((SOURCE/'protocol.json').read_text()),
        audit_source_sha256=hashlib.sha256((iterate.ROOT/'audit_candidate.py').read_bytes()).hexdigest())
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('AUDIT',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
