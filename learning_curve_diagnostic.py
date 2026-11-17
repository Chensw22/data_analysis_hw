"""One frozen nested-subset learning curve, no model selection or upload."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
import audit_top5_references as audit
import three_family_mean_screen as history
import iterate

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/learning_curve_20261117'
REPORT=ROOT/'reports/learning_curve_20261117'


def nested_subsets(y,seed):
    rng=np.random.default_rng(seed)
    order={label:rng.permutation(np.flatnonzero(y==label)) for label in [0,1]}
    previous=set(); result={}
    for n in [512,1536,3090]:
        ones=int(round(n*np.mean(y)))
        subset=np.sort(np.r_[order[0][:n-ones],order[1][:ones]])
        assert len(subset)==n and len(np.unique(subset))==n and previous<=set(subset)
        result[n]=subset;previous=set(subset)
    return result


def main():
    if OUT.exists():raise FileExistsError('No repeated sampling paths.')
    own=['learning_curve_diagnostic.py','reports/learning_curve_20261117/protocol.json','audit_top5_references.py','three_family_mean_screen.py','iterate.py','experiment.py']
    commit=history.require_committed(own)
    audit.check_v44_dependencies()
    history.verify_protocol(ROOT/'artifacts/run_019')
    matrices,dev,y,groups,passengers=audit.load_audit_data(iterate)
    splits=list(StratifiedKFold(3,shuffle=True,random_state=20261112).split(dev,y))
    path=ROOT/'artifacts/run_038/v6_no_id_features_seed20261112.npz'
    with np.load(path) as z:
        full=z['prob'].copy();history.cached_fold_scores(full,z['fold_accuracy'],y,splits)
    with np.load(ROOT/'artifacts/run_019/v6_no_id_features_seed20261112.npz') as z:assert np.array_equal(full,z['prob'])
    source={name:audit.file_hash(ROOT/name) for name in own}
    freeze=dict(source_commit=commit,source_sha256=source,full_reference_sha256=audit.file_hash(path),
                environment=audit.file_hash(ROOT/'artifacts/run_019/requirements.lock.txt'),
                model_parameters=iterate.CAT,dev_ids_sha256=audit.row_hash(dev))
    OUT.mkdir(parents=True);audit.json_write(REPORT/'freeze.json',freeze)
    predictions={n:np.full(len(y),np.nan) for n in [512,1536,3090]};predictions[4636]=full
    logs=[];started=time.monotonic()
    for fold,(a,b) in enumerate(splits):
        for n,selected in nested_subsets(y[a],20261117+fold).items():
            ids=dev[a[selected]];assert not np.intersect1d(ids,dev[b]).size
            p,model=iterate.fit(iterate.BASELINE,matrices,ids,y[a[selected]],dev[b],20261112+fold)
            predictions[n][b]=p;del model
            logs.append(dict(fold=fold,training_rows=n,training_ids_sha256=audit.row_hash(ids),
                validation_ids_sha256=audit.row_hash(dev[b]),sampling_seed=20261117+fold,model_seed=20261112+fold,
                accuracy=float(accuracy_score(y[b],p>=.5)),log_loss=float(log_loss(y[b],p))))
            print(f'Learning curve {len(logs)}/9: fold{fold} n{n} accuracy={logs[-1]["accuracy"]:.6f}',flush=True)
            if time.monotonic()-started>600:raise TimeoutError('Fixed CPU diagnostic budget exceeded.')
    result=dict(status='complete',diagnostic_only=True,model_selection=False,automatic_upload=False,
                old_holdout_labels_read=False,fit_count=9,seconds=time.monotonic()-started,folds=logs,metrics=[])
    for n,p in predictions.items():
        assert np.isfinite(p).all()
        row=dict(training_rows=n,accuracy=float(accuracy_score(y,p>=.5)),log_loss=float(log_loss(y,p)),
            fold_accuracy=[float(accuracy_score(y[b],p[b]>=.5)) for _,b in splits],
            fold_log_loss=[float(log_loss(y[b],p[b])) for _,b in splits],
            accuracy_minus_full=float(accuracy_score(y,p>=.5)-accuracy_score(y,full>=.5)),
            log_loss_minus_full=float(log_loss(y,p)-log_loss(y,full)))
        result['metrics'].append(row)
    for name,digest in source.items():assert audit.file_hash(ROOT/name)==digest
    assert audit.file_hash(path)==freeze['full_reference_sha256']
    np.savez(OUT/'oof.npz',dev_ids=dev,**{f'n{n}':p for n,p in predictions.items()})
    result['oof_sha256']=audit.file_hash(OUT/'oof.npz')
    for folder in [OUT,REPORT]:audit.json_write(folder/'result.json',result)
    print(json.dumps(result['metrics'],indent=2))


if __name__=='__main__':main()
