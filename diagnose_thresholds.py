"""Optimistic development-only oracle bounds, never a submission candidate."""
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
import iterate


def oracle_correct(y,p):
    order=np.argsort(p);sorted_p=p[order];sorted_y=y[order]
    # Increasing the threshold flips equal-probability groups together.
    ends=np.r_[np.flatnonzero(np.diff(sorted_p)!=0),len(y)-1]
    gains=np.cumsum(1-2*sorted_y)[ends]
    return int(y.sum()+max(0,int(gains.max())))


if __name__=='__main__':
    train,_,_,dev=iterate.dataset();y=train.Transported.iloc[dev].astype(int).to_numpy()
    records=[]
    for run,name in [('run_012','v34_contrasts_spatial'),('run_019','v44_seed_ensemble')]:
        p=np.load(iterate.ROOT/'artifacts'/run/f'{name}_seed{iterate.SEEDS[0]}.npz')['prob']
        original=int(((p>=.5)==y).sum());best=oracle_correct(y,p)
        fold_oracle=sum(oracle_correct(y[b],p[b]) for _,b in StratifiedKFold(3,shuffle=True,random_state=iterate.SEEDS[0]).split(dev,y))
        records.append(dict(version=name,default_accuracy=original/len(y),oracle_single_threshold_accuracy=best/len(y),
            oracle_net_extra_correct=best-original,oracle_separate_outer_fold_accuracy=fold_oracle/len(y)))
    report=dict(records=records,note='Intentionally optimistic upper bounds using the same development labels for threshold choice and scoring. Not valid performance estimates, not an independent validation, never an exportable model. Used only to reject threshold-only improvement as the main path to >83%.')
    (iterate.ROOT/'reports/threshold_oracle_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
