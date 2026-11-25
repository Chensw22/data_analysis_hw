"""Measure direction-specific error separability and repeated failure across inner and outer Mitra teachers."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from full_roster_booking_validation_n23 import ROOT,data
from matched_error_ranker_n25 import SOURCE

REPORT=ROOT/'reports/nov23_matched_learning_curve/robust_error_directions.json'


def run():
    raw,y,g,fold=data()
    audit=pd.read_csv(ROOT/'artifacts/nov25_within_bin_bill_precision/passenger_error_audit.csv',dtype={'PassengerId':str})
    np.testing.assert_array_equal(audit.PassengerId,raw.PassengerId)
    np.testing.assert_array_equal(audit.transported,y)
    np.testing.assert_array_equal(audit.booking_fold,fold)
    votes=audit.original_votes_true.to_numpy()
    majority=votes>34
    common=(votes==0)|(votes==68)
    wrong=majority!=y
    solo=~pd.Series(g).duplicated(keep=False).to_numpy()
    awake=raw.CryoSleep.eq(False).to_numpy()
    p=np.empty(len(y))
    errors=np.zeros(len(y),int)
    repeats=np.zeros(len(y),int)
    total_shared=0
    for k in range(8):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        with np.load(SOURCE/f'release{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            p[q]=z['native']
        with np.load(SOURCE/f'outer{k}.npz') as z:
            np.testing.assert_array_equal(z['t'],t)
            np.testing.assert_array_equal(z['labels'],y[t])
            errors[t]+=(z['p']>=.5)!=y[t]
            repeats[t]+=1
            for i in np.unique(z['fold']):
                sub=z['fold']==i
                total_shared+=int(np.isin(g[t[sub]],g[t[~sub]]).sum())
    assert (repeats==7).all()
    scores=np.where(majority,1-p,p)
    common_wrong=common&wrong
    groups={}
    for scope,within in [('all',np.ones(len(y),bool)),('new',audit.new_order.to_numpy()),('awake_solo',awake&solo)]:
        for label in (False,True):
            m=within&common&(majority==label)
            if not m.any():
                continue
            target=wrong[m]
            groups[f'{scope}/predicted_{label}']=dict(passengers=int(m.sum()),mistakes=int(target.sum()),
                auc=float(roc_auc_score(target,scores[m])) if np.unique(target).size==2 else None,
                median_if_wrong=float(np.median(scores[m&wrong])) if (m&wrong).any() else None,
                median_if_correct=float(np.median(scores[m&~wrong])) if (m&~wrong).any() else None)
    result=dict(per_direction=groups, repeated_inner_teacher=dict(common_errors=int(common_wrong.sum()),
        all_seven_wrong=int((common_wrong&(errors==7)).sum()),wrong_predictions=int(errors[common_wrong].sum()),
        opportunities=int(repeats[common_wrong].sum()), inner_queries_with_fellow_booking_in_training=total_shared,
        inner_queries=int(repeats.sum())))
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run()
