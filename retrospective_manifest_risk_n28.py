"""Fit within-stratum class-balanced outcomes and restore the observed population odds."""
import json

import lightgbm as lgb
import numpy as np
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from conditional_shortcut_boost_n28 import ART,ROOT,baseline,count,experiment_inputs


def run():
    frame,y,f,parent,groups,x,cell,known,fresh,wrong,masks,ordered,anchor=experiment_inputs()
    n=len(y)
    report,saved={},{}
    for tag,parts in [('random',f),('ordered',ordered)]:
        scores=np.empty((len(frame),5))
        for k in range(9):
            t=np.flatnonzero(parts!=k)
            q=np.flatnonzero(parts==k) if k<8 else np.arange(n,len(frame))
            assert not np.isin(groups[t],groups[q]).any()
            prior=baseline(cell,t,y,groups)
            weight=.5/expit(np.where(y[t],1.,-1.)*prior[t])
            # Within each true outcome, permuting weights preserves the global class balance.
            null=weight.copy()
            rng=np.random.default_rng(288)
            for label in [False,True]:
                ix=y[t]==label
                null[ix]=rng.permutation(weight[ix])
            predictions=[]
            for weights in [np.ones(len(t)),weight,null]:
                model=lgb.LGBMClassifier(n_estimators=225,learning_rate=.045,num_leaves=21,min_child_samples=36,
                    reg_lambda=5,verbosity=-1,n_jobs=2,force_col_wise=True)
                model.fit(x[t],y[t],sample_weight=weights/weights.mean())
                predictions.append(model.predict(x[q],raw_score=True,num_threads=2))
            direct,balanced,shuffled=predictions
            # Undo global outcome upsampling in the shuffled control.
            global_shift=np.log(null[y[t]].mean()/null[~y[t]].mean())
            scores[q]=np.column_stack([direct,balanced+prior[q],shuffled-global_shift,balanced,prior[q]])
        saved[tag]=scores
        oppose=(scores>=0)!=parent[:,None]
        rules=dict(restored=oppose[:,1],unique=oppose[:,1]&~oppose[:,0]&~oppose[:,2],
                   incremental=oppose[:,1]&((anchor+scores[:,1]-scores[:,0]>=0)!=parent),
                   exceptional=oppose[:,1]&oppose[:,3]&~oppose[:,4]&~oppose[:,0]&~oppose[:,2])
        report[tag]=dict(accuracy=np.mean((scores[:n]>=0)==y[:,None],axis=0).tolist(),rules={})
        for name,flag in rules.items():
            saved[tag+'_'+name]=flag
            report[tag]['rules'][name]=dict(scopes={s:count(flag[:n],m,wrong) for s,m in masks.items()},
                accuracy=float(np.mean((parent[:n]^flag[:n])[known]==y[known])),test=int(flag[n:].sum()))
        print(tag,report[tag]['rules']['exceptional'],flush=True)
    np.testing.assert_allclose(saved['random'][n:],saved['ordered'][n:])
    dest=ART/'nov28_retrospective_manifest_risk'
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=f,**saved)
    (ROOT/'reports/nov23_matched_learning_curve/retrospective_manifest_risk.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
