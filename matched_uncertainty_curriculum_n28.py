"""Train a fresh specialist on difficult bookings while preserving the local class distribution."""
import json

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from conditional_shortcut_boost_n28 import experiment_inputs
from repeated_archive_misses_n26 import ART, count
from full_roster_booking_validation_n23 import ROOT


def tree(train,query,x,y,weight=None):
    model=lgb.LGBMClassifier(n_estimators=160,num_leaves=17,min_child_samples=32,learning_rate=.055,
                             reg_lambda=5,verbosity=-1,n_jobs=2,force_col_wise=True)
    return model.fit(x[train],y[train],sample_weight=weight).predict(x[query],raw_score=True,num_threads=2)


def fit(t,q,x,y,g):
    train_scores=np.empty(len(t))
    for a,b in GroupKFold(3,shuffle=True,random_state=28291).split(t,groups=g[t]):
        train_scores[b]=tree(t[a],t[b],x,y)
    raw=tree(t,q,x,y)
    # Training weights depend only on out-of-fold confidence.
    uncertainty=np.argsort(np.argsort(abs(train_scores)))/len(t)
    weights=.35+2.65*(1-uncertainty)**3
    sham=weights.copy()
    rng=np.random.default_rng(28292)
    for label in [False,True]:
        use=train_scores>=0 if label else train_scores<0
        sham[use]=rng.permutation(weights[use])
    outputs=[tree(t,q,x,y,v/v.mean()) for v in [weights,sham]]
    return np.column_stack([raw,*outputs]), np.quantile(abs(train_scores),.3)


def run():
    frame,y,fold,parent,g,x,_,known,_,wrong,masks,ordered,anchor=experiment_inputs()
    n=len(y); archives={}; report={}
    for tag,parts in [('random',fold),('ordered',ordered)]:
        scores=np.empty((len(frame),3)); ambiguous=np.zeros(len(frame),bool)
        for k in range(9):
            t=np.flatnonzero(parts!=k)
            q=np.flatnonzero(parts==k) if k<8 else np.arange(n,len(frame))
            assert not np.isin(g[t],g[q]).any()
            if k==8 and tag=='ordered':
                scores[q]=archives['random'][q];ambiguous[q]=archives['random_ambiguous'][q]
            else:
                scores[q],bound=fit(t,q,x,y,g);ambiguous[q]=abs(scores[q,0])<=bound
        plain,trained,shuffled=scores.T
        oppose=(trained>=0)!=parent
        inc=oppose&((anchor+trained-plain>=0)!=parent)
        conf=inc&((anchor+trained-shuffled>=0)!=parent)
        rules={'direct':oppose,'uncertain_increment':ambiguous&inc,'supported_in_uncertain':ambiguous&conf,
               'supported_all':conf}
        report[tag]={'raw_accuracy':np.mean((scores[:n]>=0)==y[:,None],axis=0).tolist(),'rules':{}}
        for name,flag in rules.items():
            report[tag]['rules'][name]=dict(scopes={s:count(flag[:n],m,wrong) for s,m in masks.items()},
                accuracy=float(((parent[:n]^flag[:n])[known]==y[known]).mean()), test=int(flag[n:].sum()))
        archives[tag]=scores;archives[tag+'_ambiguous']=ambiguous
        print(tag, report[tag],flush=True)
    dest=ART/'nov28_matched_uncertainty_curriculum';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=fold,**archives)
    (ROOT/'reports/nov23_matched_learning_curve/matched_uncertainty_curriculum.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
