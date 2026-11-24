"""Test whether per-passenger changes after retraining predict errors of the full-roster rule."""
import json

import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold

from broad_booking_policy_replay_n23 import interval
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_repartition_gate_n24 import ROOT, panel_data
from full_roster_same_model_calibration_n23 import learn


def run():
    raw,y,g,fold,names,old,full,ref,base,cols,rotated=panel_data()
    gate,_=gate_probabilities(names,old,full,y,g,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    def logits(p):
        return logit(p.clip(.005,.995))
    z=logits(full[:,cols])
    oldz,newz=logits(old[:,cols]),logits(rotated)
    shift=(oldz+newz)/2-z
    spread=abs(oldz-newz)
    intercept=np.ones(len(y))
    base_x=np.column_stack([intercept,logits(gate),z])
    designs={'scores_only':base_x,'training_shift':np.column_stack([base_x,shift]),
             'shift_and_spread':np.column_stack([base_x,shift,spread])}
    small_mean=(old[:,:5]+rotated[:,:5]).mean(1)/2
    predicts={'small_mean_mix':(.75*gate+.25*small_mean)>=.5}
    for k,x in designs.items():
        pred=np.zeros(len(y),bool)
        for f in np.unique(fold):
            outer=np.flatnonzero(fold==f)
            for a,b in GroupKFold(4,shuffle=True,random_state=462).split(outer,groups=g[outer]):
                t,q=outer[a],outer[b]
                assert not np.isin(g[t],g[q]).any()
                w=learn(x[t],y[t],logits(gate[t]),np.zeros(x.shape[1]))
                pred[q]=expit(logits(gate[q])+x[q]@w)>=.5
        predicts[k]=pred
    shared=((full>=.5)!=y[:,None]).all(1)
    all_right=((full>=.5)==y[:,None]).all(1)
    all_small=np.column_stack([old[:,:5],rotated[:,:5]])
    dissent=(all_small>=.5)!=ref[:,None]
    coverage={}
    for n in [1,3,5,8,10]:
        challenged=dissent.sum(1)>=n
        coverage[str(n)]=dict(coverable=int((challenged&shared).sum()),unanimous_correct_challenged=int((challenged&all_right).sum()))
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as zfile:
        dev=np.sort(zfile['development'])
    report={}
    middle=(gate>.3)&(gate<.7)
    for k,pred in predicts.items():
        delta=(pred==y).astype(int)-(ref==y)
        report[k]=dict(accuracy=float((pred==y).mean()),fixed=int((delta==1).sum()),harmed=int((delta==-1).sum()),
                       common_fixed=int((delta[shared]==1).sum()),middle_net=int(delta[middle].sum()),
                       additional_net=int(delta.sum()-delta[dev].sum()),fold_net=[int(delta[fold==f].sum()) for f in np.unique(fold)],
                       ci_pp=interval(delta,g))
    (ROOT/'reports/nov23_matched_learning_curve/training_sensitivity.json').write_text(
        json.dumps(dict(results=report,coverage=coverage),indent=2)+'\n')
    print(report,coverage)


if __name__=='__main__':
    run()
