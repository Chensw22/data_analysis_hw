"""Measure overlap and test joint reversals from a maxout network and sparse tree rules."""
import json

import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_supervised_rules_n24 import design,fit_folds


def load_maxout(rows,fold,groups,y,tag):
    out=np.empty(len(rows))
    for f in np.unique(fold):
        q=rows[fold==f]
        t=rows[(fold!=f)&~np.isin(groups[rows],groups[q])]
        with np.load(ROOT/f'artifacts/nov24_maxout/maxout_{tag}_{f}.npz') as z:
            for key,expected in [('q',q),('t',t),('y',y[t])]:
                np.testing.assert_array_equal(z[key],expected)
            out[fold==f]=z['pair'].mean(1)
    return out


def rules(scores,p,protect,labels,group,fold):
    base=p>=.5
    risk=np.minimum(p,1-p)
    opposing=np.where(base[:,None],1-scores,scores)
    vote=opposing>.5
    both=vote.all(1)
    feature=np.column_stack([logit(risk.clip(.001,.999)),logit(opposing.clip(.001,.999))])
    fitted=base.copy()
    for f in np.unique(fold):
        outer=np.flatnonzero(fold==f)
        for a,b in GroupKFold(4,shuffle=True,random_state=462).split(outer,groups=group[outer]):
            t,q=outer[a],outer[b]
            assert not np.isin(group[t],group[q]).any()
            # Include all training rows. In particular, historical protected votes
            # are never used to select calibration rows or generate their targets.
            learner=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=35,
                         learning_rate=.05,l2_regularization=5,monotonic_cst=[1,1,1],random_state=640)
            learner.fit(feature[t],base[t]!=labels[t])
            approved=q[vote[q].any(1)&(learner.predict_proba(feature[q])[:,1]>=.5)]
            fitted[approved]=~base[approved]
    decisions={'either':base ^ vote.any(1),'both':base ^ both,
               'both_more_confident':base ^ (both&(opposing.min(1)>1-risk)),
               'monotone':fitted}
    return {k:np.where(protect!=base,protect,v) for k,v in decisions.items()}


def run():
    _,y,g,rows,oldfold,_,champ,_,_,_,_,old,_=aligned()
    raw,labels,group,fold=data()
    names,half,full=banks(raw,labels,fold)
    ref=latest_reference(labels,group,fold,names,half,full)
    gate,_=gate_probabilities(names,half,full,labels,group,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    all_rows=np.arange(len(raw))
    x=design(raw)
    def experts(ix,splits,tag):
        prob,_=fit_folds(x,labels,group,ix,splits)
        return np.column_stack([load_maxout(ix,splits,group,labels,tag),prob['disjoint_orders']])
    a,b=experts(rows,oldfold,'historical'),experts(all_rows,fold,'eightfold')
    original=rules(a,old[:,0],champ,y,g,oldfold)
    current=rules(b,gate,ref,labels,group,fold)
    extra=~np.isin(group,g)
    shared=((full>=.5)!=labels[:,None]).all(1)
    paired=shared[rows]&(champ!=y)
    old_correct=(a>=.5)==y[:,None]
    new_correct=(b[rows]>=.5)==y[:,None]
    repeated=old_correct&new_correct
    coverage=dict(raw_either_can_fix_257=int((shared&((b>=.5)==labels[:,None]).any(1)).sum()),
                  raw_both_can_fix_257=int((shared&((b>=.5)==labels[:,None]).all(1)).sum()),
                  persistent_201_by_model=repeated[paired].sum(0).tolist(),
                  persistent_201_either=int((paired&repeated.any(1)).sum()),
                  persistent_201_both=int((paired&repeated.all(1)).sum()))
    results={}
    for k,pred in current.items():
        def measure(z,anchor,target,grp,splits):
            d=(z==target).astype(int)-(anchor==target)
            return dict(fixed=int((d==1).sum()),harmed=int((d==-1).sum()),accuracy=float((z==target).mean()),
                        ci_pp=interval(d,grp),fold_net=[int(d[splits==f].sum()) for f in np.unique(splits)])
        results[k]=dict(historical=measure(original[k],champ,y,g,oldfold),full=measure(pred,ref,labels,group,fold),
                    extra_new_orders=measure(pred[extra],ref[extra],labels[extra],group[extra],fold[extra]),
                    paired_fixed=int((paired&(original[k]==y)&(pred[rows]==y)).sum()),
                    common_fixed=int((shared&(pred==labels)).sum()))
    print(coverage,[(k,v['historical']['fixed']-v['historical']['harmed'],v['full']['fixed']-v['full']['harmed'],
                     v['extra_new_orders']['fixed']-v['extra_new_orders']['harmed']) for k,v in results.items()])
    path=ROOT/'reports/nov23_matched_learning_curve/joint_raw_dissent.json'
    path.write_text(json.dumps(dict(coverage=coverage,results=results),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
