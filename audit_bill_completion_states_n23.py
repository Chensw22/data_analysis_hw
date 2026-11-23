"""Measure whether competing interpretations of unrecorded bills help select actual mistakes."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from audit_flip_capacity_n23 import required
from broad_bill_completion_states_n23 import LABEL, REPORT
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write


def calibrated(base, zero, paid, neutral, eligible, fold, groups, y):
    z = logit(np.column_stack([base, neutral, zero, paid]).clip(.002, .998))
    x = np.column_stack([z, np.abs(z[:, 2]-z[:, 3])])
    output = {k: base>=.5 for k in ['margin_only', 'margin_and_states']}
    for f in np.unique(fold):
        ids = np.flatnonzero((fold==f)&eligible)
        for a,b in GroupKFold(3,shuffle=True,random_state=447).split(ids,groups=groups[ids]):
            t,q = ids[a],ids[b]
            assert not np.intersect1d(groups[t],groups[q]).size
            for k,features in [('margin_only',x[:,:1]),('margin_and_states',x)]:
                model = LogisticRegression(C=.3, max_iter=250).fit(features[t],y[t])
                output[k][q] = model.predict(features[q])
    return output


def main():
    row,y,frame,groups,_=load()
    wrong,right,hard=cohorts(row,y)
    missing=frame[BILLS].isna().any(axis=1).to_numpy()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=BILLS)
    sizes=dict(missing=int(missing.sum()), hard_missing=int((hard&missing).sum()),
               test_missing=int(test.isna().any(axis=1).sum()), people=len(y))
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_{LABEL}/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,base,ref=z['fold'],z['base'],z['ref']
            zero,paid,neutral=z['missing_zero'],z['missing_paid'],z['missing_marginal']
            conditional=z['conditioned']
        with np.load(ART/f'nov23_service_bag_trees/{seed}.npz') as z:
            for k,v in [('row',row),('y',y),('fold',fold),('base',base),('ref',ref)]:
                np.testing.assert_array_equal(z[k],v)
            direct=z['direct']
        active=ref==(base>=.5)
        eligible=active&missing
        agree=(zero>=.5)==(paid>=.5)
        separate=eligible&~agree
        before=ref==y
        candidates={'known_or_marginal':conditional>=.5,
                    'robust_agreement':np.where(agree,zero>=.5,ref),
                    'either_extreme_midpoint':.5*(paid+zero)>=.5,
                    'quarter_marginal':.75*base+.25*neutral>=.5,
                    'quarter_always_zero':.75*base+.25*zero>=.5,
                    'quarter_always_paid':.75*base+.25*paid>=.5,
                    'quarter_direct_tree':.75*base+.25*direct>=.5,
                    **calibrated(base,zero,paid,neutral,eligible,fold,groups,y)}
        votes={k:np.where(eligible,v,ref) for k,v in candidates.items()}
        votes['all_quarter_conditioned']=np.where(active,.75*base+.25*conditional>=.5,ref)
        votes['complete_quarter_conditioned']=np.where(active&~missing,.75*base+.25*conditional>=.5,ref)
        votes['all_quarter_direct_tree']=np.where(active,.75*base+.25*direct>=.5,ref)
        votes['reference']=ref
        metrics={k:paired(v==y,before) for k,v in votes.items()}
        needed=required(len(y))-int(before.sum())
        plans[seed]=dict(eligible=int(eligible.sum()),actual_errors=int((eligible&~before).sum()),
                         even_perfect_gap=needed-int((eligible&~before).sum()),
                         disagree=int(separate.sum()),disagree_errors=int((separate&~before).sum()),
                         disagree_hard=int((separate&hard).sum()), agree_errors=int((eligible&agree&~before).sum()),
                         metrics=metrics)
        if seed!='v131':
            for k,v in votes.items():
                history.setdefault(k,[]).append(v==y)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in metrics.items()},flush=True)
    prev=np.stack(history['reference'])
    stable={}
    for k,s in history.items():
        h=np.stack(s)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-prev).mean(0),groups))
    write(REPORT/'audit.json',dict(sizes=sizes,plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
