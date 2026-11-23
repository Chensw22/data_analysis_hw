"""Learn adoption on dissenting records in separate bookings, and track the 34 stable rescues."""
import numpy as np
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_opposing_passengers_n23 import REPORT
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write
from exaone_validation_data import CATEGORICAL


def decisions(data,frame,y,g):
    base,ref,fold=data['base'],data['ref'],data['fold']
    candidate=data['matched_conditional']
    eligible=((candidate>=.5)!=ref)&(ref==(base>=.5))
    # The adoption target is defined only when changing the decision can affect accuracy.
    benefit=ref!=y
    sign=np.where(ref,-1.,1.)
    results={'all_disputes':np.where(eligible,~ref,ref),'reference':ref}
    arms={}
    for k in ('binary','matched','shuffled'):
        if k=='binary':
            inputs=np.column_stack([sign,sign*logit(base.clip(.002,.998)),sign*logit(data[k].clip(.002,.998))])
        else:
            inside=data[k+'_conditional']
            mass=(data[k]-data[k+'_neutral'])/(inside-.5)
            other=(data[k]-mass*inside)/(1-mass)
            inputs=np.column_stack([sign,sign*logit(base.clip(.002,.998)),logit(mass),
                                    sign*logit(inside),sign*logit(other),sign*logit(data[k])])
        arms[k]=inputs
    for f in np.unique(fold):
        ids=np.flatnonzero(fold==f)
        for a,b in GroupKFold(3,shuffle=True,random_state=719).split(ids,groups=g[ids]):
            t,q=ids[a],ids[b]
            assert not np.intersect1d(g[t],g[q]).size
            t,q=t[eligible[t]],q[eligible[q]]
            for k,x in arms.items():
                scaler=StandardScaler().fit(x[t])
                lr=LogisticRegression(C=.15,max_iter=500).fit(scaler.transform(x[t]),benefit[t])
                p=lr.predict_proba(scaler.transform(x[q]))[:,1]
                results.setdefault(k+'_scores',ref.copy())[q]=ref[q]^(p>=.5)
                table=frame.copy()
                for col,values in enumerate(x.T):
                    table['probability_'+str(col)]=values
                tree=CatBoostClassifier(iterations=120,depth=3,learning_rate=.04,l2_leaf_reg=8,
                                        random_seed=327,thread_count=2,verbose=False,allow_writing_files=False)
                tree.fit(table.iloc[t],benefit[t],cat_features=CATEGORICAL)
                p=tree.predict_proba(table.iloc[q])[:,1]
                results.setdefault(k+'_context',ref.copy())[q]=ref[q]^(p>=.5)
    return results


def main():
    row,y,frame,g,_=load()
    wrong,right,hard=cohorts(row,y)
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_opposing_passengers/{seed}.npz') as z:
            np.testing.assert_array_equal(row,z['row'])
            np.testing.assert_array_equal(y,z['y'])
            results=decisions(z,frame,y,g)
            ref=z['ref']
        hits={k:v==y for k,v in results.items()}
        plans[seed]={k:paired(h,ref==y) for k,h in hits.items()}
        if seed!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in plans[seed].items()},flush=True)
    baseline=np.stack(history['reference'])
    expert=hard & np.all(history['all_disputes'],axis=0)
    assert expert.sum()==34
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       recovered_34_ever=int((expert&h.any(0)).sum()),common_stable=int((wrong&h.all(0)).sum()),
                       stable_harms=int((right&~h.any(0)).sum()),gain95_pp=interval((h.astype(float)-baseline).mean(0),g))
    write(REPORT/'dispute_decisions.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
