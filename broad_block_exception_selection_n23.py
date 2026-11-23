"""Adjudicate strong block-dependence reversals on other complete bookings in the same outer fold."""
import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS,load
from broad_local_profile_fit_n23 import cohorts,coordinates
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_unexpected_bill_n23 import REPORT


def main():
    row,y,frame,g,_=load()
    wrong,right,hard=cohorts(row,y)
    fees=frame[BILLS]
    population={'awake':frame.CryoSleep.eq('False').to_numpy(), 'asleep':frame.CryoSleep.eq('True').to_numpy(),
                'no_positive_fee':~fees.gt(0).any(axis=1).to_numpy(),'positive_fee':fees.gt(0).any(axis=1).to_numpy(),
                'complete_bills':fees.notna().all(axis=1).to_numpy()}
    history,plans={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_manifest_block_dependence/{seed}.npz') as z:
            for k,v in [('row',row),('y',y)]:
                np.testing.assert_array_equal(z[k],v)
            fold,ref,base=z['fold'],z['ref'],z['base']
            odds=lambda v:logit(v.clip(.002,.998))
            real,synth,sham_real,sham_synth=[odds(z[k]) for k in ('authentic_real','authentic_synthetic','sham_real','sham_synthetic')]
            direct=odds(z['direct'])
        anchor=odds(base)
        proposed=(anchor+real-synth>=0)
        disputed=(ref==(base>=.5)) & (proposed!=ref)
        error=y!=ref
        sign=np.where(ref,-1.,1.)
        risk=np.column_stack([ref,sign*anchor])
        values=np.column_stack([risk,sign*real,sign*synth,sign*(real-synth),sign*direct,sign*(sham_real-sham_synth)])
        routes={'all':disputed,'to_positive':disputed&~ref,'to_negative':disputed&ref}
        for name in ('risk','evidence','context','tree'):
            routes[name]=np.zeros(len(row),bool)
        for f in np.unique(fold):
            ids=np.flatnonzero((fold==f)&disputed)
            for a,b in GroupKFold(3,shuffle=True,random_state=729).split(ids,groups=g[ids]):
                t,q=ids[a],ids[b]
                assert not np.intersect1d(g[t],g[q]).size
                x,_=coordinates(frame,t)
                context=np.column_stack([values,x])
                for name,features in [('risk',risk),('evidence',values),('context',context),('tree',context)]:
                    if name=='tree':
                        model=HistGradientBoostingClassifier(max_leaf_nodes=5,min_samples_leaf=25,max_iter=60,
                                                            early_stopping=False,l2_regularization=8.,random_state=921)
                        model.fit(features[t],error[t])
                        p=model.predict_proba(features[q])[:,1]
                    else:
                        scale=StandardScaler().fit(features[t])
                        model=LogisticRegression(C=.15,max_iter=250).fit(scale.transform(features[t]),error[t])
                        p=model.predict_proba(scale.transform(features[q]))[:,1]
                    routes[name][q]=p>=.5
        hits={k:(~error ^ v) for k,v in routes.items()}
        hits['reference']=~error
        scopes={k:dict(disputed=int((mask&disputed).sum()),actually_wrong=int((mask&disputed&error).sum()),
                       hard_corrected=int((mask&disputed&error&hard).sum())) for k,mask in population.items()}
        metrics={k:paired(h,~error) for k,h in hits.items()}
        plans[seed]=dict(metrics=metrics,scopes=scopes)
        if seed!='v131':
            for k,h in hits.items():
                history.setdefault(k,[]).append(h)
        print(seed,{k:(v['fixed'],v['harmed']) for k,v in metrics.items()},flush=True)
    prev=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       stable_harms=int((right&~h.any(0)).sum()),common_stable=int((wrong&h.all(0)).sum()),
                       gain95_pp=interval((h.astype(float)-prev).mean(0),g))
    write(REPORT/'exception_selection.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
