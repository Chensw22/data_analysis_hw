"""Estimate the cost of reaching broad shares of shared errors on held-out bookings."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from archive_source_harm_veto_n28 import inputs
from full_roster_booking_validation_n23 import ROOT
from repeated_archive_misses_n26 import ART, cohort


def run():
    frame,y,g,fold,ids,_,_,opinions,versions=inputs()
    n=len(y)
    _,_,known,_=cohort(frame.iloc[:n],y,fold)
    parent=versions['random'][:n,0]
    wrong=parent!=y
    signed=np.where(parent,-1.,1.)
    seven=logit(opinions[:n].clip(.002,.998))*signed[:,None]
    ordered=np.empty(n,int)
    for k,groups in enumerate(np.array_split(np.unique(g),8)):
        ordered[np.isin(g,groups)]=k
    budgets=[.05,.20,.40]
    result={}
    for name,partition in [('random',fold),('ordered',ordered)]:
        common=known&(versions[name][:n]==parent[:,None]).all(1)&((opinions[:n]>=.5)==parent[:,None]).all(1)
        views=[]
        for file,col in [('service_transition_likelihood',1),('receipt_subset_levels',1),('cross_order_texture',-1)]:
            with np.load(ART/f'nov28_{file}/aligned.npz') as z:
                np.testing.assert_array_equal(z['ids' if file=='cross_order_texture' else 'passenger'],ids)
                val=z[name+'_z' if file=='cross_order_texture' else name][:n,col]
                views.append(val*signed)
        scores=np.column_stack([seven,*views,parent])
        budgets_hit=np.zeros((n,2,len(budgets)),bool)
        majority=np.zeros((n,2),bool)
        for k in range(8):
            rows=np.flatnonzero((partition==k)&common)
            groups=np.unique(g[rows])
            if name=='random':
                groups=np.random.default_rng(28290+k).permutation(groups)
            for held in np.array_split(groups,4):
                mask=np.isin(g[rows],held)
                t,q=rows[~mask],rows[mask]
                for j,x in enumerate([np.median(seven,axis=1)[:,None],scores]):
                    normal=StandardScaler().fit(x[t])
                    model=LogisticRegression(C=.15,solver='newton-cholesky').fit(normal.transform(x[t]),wrong[t])
                    train_risk=model.decision_function(normal.transform(x[t]))
                    risk=model.decision_function(normal.transform(x[q]))
                    majority[q,j]=risk>0
                    for b,size in enumerate(budgets):
                        budgets_hit[q,j,b]=risk>=np.quantile(train_risk,1-size)
        def measure(flag):
            fixed=int((flag&common&wrong).sum())
            harmed=int((flag&common&~wrong).sum())
            return dict(fixed=fixed,harmed=harmed,precision=fixed/max(1,fixed+harmed),recall=fixed/int((common&wrong).sum()))
        report={'passengers':int(common.sum()),'errors':int((common&wrong).sum())}
        for j,title in enumerate(['seven_median','seven_plus_raw']):
            report[title]={str(s):measure(budgets_hit[:,j,b]) for b,s in enumerate(budgets)}
            report[title]['estimated_majority_wrong']=measure(majority[:,j])
        result[name]=report
    path=ROOT/'reports/nov23_matched_learning_curve/shared_risk_coverage.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
