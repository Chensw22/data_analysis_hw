"""Learn when three raw-data classifiers can jointly reverse the frozen direction model."""
import json
import numpy as np
from scipy.special import logit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from archive_source_harm_veto_n28 import inputs
from repeated_archive_misses_n26 import ART, cohort, count
from full_roster_booking_validation_n23 import ROOT


def run():
    frame,y,g,fold,ids,_,_,seven,versions=inputs()
    n=len(y)
    parent=versions['random'][:,0]
    wrong=parent[:n]!=y
    _,_,known,fresh=cohort(frame.iloc[:n],y,fold)
    signs=np.where(parent,1.,-1.)
    magnitude=np.abs(logit(np.median(seven[:,1:4],axis=1).clip(.003,.997)))
    ordered=np.empty(n,int)
    for k,chunk in enumerate(np.array_split(np.unique(g),8)):
        ordered[np.isin(g,chunk)]=k
    folders=['service_transition_likelihood','receipt_subset_levels','cross_order_texture']
    result={}
    for name,part in [('random',fold),('ordered',ordered)]:
        raw=[]
        for s in folders:
            with np.load(ART/f'nov28_{s}/aligned.npz') as z:
                np.testing.assert_array_equal(z['ids' if s==folders[-1] else 'passenger'],ids)
                np.testing.assert_array_equal(z['fold'],fold)
                np.testing.assert_array_equal(z['y'],y)
                raw.append(z[name+'_z' if s==folders[-1] else name])
        new=np.column_stack([raw[0][:,1], raw[1][:,1], raw[2][:,-1]])*-signs[:,None]
        control=np.column_stack([raw[0][:,0],raw[1][:,0],raw[2][:,1]])*-signs[:,None]
        votes=(new>0).sum(1)
        metadata=np.column_stack([magnitude,parent,((seven>=.5)!=parent[:,None]).sum(1)])
        features=np.column_stack([metadata,new,control,new-control,new.min(1),new.max(1),votes])
        prob=np.full((n,2),np.nan)
        for k in range(8):
            eligible=np.flatnonzero((part==k)&known)
            orders=np.unique(g[eligible])
            if name=='random':
                orders=np.random.default_rng(28250+k).permutation(orders)
            for hold in np.array_split(orders,4):
                inside=np.isin(g[eligible],hold)
                t,q=eligible[~inside],eligible[inside]
                for j,x in enumerate([metadata,features]):
                    standard=StandardScaler().fit(x[t])
                    head=LogisticRegression(C=.10,solver='newton-cholesky').fit(standard.transform(x[t]),wrong[t])
                    prob[q,j]=head.decision_function(standard.transform(x[q]))
        assert np.isfinite(prob[known]).all()
        common=((seven[:n]>=.5)==parent[:n,None]).all(1)&(versions[name][:n]==parent[:n,None]).all(1)
        scope=dict(all=known,new_orders=known&fresh,four_seven=known&common)
        rules={'all_opposed':votes[:n]==3,'majority_with_readout':(votes[:n]>=2)&(prob[:,1]>0),
               'new_over_reference':(votes[:n]>=2)&(prob[:,1]>0)&(prob[:,0]<=0)}
        report={}
        for rule,flag in rules.items():
            report[rule]={s:count(flag,m,wrong) for s,m in scope.items()}
            report[rule]['accuracy']=float(((parent[:n]^flag)[known]==y[known]).mean())
        result[name]=report
    (ROOT/'reports/nov23_matched_learning_curve/adjudicate_three_raw_views.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
