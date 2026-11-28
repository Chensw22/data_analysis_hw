"""Learn an exception policy on pooled inner out-of-fold predictions and apply it once per outer order."""
import json
import lightgbm as lgb
import numpy as np
from scipy.special import expit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from conditional_shortcut_boost_n28 import ART,ROOT,baseline,count,experiment_inputs
from joint_exception_frequency_n28 import estimate,features


def fit_scores(x,cell,groups,y,train,q):
    prior=baseline(cell,train,y,groups)
    weight=.5/expit(np.where(y[train],1.,-1.)*prior[train])
    preds=[]
    for offset,weights in [(np.zeros(len(train)),np.ones(len(train))),(prior[train],np.ones(len(train))),
                           (np.zeros(len(train)),weight/weight.mean())]:
        def objective(z,data):
            p=expit(z+offset)
            return weights*(p-y[train]),weights*np.maximum(p*(1-p),1e-5)
        params=dict(objective=objective,num_threads=1,verbosity=-1,force_col_wise=True,
                    learning_rate=.045,num_leaves=21,min_data_in_leaf=36,lambda_l2=5)
        model=lgb.train(params,lgb.Dataset(x[train],label=y[train]),num_boost_round=225)
        preds.append(model.predict(x[q],raw_score=True,num_threads=1))
    return np.column_stack([prior[q],preds[0],preds[1]+prior[q],preds[2]+prior[q]])


def run():
    frame,y,f,parent,groups,x,cell,known,_,wrong,masks,ordered,_=experiment_inputs()
    n=len(y)
    # Use a separately trained base tree as the reference for learning error patterns.
    # Comparing exceptions to that tree keeps all policy labels independent of the outer orders.
    saved,report={},{}
    for scheme,partition in [('random',f),('ordered',ordered)]:
        flags=np.zeros((len(frame),3),bool)
        for k in range(9):
            train=np.flatnonzero(partition!=k)
            q=np.flatnonzero(partition==k) if k<8 else np.arange(n,len(frame))
            assert not np.isin(groups[q],groups[train]).any()
            nested=np.empty((len(train),4))
            for a,b in GroupKFold(3,shuffle=True,random_state=821).split(train,groups=groups[train]):
                assert not np.isin(groups[train[a]],groups[train[b]]).any()
                nested[b]=fit_scores(x,cell,groups,y,train[a],train[b])
            held=fit_scores(x,cell,groups,y,train,q)
            scores=np.vstack([nested,held])
            direct=scores[:,1]>=0
            strata,pattern=features(direct,*scores.T)
            t=np.arange(len(train))
            ask=len(t)+np.arange(len(q))
            broad,specific=estimate(strata,pattern,t,ask,direct[t]!=y[train])
            eligible=(pattern[ask]>0)
            trained=(specific>.5)&eligible
            corrected=direct[ask]^trained
            flags[q,0]=trained & (corrected!=parent[q])
            flags[q,1]=flags[q,0] & (broad<=.5)
            flags[q,2]=flags[q,1] & (pattern[ask]==3)
        report[scheme]={}
        for j,name in enumerate(['correction','additional','both_sources']):
            action=flags[:,j]
            saved[scheme+'_'+name]=action
            report[scheme][name]=dict(scopes={s:count(action[:n],m,wrong) for s,m in masks.items()},
                accuracy=float(np.mean((parent[:n]^action[:n])[known]==y[known])),test=int(action[n:].sum()))
        print(scheme,report[scheme],flush=True)
    for name in report['random']:
        np.testing.assert_array_equal(saved['random_'+name][n:],saved['ordered_'+name][n:])
    dest=ART/'nov28_nested_exception_frequency'
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=f,**saved)
    (ROOT/'reports/nov23_matched_learning_curve/nested_exception_frequency.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):run()
