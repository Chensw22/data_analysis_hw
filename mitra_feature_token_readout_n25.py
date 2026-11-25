"""Fit residual readouts from feature tokens of the fixed15 model on separate held-out orders."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_conditional_pool_n23 import ROOT,data
from full_roster_disagreement_calibration_n24 import stats
from full_roster_six_failure_archive_n24 import prediction_sets
from historical_nine_release_n25 import OUT as HISTORY

STORE=ROOT/'artifacts/nov25_mitra_fp32_head'
METHODS=('native_margin','feature_tokens','label_hidden','both')


def estimate(train,ask,target,offset):
    mean,scale=train.mean(0),train.std(0).clip(.03)
    a,b=[np.column_stack([np.ones(len(x)),((x-mean)/scale).clip(-5,5)]) for x in (train,ask)]
    def loss(w):
        v=offset+a@w
        return (np.logaddexp(0,v)-target*v).mean()+.025*(w@w), a.T@(expit(v)-target)/len(v)+.05*w
    opt=minimize(loss,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B')
    if not opt.success:
        raise RuntimeError(opt.message)
    return b@opt.x


def predict(z,query,target,t,q):
    # Optional PCA uses only the training passengers of the supervised readout.
    projection=PCA(n_components=16,svd_solver='randomized',random_state=316).fit(z['label_hidden'][t].astype(float))
    def designs(z,ids):
        margin=z['fp32_margin'][ids,None]
        features=z['feature_projection'][ids]
        hidden=projection.transform(z['label_hidden'][ids].astype(float))
        return [margin,np.column_stack([margin,features]),np.column_stack([margin,hidden]),
                np.column_stack([margin,features,hidden])]
    train,ask=designs(z,t),designs(query,q)
    return np.column_stack([estimate(a,b,target[t],z['fp32_margin'][t]) for a,b in zip(train,ask,strict=True)])


def run():
    raw,y,g,f=data()
    with np.load(HISTORY/'historical.npz') as z:
        rows,fold,inc,v131=z['rows'],z['fold'],z['nine'],z['v131']
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[rows])
    _,panels=prediction_sets(raw,y,g,f,rows,fold)
    truth=y[rows]
    strict=(truth!=inc)&((np.column_stack(panels[1])>=.5)==inc[:,None]).all(1)
    shifts=np.full((len(rows),len(METHODS)),np.nan)
    margin=np.full(len(rows),np.nan)
    native=np.full(len(rows),np.nan)
    packs=[]
    with np.load(STORE/'test.npz') as z:
        test={k:z[k] for k in ('fp32_margin','feature_projection','label_hidden','passenger','feature_names','native')}
    for level in np.unique(fold):
        ids=np.flatnonzero(fold==level)
        with np.load(STORE/f'fold{level}.npz') as z:
            np.testing.assert_array_equal(z['row'],rows[ids])
            np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[rows[ids]])
            np.testing.assert_array_equal(z['feature_names'],test['feature_names'])
            cache={k:z[k] for k in ('fp32_margin','feature_projection','label_hidden','native')}
        margin[ids]=cache['fp32_margin']; native[ids]=cache['native']
        group=g[rows][ids]
        free=((native[ids]>=.5)==v131[ids])&(v131[ids]==inc[ids])
        for t,q in GroupKFold(4,shuffle=True,random_state=128).split(ids,groups=group):
            assert not np.isin(group[t],group[q]).any()
            shifts[ids[q]]=predict(cache,cache,truth[ids],t[free[t]],q)
        packs.append(predict(cache,test,truth[ids],np.flatnonzero(free),np.arange(len(test['native']))))
    assert np.isfinite(shifts).all() and np.isfinite(margin).all()
    from export_budget_quarter_n23 import source
    import pandas as pd
    df=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    np.testing.assert_array_equal(test['passenger'],df.PassengerId)
    _,prev,_=source('proposal_v131_party_consensus',df)
    _,best,_=source('proposal_n24_selected_dissent',df)
    free=((native>=.5)==v131)&(v131==inc)
    test_free=((test['native']>=.5)==prev)&(prev==best)
    results={}
    for j,name in enumerate(METHODS):
        out=np.where(free,margin+shifts[:,j]>=0,inc)
        held=stats(out,inc,truth,g[rows],fold)
        held['common_fixed']=int((strict&(out==truth)).sum())
        decisions=np.where(test_free,np.mean(packs,axis=0)[:,j]+test['fp32_margin']>=0,best)
        held['test_changes']=int((decisions!=best).sum())
        held['test_head_disagreement']=int(((np.ptp(np.stack([test['fp32_margin']+a[:,j]>=0 for a in packs]).astype(int),axis=0)>0)&test_free).sum())
        results[name]=held
        print(name,held['corrected'],held['harmed'],held['fold_net'],held['common_fixed'],held['test_changes'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/mitra_feature_token_readout.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
