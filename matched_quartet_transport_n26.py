"""Train trees to identify the transported people among matched sets of four separate bookings."""
import json
from itertools import combinations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import expit, logit, softmax, logsumexp
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT, data
from full_roster_two_view_reliability_n24 import design
from repeated_archive_misses_n26 import ART, cohort, count

OUT=ART/'nov26_matched_quartet_transport'
REPORT=ROOT/'reports/nov23_matched_learning_curve/matched_quartet_transport.json'
STATES={k:np.array([[int(i in pick) for i in range(4)] for pick in combinations(range(4),k)],float)
        for k in (1,2,3)}


def sets(x,groups,seed,local):
    rng=np.random.default_rng(seed)
    perm=rng.permutation(len(x))
    candidates=cKDTree(x).query(x,k=64)[1] if local else None
    available=np.ones(len(x),bool)
    result=[]
    for i in perm:
        if not available[i]:
            continue
        if available.sum()<4:break
        donors=candidates[i] if local else rng.integers(len(x),size=32)
        source=available.copy()
        source[i]=False
        others=[]
        for pool in (donors,None):
            if pool is None:
                pool=rng.permutation(np.flatnonzero(source))
            for j in pool:
                if source[j] and groups[j]!=groups[i] and all(groups[j]!=groups[b] for b in others):
                    others.append(j)
                    source[j]=False
                    if len(others)==3:break
            if len(others)==3:break
        if len(others)<3:continue
        ids=[i,*others]
        available[ids]=False
        result.append(ids)
    result=np.asarray(result)
    assert len(np.unique(result))==result.size and result.size>=len(x)-10
    return result


def assemble(bags,y):
    idx=np.vstack(bags)
    totals=y[idx].sum(1)
    return {k:idx[totals==k] for k in STATES}


def gradients(z,y,parts,replicas):
    point=expit(z)
    grad=point-y
    hess=point*(1-point)
    for k,ids in parts.items():
        state=STATES[k]
        inclusion=softmax(z[ids]@state.T,axis=1)@state
        grad+=np.bincount(ids.ravel(),weights=((inclusion-y[ids])/replicas).ravel(),minlength=len(y))
        hess+=np.bincount(ids.ravel(),weights=(inclusion*(1-inclusion)/replicas).ravel(),minlength=len(y))
    return grad,hess.clip(.001)


def check_gradient():
    z=np.array([-.1,1.2,.3,-.6,-2,.1,.3,.9])
    y=np.array([0,1,1,0,0,0,0,1],float)
    parts=assemble([np.arange(8).reshape(2,4)],y)
    def loss(x):
        r=np.logaddexp(0,x).sum()-y@x
        for k,idx in parts.items():
            r+=(logsumexp(x[idx]@STATES[k].T,axis=1)-(x[idx]*y[idx]).sum(1)).sum()
        return r
    g,h=gradients(z,y,parts,1)
    eps=.0002
    for i in range(len(z)):
        e=eps*np.eye(len(z))[i]
        np.testing.assert_allclose(g[i],(loss(z+e)-loss(z-e))/(2*eps),atol=1e-6)
        np.testing.assert_allclose(h[i],(loss(z+e)-2*loss(z)+loss(z-e))/eps**2,atol=1e-6)


def boost(mat,truth,t,q,parts,replicas):
    target=lgb.Dataset(mat[t],label=truth,free_raw_data=False)
    def objective(z,dataset):
        return gradients(z,truth,parts,replicas)
    settings=dict(verbosity=-1,learning_rate=.04,num_leaves=18,min_data_in_leaf=30,
                  lambda_l2=5,num_threads=2,force_col_wise=True,feature_pre_filter=False,
                  seed=622,objective=objective)
    model=lgb.train(settings,target,num_boost_round=240)
    return expit(model.predict(mat[q],raw_score=True,num_threads=2))


def fit(mat,geometry,y,group,t,q,k):
    focused=geometry[t]
    geom=(focused-focused.mean(0))/focused.std(0).clip(.3)
    truth=y[t].astype(float)
    scores={}
    audit={}
    used={}
    for mode in ('ordinary','random','matched'):
        partitions=[]
        if mode!='ordinary':
            for j in range(2):
                partitions.append(sets(geom,group[t],2380+k*2+j,mode=='matched'))
        parts=assemble(partitions,truth) if partitions else {}
        used[mode]=parts
        scores[mode]=boost(mat,truth,t,q,parts,max(1,len(partitions)))
        audit[mode]=[sum(map(len,partitions)),sum(map(len,parts.values()))]
    rng=np.random.default_rng(33062+k)
    trimmed={}
    for number,rows in used['random'].items():
        wanted=len(used['matched'][number])
        assert len(rows)>=wanted
        trimmed[number]=rows[rng.choice(len(rows),wanted,replace=False)]
    scores['random_same_counts']=boost(mat,truth,t,q,trimmed,2)
    audit['random_same_counts']=[audit['random'][0],sum(map(len,trimmed.values()))]
    return scores,audit

def run():
    check_gradient()
    raw,y,g,f=data()
    base,blind,known,fresh=cohort(raw,y,f)
    frame=pd.concat([raw,pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)],ignore_index=True)
    groups=frame.PassengerId.str.split('_').str[0].to_numpy()
    reg,bills=design(frame,groups)
    mat=np.column_stack([reg,bills])
    med=np.nanmedian(mat,axis=0)
    geometry=np.where(np.isnan(mat),np.nan_to_num(med),mat)
    n=len(y)
    scores={k:np.full(len(mat),np.nan) for k in ['ordinary','random','matched','random_same_counts']}
    audits=[]
    OUT.mkdir(exist_ok=True)
    for k in range(9):
        t=np.flatnonzero(f!=k) if k<8 else np.arange(n)
        q=np.flatnonzero(f==k) if k<8 else np.arange(n,len(mat))
        assert not np.isin(groups[t],groups[q]).any()
        path=OUT/f'controlled_fold{k}.npz'
        if not path.exists():
            result,audit=fit(mat,geometry,y,groups,t,q,k)
            tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,t=t,q=q,labels=y[t],ids=frame.PassengerId.iloc[q].to_numpy(str),
                                audit=json.dumps(audit),**result)
            tmp.replace(path)
        with np.load(path) as z:
            for key,expected in [('t',t),('q',q),('labels',y[t]),('ids',frame.PassengerId.iloc[q])]:
                np.testing.assert_array_equal(z[key],expected)
            for mode in scores:scores[mode][q]=z[mode]
            audits.append(json.loads(str(z['audit'])))
        print(k,flush=True)
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
        anchor=z['anchor']
        strength=abs(logit(z['scores'][:,-3:].mean(1).clip(.005,.995)))
    np.testing.assert_array_equal(anchor[:n],base)
    report=dict(accuracy={k:float(((v[:n]>=.5)==y).mean()) for k,v in scores.items()},sets=audits,rules={})
    for control in ('ordinary','random','random_same_counts'):
        increment=logit(scores['matched'].clip(.005,.995))-logit(scores[control].clip(.005,.995))
        scores['against_'+control]=expit(np.where(anchor,1.,-1.)*strength+increment)
    edits={k:(v>=.5)!=anchor for k,v in scores.items()}
    edits['matched_only']=edits['matched'] & ~edits['ordinary'] & ~edits['random'] & ~edits['random_same_counts']
    edits['both_increments']=edits['against_ordinary'] & edits['against_random']
    for key,act in edits.items():
        delta=act[:n]*np.where(base!=y,1,-1)
        report['rules'][key]=dict(test_changes=int(act[n:].sum()),by_fold=[int(delta[known&(f==k)].sum()) for k in range(8)],
            scopes={s:count(act[:n],m,base!=y) for s,m in [('all',known),('fresh',fresh&known),('blind',blind)]})
    np.savez_compressed(OUT/'aligned.npz',ids=frame.PassengerId.to_numpy(str),y=y,fold=f,**scores)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['accuracy'],report['rules']['matched_only'])


if __name__=='__main__':
    with threadpool_limits(2):
        run()
