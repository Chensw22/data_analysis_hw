"""Learn P(outcome, teacher vote pattern | manifest), then condition on held teacher opinions."""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from threadpoolctl import threadpool_limits

from distinct_challenger_error_reach_n26 import population
from exaone_validation_data import CATEGORICAL,compact_features
from full_roster_booking_validation_n23 import ROOT,data
from tabnet_attention_transfer_n26 import RESULT,collect
from target_mix_expert_routing_n26 import ticker

CACHE=ROOT/'artifacts/nov26_booking_block_teacher/nested'
OUT=ROOT/'artifacts/nov26_conditional_teacher_outcomes'
REPORT=RESULT.parent/'conditional_teacher_outcomes.json'


def train(x,label,multiclass=False):
    return CatBoostClassifier(iterations=260,depth=5,learning_rate=.065,l2_leaf_reg=6,random_seed=628,
                              thread_count=3,verbose=False,allow_writing_files=False,cat_features=CATEGORICAL,
                              loss_function='MultiClass' if multiclass else 'Logloss').fit(x,label)


def code(p):
    return (p>=.5).astype(int)@np.array([1,2,4])


def given(prob,pattern):
    states=np.arange(8)
    distance=np.array([int(i).bit_count() for i in range(8)])[states[None,:]^pattern[:,None]]
    def calculate(weight):
        mass=(prob*weight[:,:,None]).sum(1)
        return mass[:,1]/mass.sum(1).clip(1.e-12)
    return calculate(distance==0),calculate(.1**distance)


def enriched(x,opinions):
    return x.assign(**{f'teacher{j}':logit(opinions[:,j].clip(.002,.998)) for j in range(3)})


def predict(frame,y,g,f,k):
    t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
    with np.load(CACHE/f'fold{k}.npz') as z:
        for n,x in [('t',t),('q',q),('train_y',y[t])]:
            np.testing.assert_array_equal(z[n],x)
        half,soft,held=z['inner_fold'],z['soft'],z['outer_single']
        for i in range(3):
            assert not np.isin(g[t[half==i]],g[t[half!=i]]).any()
    assert not np.isin(g[t],g[q]).any() and held.shape==(3,len(q),3)
    with np.load(CACHE.parent/f'fold{k}.npz') as z:
        np.testing.assert_allclose(held.mean(0),np.column_stack([z[n] for n in ['extra_trees','hist_boost','additive_logistic']]),atol=1.e-7,rtol=0)
    joint=train(frame.iloc[t],2*code(soft)+y[t],True)
    probs=np.zeros((len(q),16))
    probs[:,joint.classes_.astype(int)]=joint.predict_proba(frame.iloc[q])
    tensor=probs.reshape(len(q),8,2)
    xt=enriched(frame.iloc[t],soft)
    stack=train(xt,y[t])
    residual=train(xt,y[t]!=(soft.mean(1)>=.5))
    raw=train(frame.iloc[t],y[t])
    out=dict(raw=raw.predict_proba(frame.iloc[q])[:,1],joint_marginal=tensor[:,:,1].sum(1))
    parts={}
    for i,views in enumerate(held):
        pattern=code(views)
        majority=views.mean(1)>=.5
        # Break identities within the same majority and CryoSleep group, without labels.
        perm=np.arange(len(q))
        groups=pd.factorize(frame.CryoSleep.iloc[q])[0]*2+majority
        rng=np.random.default_rng(7129+k*3+i)
        for level in np.unique(groups):
            idx=np.flatnonzero(groups==level)
            perm[idx]=rng.permutation(idx)
        exact,smooth=given(tensor,pattern)
        _,placebo=given(tensor,pattern[perm])
        xq=enriched(frame.iloc[q],views)
        error=residual.predict_proba(xq)[:,1]
        for name,p in dict(exact=exact,smooth=smooth,shuffled=placebo,stack=stack.predict_proba(xq)[:,1],
                           residual=np.where(majority,1-error,error)).items():
            parts.setdefault(name,[]).append(p)
    out.update({key:np.mean(value,axis=0) for key,value in parts.items()})
    assert all(np.isfinite(v).all() and ((v>=0)&(v<=1)).all() for v in out.values())
    return q,out


def run():
    raw,y,g,fold=data()
    frame=compact_features(raw).assign(order_size=raw.groupby(g).PassengerId.transform('size'))
    p={}
    OUT.mkdir(exist_ok=True)
    for k in range(8):
        q,cols=predict(frame,y,g,fold,k)
        for name,values in cols.items():
            p.setdefault(name,np.full(len(y),np.nan))[q]=values
        print('fold',k,flush=True)
    np.savez_compressed(OUT/'scores.npz',ids=raw.PassengerId.to_numpy(str),fold=fold,y=y,**p)
    raw,truth,_,_,parent,same,reach,_,_,wrong,_,_=population()
    np.testing.assert_array_equal(truth,y)
    hard={n:s>=.5 for n,s in p.items()}
    flips={n:s!=parent for n,s in hard.items()}
    flips.update(smooth_extra=flips['smooth']&~flips['joint_marginal'],
                 only_true_opinions=flips['smooth']&~flips['shuffled'],
                 both_direct_heads=flips['stack']&flips['residual'])
    report={'accuracy':{n:float(np.mean(s==y)) for n,s in hard.items()}}
    for name,scope in dict(ten=same,prior_dissent=same&(reach>0),none=same&(reach==0)).items():
        report[name]={n:ticker(scope,wrong,a) for n,a in flips.items()}
    by_tag={}
    for tag,items in collect()[0].items():
        ids,common,now=(np.concatenate([a[j] for a in items]) for j in range(3))
        scope=common&~np.concatenate([a[3]['actual_and_scores'] for a in items])
        target=y[ids]!=now
        s=hard['smooth'][ids]!=now
        h=hard['stack'][ids]!=now
        actions={'conditional_only':s&(hard['joint_marginal'][ids]==now),
                 'conditional_and_stack':s&h,'additional_to_shuffled':s&(hard['shuffled'][ids]==now)}
        by_tag[tag]={}
        for key,a in actions.items():
            r=ticker(scope,target,a)
            r['net_by_fold']=[int((np.where(target,1,-1)*scope*a*(fold[ids]==k)).sum()) for k in range(8)]
            by_tag[tag][key]=r
    report['beyond_queued_tabnet']=by_tag
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['accuracy'],report['ten'],flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
