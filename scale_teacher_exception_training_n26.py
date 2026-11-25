"""Test how many independently predicted training passengers a teacher-error specialist needs."""
import json
import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from distinct_challenger_error_reach_n26 import population
from exaone_validation_data import CATEGORICAL,compact_features
from full_roster_booking_validation_n23 import ROOT
from price_implied_deck_n26 import FEES
from tabnet_attention_transfer_n26 import RESULT,collect
from target_mix_expert_routing_n26 import ticker

CACHE=ROOT/'artifacts/nov26_booking_block_teacher/nested'
REPORT=RESULT.parent/'scale_teacher_exception_training.json'


def manifest_features(raw,group):
    frame=compact_features(raw)
    frame[FEES]=np.log1p(frame[FEES])
    # Explicit levels keep the design identical across independently generated rosters.
    categories={'HomePlanet':['Earth','Europa','Mars'],'Destination':['TRAPPIST-1e','55 Cancri e','PSO J318.5-22'],
                'Deck':list('ABCDEFGT'),'Side':['P','S'],'CryoSleep':['True','False'],'VIP':['True','False']}
    encoded=[frame[n].eq(v).to_numpy() for n in CATEGORICAL for v in [*categories[n],'__MISSING__']]
    num=frame.drop(columns=CATEGORICAL)
    return np.column_stack([num.fillna(-1),num.isna(),group,*encoded])


def features(raw,group,opinions):
    ref=opinions.mean(1)>=.5
    signed=logit(opinions.clip(.002,.998))*np.where(ref,-1,1)[:,None]
    return np.column_stack([ref,signed,signed.std(1),manifest_features(raw,group)]),ref


def run():
    raw,y,g,fold,parent,same,reach,_,_,wrong,_,_=population()
    size=raw.PassengerId.groupby(g).transform('size').to_numpy()
    joint={}
    active={}
    support={}
    for k in range(8):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        with np.load(CACHE/f'fold{k}.npz') as z:
            for key,e in [('t',t),('q',q),('train_y',y[t])]:
                np.testing.assert_array_equal(z[key],e)
            p,held=z['soft'],z['outer_single']
            for i in range(3):
                assert not np.isin(g[t[z['inner_fold']!=i]],np.r_[g[q],g[t[z['inner_fold']==i]]]).any()
        train,base=features(raw.iloc[t],size[t],p)
        qt=[features(raw.iloc[q],size[q],z) for z in held]
        same_t=((p>=.5)==base[:,None]).all(1)
        unanimous=np.stack([((z>=.5)==prior[:,None]).all(1) for z,(_,prior) in zip(held,qt,strict=True)])
        pool=partition(g[t],'error-specialist-size:',10)
        for fraction in [1,5,10]:
            select=same_t&(pool<fraction)
            support.setdefault(str(fraction),[]).append(int(select.sum()))
            a=train[select]
            scaler=StandardScaler().fit(a)
            h=scaler.transform(a).clip(-5,5)
            label=y[t][select]!=base[select]
            models={'linear':LogisticRegression(C=.12,max_iter=400),
                    'tree':HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=6,min_samples_leaf=25,
                               learning_rate=.055,l2_regularization=8,early_stopping=False,random_state=98)}
            for name,m in models.items():
                key=f'{name}_{fraction}0pct'
                m.fit(h,label)
                predictions=[]
                for x,prior in qt:
                    err=m.predict_proba(scaler.transform(x).clip(-5,5))[:,1]
                    predictions.append(np.where(prior,1-err,err))
                pr=np.stack(predictions)
                # Apply only when every independently fitted inner-teacher panel is internally unanimous
                # and all panels would have retained the same reference prediction.
                aligned=unanimous.all(0)&np.all(np.stack([prior==qt[0][1] for _,prior in qt]),axis=0)
                target=qt[0][1]
                active.setdefault(key,np.zeros(len(y),bool))[q]=aligned&(target==parent[q])
                joint.setdefault(key,np.full(len(y),np.nan))[q]=pr.mean(0)
    report=dict(training_rows_by_fold=support,eightfold={})
    for name,score in joint.items():
        eligible=same&active[name]
        assert np.isfinite(score).all()
        act=(score>=.5)!=parent
        report['eightfold'][name]={key:ticker(mask,wrong,act) for key,mask in
                                    [('all_aligned',eligible),('with_any_challenger',eligible&(reach>0)),('no_challenger',eligible&(reach==0))]}
    for tag,items in collect()[0].items():
        ids,common,now=(np.concatenate([r[j] for r in items]) for j in range(3))
        keep=common&~np.concatenate([r[3]['actual_and_scores'] for r in items])
        report[tag]={}
        for name,score in joint.items():
            # The active mask is signed relative to eightfold parent, checked again for the replay parent.
            eligible=keep&active[name][ids]&(now==parent[ids])
            act=(score[ids]>=.5)!=now
            report[tag][name]=ticker(eligible,y[ids]!=now,act)
        print(tag,{n:(r['fixed'],r['harmed']) for n,r in report[tag].items()},flush=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
