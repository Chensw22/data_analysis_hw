"""Learn pairwise error rankings within physically similar passenger groups and calibrate separately."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from directional_foundation_transfer_n27 import calibrate
from dissent_likelihood_transfer_n27 import splits
from exaone_validation_data import compact_features, CATEGORICAL
from unanimous_error_reach_n27 import inputs
from repeated_archive_misses_n26 import ART, cohort, count
from full_roster_booking_validation_n23 import ROOT
from price_implied_deck_n26 import FEES

OUT=ART/'nov27_rank_passenger_mistakes'
REPORT=ROOT/'reports/nov23_matched_learning_curve/rank_passenger_mistakes.json'


def run():
    raw,frame,y,g,f,first,random,ordered,wide,alt=inputs()
    n=len(y)
    _,blind,known,fresh=cohort(raw,y,f)
    err=first[:n]!=y
    odds=lambda v:logit(np.clip(v,.003,.997))
    cats=pd.get_dummies(compact_features(frame),columns=CATEGORICAL,dtype=float).to_numpy()
    disagree=np.where(first[:,None],1-wide,wide)
    alt_scores=np.column_stack(list(alt.values()))
    x=np.column_stack([cats,odds(disagree),odds(np.where(first[:,None],1-alt_scores,alt_scores))])
    paid=frame[FEES].gt(0).any(axis=1).to_numpy()
    sleep=frame.CryoSleep.eq(True).to_numpy()
    cohort_id=first*4+sleep*2+paid
    all_test=np.arange(n,len(frame))
    report={}; model_actions={}
    for scheme,protected in [('random',random),('contiguous',ordered)]:
        group_split=splits(g,f,scheme)
        flags={k:np.zeros(len(frame),bool) for k in ('rank','binary','rank_unique','both')}
        test_votes={k:[] for k in flags}
        for k in range(8):
            for j in range(4):
                q=np.flatnonzero((f==k)&(group_split==j))
                m=(j+1)%4
                bank=np.flatnonzero(known&(f==k)&(group_split!=j)&(group_split!=m))
                cal=np.flatnonzero(known&(f==k)&(group_split==m))
                assert not np.isin(g[bank],g[cal]).any() and not np.isin(g[q],g[np.r_[bank,cal]]).any()
                order=np.argsort(cohort_id[bank],kind='stable')
                bank=bank[order]
                _,length=np.unique(cohort_id[bank],return_counts=True)
                args=dict(n_estimators=65,num_leaves=6,learning_rate=.045,min_child_samples=22,verbosity=-1,
                          reg_lambda=8.,n_jobs=1,random_state=114)
                models={'rank':lgb.LGBMRanker(**args,objective='lambdarank',lambdarank_truncation_level=30),
                        'binary':lgb.LGBMClassifier(**args)}
                query=np.r_[q,all_test]
                actions={}
                for key,model in models.items():
                    kw={'group':length} if key=='rank' else {}
                    model.fit(x[bank],err[bank].astype(int),**kw)
                    risks=np.empty(len(frame))
                    # Monotone map retains all rank candidates so the calibration set can choose any rank threshold.
                    allq=np.r_[cal,query]
                    risks[allq]=.50001+.49998*expit(model.booster_.predict(x[allq],raw_score=True))
                    cut=calibrate(risks,first,y,cal,1.)
                    actions[key]=(risks[query]>=cut[first[query].astype(int)])&~protected[query]
                actions['rank_unique']=actions['rank']&~actions['binary']
                actions['both']=actions['rank']&actions['binary']
                for key,a in actions.items():
                    flags[key][q]=a[:len(q)]
                    test_votes[key].append(a[len(q):])
        for key,a in flags.items():
            a[n:]=np.mean(test_votes[key],axis=0)>.5
            delta=a[:n]*np.where(err,1,-1)
            report.setdefault(key,{})[scheme]=dict(all=count(a[:n],known,err),fresh=count(a[:n],fresh&known,err),
                former_unanimous=count(a[:n],blind,err),agreed_seven=count(a[:n],known&(disagree[:n]<.5).all(1),err),
                folds=[int(delta[known&(f==j)].sum()) for j in range(8)],test=int(a[n:].sum()))
            model_actions[scheme+'_'+key]=a
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=f,**model_actions)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
