"""Learn from equally plausible helpful and harmful expert reversals on independent bookings."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import expit,logit
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from broad_dispute_leaf_evidence_n23 import ART, ROOT, SEEDS, cohorts
from class_analogue_experts_n23 import design
from exaone_validation_data import compact_features
from relational_error_corrector import write

OUT=ROOT/"reports/nov23_broad_dispute_leaves"


def pair_rows(ids,helpful,score,groups,match):
    cases=ids[helpful[ids]]
    controls=ids[~helpful[ids]]
    distance=cdist(score[cases],score[controls],metric="sqeuclidean")
    if not match:
        distance=np.random.default_rng(247).random(distance.shape)
    distance[groups[cases,None]==groups[controls]]=1e5
    i,j=linear_sum_assignment(distance)
    assert not (groups[cases[i]]==groups[controls[j]]).any()
    return cases[i],controls[j]


def rank_score(x,offset,benefit,train,test,cases,controls):
    # A difference cancels the pair intercept, so similar confidence levels
    # cannot earn much credit on the auxiliary discrimination task.
    xx=x[train]
    dx=x[cases]-x[controls]
    prior=offset[train]
    gap=offset[cases]-offset[controls]
    weight=len(train)/len(cases)
    ridge=16.
    def objective(beta):
        u=prior+xx@beta
        margin=gap+dx@beta
        value=np.logaddexp(0,u).sum()-benefit[train]@u+weight*np.logaddexp(0,-margin).sum()
        gradient=xx.T@(expit(u)-benefit[train])-weight*(dx.T@expit(-margin))
        return value+.5*ridge*(beta@beta),gradient+ridge*beta
    fitted=minimize(objective,np.zeros(x.shape[1]),method="L-BFGS-B",jac=True)
    assert fitted.success,fitted.message
    return expit(offset[test]+x[test]@fitted.x)


def run():
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Name","Transported"))
    all_hit={}
    refs=[]
    result={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_broad_shared_passenger_modes/{seed}.npz") as z:
            row,y,fold,base,shape,own,fake,pca=[z[k] for k in ("row","y","fold","base","real_only","own","fake_only","pca_only")]
        frame=compact_features(raw.iloc[row])
        group=raw.PassengerId.str.split("_").str[0]
        size=group.map(group.value_counts()).to_numpy()[row]
        group=group.to_numpy()[row]
        ref=base>=.5
        dispute=ref!=(shape>=.5)
        helpful=ref!=y
        refs.append(~helpful)
        # Positive means stronger evidence for the proposed new direction.
        sign=np.where(ref,-1,1)
        scores=logit(np.column_stack([base,shape,own]).clip(.002,.998))*sign[:,None]
        result[seed]={}
        wins={}
        rates={}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            x,phi=design(frame,t)
            predictors=np.column_stack([phi,np.log(size),
                       sign[:,None]*logit(np.column_stack([fake,pca]).clip(.002,.998))])
            for a,b in GroupKFold(3,shuffle=True,random_state=756).split(q,groups=group[q]):
                assert not np.intersect1d(group[q[a]],group[q[b]]).size
                for direction in (False,True):
                    eligible=dispute&(ref==direction)
                    train=q[a][eligible[q[a]]]
                    test=q[b][eligible[q[b]]]
                    if not len(test):
                        continue
                    scaler=scores[train].std(0).clip(.3)
                    control=(scores-scores[train].mean(0))/scaler
                    model=LogisticRegression(C=.3).fit(control[train],helpful[train])
                    offset=model.decision_function(control)
                    rank_input=(predictors-predictors[train].mean(0))/predictors[train].std(0).clip(.3)
                    rank_input=np.clip(rank_input,-4,4)
                    prob={"scores":expit(offset[test])}
                    for name,match in (("matched",True),("random_pairs",False)):
                        ca,co=pair_rows(train,helpful,control,group,match)
                        prob[name]=rank_score(rank_input,offset,helpful,train,test,ca,co)
                    # This is an after-prediction diagnostic only. Test labels
                    # identify the one helpful reversal in each test comparison.
                    ca,co=pair_rows(test,helpful,control,group,True)
                    for name,risk in prob.items():
                        all_risk=rates.setdefault(name,np.full(len(y),np.nan))
                        all_risk[test]=risk
                        wins.setdefault(name,[]).extend((all_risk[ca]>all_risk[co])+.5*(all_risk[ca]==all_risk[co]))
        for name,risk in rates.items():
            assert np.isfinite(risk[dispute]).all() and np.isnan(risk[~dispute]).all()
            flip=dispute&(risk>=.5)
            answer=np.where(flip,~ref,ref)
            all_hit.setdefault(name,[]).append(answer==y)
            # Optimistic cut, assessed on this evaluation's labels. It provides
            # an upper bound only for thresholding this fixed risk ranking.
            ids=np.flatnonzero(dispute)
            order=ids[np.argsort(-risk[ids],kind="stable")]
            ends=np.r_[np.flatnonzero(np.diff(risk[order])!=0),len(order)-1]
            net=np.r_[0,np.cumsum(2*helpful[order].astype(int)-1)[ends]]
            result[seed][name]=dict(auc=float(roc_auc_score(helpful[dispute],risk[dispute])),
                matched_test_pairs=len(wins[name]),fraction_prefer_helpful=float(np.mean(wins[name])),
                oracle_best_net=int(net.max()),oracle_best_accuracy=float(((~helpful).sum()+net.max())/len(y)))
        print(seed,result[seed],flush=True)
    masks=cohorts(row,y)
    ref=np.stack(refs)
    scores={}
    for k,h in all_hit.items():
        hit=np.stack(h)
        scores[k]=dict(accuracy=hit.mean(1).tolist(),fixed=(hit&~ref).sum(1).tolist(),
                       harmed=(~hit&ref).sum(1).tolist(),stable={n:int((m&hit.all(0)).sum()) for n,m in masks.items()})
    write(OUT/"matched_ranking.json",dict(scores=scores,ranking=result))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
