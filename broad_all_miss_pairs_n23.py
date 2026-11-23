"""Pair every repeated miss with a distinct correct booking at similar risk and bill state."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import load
from broad_latent_vote_patterns_n23 import ballots
from broad_partition_evidence_n23 import ART, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize
from paid_conditional_tilt_n23 import REPORT
from released_correction_audit_n23 import development


def assign(cases,controls,risk,group):
    cost=np.square(risk[:,cases,None]-risk[:,controls][:,None,:]).sum(0)
    cost[group[cases,None]==group[controls]]=1.e5
    i,j=linear_sum_assignment(cost)
    assert (group[cases[i]]!=group[controls[j]]).all()
    return cases[i],controls[j]


def observations(base,y,state,g):
    # These labels establish diagnostic matches only. No such matches enter a deployed decision.
    positives=(base>=.5).all(0)
    negatives=(base<.5).all(0)
    panels={}
    for k in range(3):
        for sign,active in [(False,negatives),(True,positives)]:
            case=np.flatnonzero(active&(state==k)&(y!=sign))
            control=np.flatnonzero(active&(state==k)&(y==sign))
            i,j=assign(case,control,base,g)
            assert len(i)==len(case)
            panels[(k,sign)]=(i,j)
    return panels


def learned_scores(x,risk,cell,group,truth,fit,query):
    # Conditional logistic ranking. A matched pair contributes both orders.
    # Pool coefficients across all payment states and both directions; include interactions with direction.
    first,second=[],[]
    for c in np.unique(cell[fit]):
        ids=fit[cell[fit]==c]
        case,control=ids[truth[ids]],ids[~truth[ids]]
        if min(len(case),len(control))<2:
            continue
        i,j=assign(case,control,risk[None,:],group)
        first.extend(i)
        second.extend(j)
    scaler=StandardScaler().fit(x[fit])
    scaled=scaler.transform(x)
    difference=scaled[first]-scaled[second]
    model=LogisticRegression(C=.035,fit_intercept=False,max_iter=300).fit(
        np.r_[difference,-difference],np.repeat([1,0],len(first)))
    return model.decision_function(scaled[query])


def main():
    row,y,frame,g,_=load()
    ids,labels,masks=historical()
    np.testing.assert_array_equal(row,ids)
    np.testing.assert_array_equal(y,labels)
    _,release,_=development()
    state=strata(frame)
    masks={**masks,'paid_transported':masks['repeat']&(state==2)&y,'zero_stayed':masks['repeat']&(state==0)&~y}
    history,refs,plans={},[],{}
    score_bank={}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,base,ref=z['fold'],z['base'],z['ref']
        np.testing.assert_array_equal(ref,release[131] if seed=='v131' else base>=.5)
        p=ballots(seed,row,y,fold,base)
        sign=np.where(ref,-1.,1.)
        risk=np.where(ref,1-base,base)
        ordinary=logit(risk.clip(.002,.998))
        cell=2*state+ref
        error=y!=ref
        vote=logit(p.clip(.002,.998))*sign[:,None]
        core=np.c_[ordinary,vote.mean(1),np.eye(6)[cell]]
        inputs=np.c_[vote,(vote-ordinary[:,None])*sign[:,None]]
        modes=['ordinary','matched']
        scores={k:risk.copy() for k in modes}
        for f in np.unique(fold):
            ix=np.flatnonzero((fold==f)&(ref==(base>=.5)))
            parts=[ix[b] for _,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix])]
            for i,q in enumerate(parts):
                pred={k:[] for k in modes}
                for j in [1,2]:
                    train,cal=parts[(i+j)%3],parts[(i+3-j)%3]
                    for a,b in [(train,cal),(train,q),(cal,q)]:assert not np.intersect1d(g[a],g[b]).size
                    cq=np.r_[cal,q]
                    rank=learned_scores(inputs,risk,cell,g,error,train,cq)
                    for k in modes:
                        x=core[cq] if k=='ordinary' else np.c_[core[cq],rank]
                        pred[k].append(calibrated(x[:len(cal)],x[len(cal):],error,cal))
                for k in modes:scores[k][q]=np.mean(pred[k],axis=0)
        outcomes={}
        for k,pr in scores.items():
            final=ref^((pr>=.5)&(ref==(base>=.5)))
            h=final==y
            outcomes[k]=dict(accuracy=float(h.mean()),fixed=int((h&error).sum()),harmed=int((~h&~error).sum()))
            if seed!='v131':history.setdefault(k,[]).append(h)
        if seed!='v131':
            refs.append(~error)
            score_bank.setdefault('base',[]).append(base)
            for k,v in dict(baseline_error=risk,mean_ballot=vote.mean(1),**scores).items():
                score_bank.setdefault(k,[]).append(v)
        plans[seed]=outcomes
    panels=observations(np.stack(score_bank.pop('base')),y,state,g)
    assert sum(len(i) for i,j in panels.values())==1070
    assert np.unique(np.concatenate([j for i,j in panels.values()])).size==1070
    diagnostics={}
    for cell,(cases,controls) in panels.items():
        measures={}
        for k,vals in score_bank.items():
            s=np.stack(vals)
            gap=s[:,cases]-s[:,controls]
            wins=(gap>0)+.5*(gap==0)
            measures[k]=dict(wins=wins.sum(1).tolist(),mean=float(wins.mean()))
        diagnostics[str(cell)]=dict(rows=len(cases),methods=measures)
    write(REPORT/'all_pairs.json',dict(diagnostic=diagnostics,plans=plans,grouped={
        k:summarize(np.stack(v),np.stack(refs),masks) for k,v in history.items()}))
    print(plans,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):
        main()
