"""Match wrong and right passengers at comparable reference confidence before learning which dissent to trust."""
import json
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from archive_source_harm_veto_n28 import inputs
from dissent_likelihood_transfer_n27 import splits
from repeated_archive_misses_n26 import ART,ROOT,cohort,count


def estimate(x,level,risk,groups,wrong,t,q):
    winners,losers=[],[]
    for cell in np.unique(level[t]):
        pos=t[(level[t]==cell)&wrong[t]]
        neg=t[(level[t]==cell)&~wrong[t]]
        for i in pos:
            others=neg[groups[neg]!=groups[i]]
            if not len(others):continue
            distances=abs(risk[others]-risk[i])
            selected=others[np.argsort(distances,kind='stable')[:3]]
            winners.extend([i]*len(selected))
            losers.extend(selected)
    diff=x[winners]-x[losers]
    target=np.r_[np.ones(len(diff)),np.zeros(len(diff))]
    paired=LogisticRegression(C=.03,fit_intercept=False,max_iter=300).fit(np.vstack([diff,-diff]),target)
    ordinary=LogisticRegression(C=.03,max_iter=300).fit(x[t],wrong[t])
    out=[]
    for model in [ordinary,paired]:
        zt=model.decision_function(x[t])
        zq=model.decision_function(x[q])
        for cell in np.unique(level[q]):
            ix=level[t]==cell
            fallback=(wrong[t].sum()+1)/(len(t)+2)
            rate=(wrong[t][ix].sum()+6*fallback)/(ix.sum()+6)
            bank=zt[ix] if ix.any() else zt
            intercept=brentq(lambda a:np.mean(expit(bank+a))-rate,-70,70)
            zq[level[q]==cell]+=intercept
        out.append(zq)
    return np.column_stack(out)


def run():
    frame,y,g,f,ids,_,p,ordinary,versions=inputs()
    n=len(y)
    groups=frame.PassengerId.str[:4].to_numpy()
    known,fresh=cohort(frame.iloc[:n],y,f)[2:]
    parent=versions['random'][:,0]
    wrong=parent[:n]!=y
    common=((ordinary>=.5)==parent[:,None]).all(1)
    risk=np.median(np.where(parent[:,None],1-ordinary,ordinary),axis=1)
    level=parent.astype(int)*6+(~common)*3+np.digitize(risk,[.18,.36])
    signed=np.where(parent[:,None],1-p,p)
    x=logit(np.clip(signed,.005,.995))
    x-=np.median(x,axis=1,keepdims=True)
    report,stored={},{}
    for tag,protocol in [('random','random'),('ordered','contiguous')]:
        part=splits(g,f,protocol)
        four=(versions[tag]==parent[:,None]).all(1)
        eligible=four&(signed>.5).any(1)
        masks=dict(all=known,fresh=known&fresh,common=known&common[:n]&four[:n])
        scores=np.empty((len(frame),2))
        votes=[]
        for k in range(8):
            for j in [0,1,2,3,None]:
                t=np.flatnonzero(known&eligible[:n]&(f==k)&((part!=j) if j is not None else True))
                q=np.arange(n,len(frame)) if j is None else np.flatnonzero((f==k)&(part==j))
                assert not np.isin(groups[t],groups[q]).any()
                z=estimate(x,level,risk,groups,wrong,t,q)
                if j is None:votes.append(z)
                else:scores[q]=z
        scores[n:]=np.median(votes,axis=0)
        ordinary_flip,paired_flip=((scores>0)&eligible[:,None]).T
        policies=dict(ordinary=ordinary_flip,matched=paired_flip,unique=paired_flip&~ordinary_flip,
                      common_only=paired_flip&common)
        report[tag]={}
        for key,flag in policies.items():
            stored[tag+'_'+key]=flag
            report[tag][key]=dict(scopes={s:count(flag[:n],m,wrong) for s,m in masks.items()},
              accuracy=float(np.mean((parent[:n]^flag[:n])[known]==y[known])),test=int(flag[n:].sum()))
        print(tag,report[tag],flush=True)
    out=ART/'nov28_matched_dissent_pairs'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'aligned.npz',passenger=ids,y=y,fold=f,**stored)
    (ROOT/'reports/nov23_matched_learning_curve/matched_dissent_pairs.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):run()
