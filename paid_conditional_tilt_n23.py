"""Estimate class odds from paid cases and controls at matched incumbent risk levels."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize
from class_analogue_experts_n23 import design
from released_correction_audit_n23 import development

REPORT = ROOT/'reports/nov23_paid_conditional_tilt'
OUT = ART/'nov23_paid_conditional_tilt'


def features(frame, train):
    x, phi = design(frame,train)
    x = np.c_[x, np.log(frame.BookingSize)]
    scale = StandardScaler().fit(x[train])
    z = np.clip(scale.transform(x), -4, 4)
    rng = np.random.default_rng(537)
    freq = rng.normal(size=(z.shape[1],96)) / np.sqrt(z.shape[1])
    nonlinear = np.sqrt(2) * np.cos(z @ freq + rng.uniform(0,2*np.pi,96))
    return np.c_[phi,np.log(frame.BookingSize),nonlinear]


def tilted_odds(x,p,y,t,q,sham=False):
    bins=np.digitize(p,np.unique(np.quantile(p[t],np.arange(1,9)/9)))
    truth=y[t].copy()
    rng=np.random.default_rng(631)
    for k in np.unique(bins[t]):
        cell=np.flatnonzero(bins[t]==k)
        if sham:
            truth[cell]=rng.permutation(truth[cell])
    # Conditional exponential tilt of the control distribution in each risk stratum.
    # Shrink its odds adjustment toward unity. Never use query labels to estimate normalization.
    fit=StandardScaler().fit(x[t])
    values=np.clip(fit.transform(x),-5,5)
    blocks=[]
    for k in np.unique(bins[t]):
        select=bins[t]==k
        cases=t[select & truth]
        controls=t[select & ~truth]
        if len(cases) and len(controls):
            blocks.append((k,len(cases),values[cases].sum(0),values[controls]))
    def objective(w):
        value=20.*w@w
        grad=40.*w
        for _,count,case_sum,ctrl in blocks:
            a=ctrl@w
            norm=logsumexp(a)
            value+=count*(norm-np.log(len(a)))-case_sum@w
            grad=grad+count*(np.exp(a-norm)@ctrl)-case_sum
        return value,grad
    res=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',
                 options=dict(maxiter=300,ftol=1.e-10))
    if not res.success:
        raise RuntimeError(res.message)
    score=np.zeros(len(q))
    for k,_,_,ctrl in blocks:
        mask=bins[q]==k
        score[mask]=values[q[mask]]@res.x-logsumexp(ctrl@res.x)+np.log(len(ctrl))
    return score


def rank_accuracy(p,baseline,y,select):
    # Compare pairs of opposite outcomes whose incumbent probabilities differ by at most one point.
    pos,neg=np.flatnonzero(select&y),np.flatnonzero(select&~y)
    same=np.abs(baseline[pos,None]-baseline[neg])<=.01
    delta=p[pos,None]-p[neg]
    return dict(pairs=int(same.sum()), concordance=float(((delta[same]>0)+.5*(delta[same]==0)).mean()))


def run():
    row,y,frame,g,_=load()
    ids,target,masks=historical()
    np.testing.assert_array_equal(row,ids)
    np.testing.assert_array_equal(y,target)
    _, releases,_=development()
    paid=strata(frame)==2
    hard=masks['repeat']&paid&y
    assert hard.sum()==428
    masks={**masks,'paid_transported':hard,'never_paid':masks['never']&paid&y}
    OUT.mkdir(exist_ok=True)
    results,hist,refs={},{},[]
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,base,ref=z['fold'],z['base'],z['ref']
        np.testing.assert_array_equal(ref,releases[131] if seed=='v131' else base>=.5)
        eligible=paid&~ref&(base<.5)
        probs={k:base.copy() for k in ['base','calibrated','conditional','shuffled']}
        for f in np.unique(fold):
            ix=np.flatnonzero((fold==f)&eligible)
            for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
                t,q=ix[a],ix[b]
                assert not np.intersect1d(g[t],g[q]).size
                x=features(frame,t)
                z=logit(base.clip(.002,.998))[:,None]
                prior=LogisticRegression(C=2.,max_iter=200).fit(z[t], y[t]).decision_function(z[q])
                probs['calibrated'][q]=expit(prior)
                for method in ['conditional','shuffled']:
                    adjustment=tilted_odds(x,base,y,t,q,sham=method=='shuffled')
                    probs[method][q]=expit(prior+adjustment)
        evaluation={}
        anchor=ref==y
        if seed!='v131':
            refs.append(anchor)
        for method,p in probs.items():
            vote=ref.copy()
            vote[eligible]=p[eligible]>=.5
            hit=vote==y
            matched_fix=0
            for f in np.unique(fold):
                ix=np.flatnonzero((fold==f)&eligible)
                k=int(vote[ix].sum())
                first=ix[np.argsort(-base[ix],kind='stable')[:k]]
                matched_fix+=int(y[first].sum())
            evaluation[method]=dict(accuracy=float(hit.mean()),fixed=int((hit&~anchor).sum()),
                harmed=int((~hit&anchor).sum()),equal_count_base_fixed=matched_fix,
                matched_pairs=rank_accuracy(p,base,y,eligible))
            if seed!='v131':
                hist.setdefault(method,[]).append(hit)
        results[seed]=dict(eligible=int(eligible.sum()),misses=int((eligible&y).sum()),models=evaluation)
        np.savez_compressed(OUT/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,baseline=base,**probs)
        print(seed,evaluation['conditional'],flush=True)
    write(REPORT/'tilt.json',dict(plans=results,grouped={
        k:summarize(np.stack(s),np.stack(refs),masks) for k,s in hist.items()}))


if __name__=='__main__':
    with threadpool_limits(2):
        run()
