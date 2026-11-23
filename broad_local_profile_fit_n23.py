"""Fit a local response surface or balance comparison passengers to each query's profile."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import expit, logit, softmax
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_matched_model_changes_n23 import read_bank
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write

REPORT = ROOT / 'reports/nov23_local_profile_fit'


def cohorts(row, y):
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(z['row'], row)
        np.testing.assert_array_equal(z['y'], y)
        wrong = (z['votes'] != y[None,:,None]).all((0,2))
        right = (z['votes'] == y[None,:,None]).all((0,2))
    ever = np.zeros(len(y), bool)
    for seed in map(str, SEEDS):
        with np.load(ART/f'nov23_joint_observed_state/{seed}.npz') as z:
            for k,v in [('row',row),('y',y)]: np.testing.assert_array_equal(z[k],v)
            old = read_bank(row,y,z['fold'],seed).reshape(len(y),-1)
            new = np.column_stack([z[k] for k in ('sleep_conditional','sleep_direct','basket_conditional','basket_direct')])
            ever |= ((np.c_[old,new] >= .5) == y[:,None]).any(1)
    unresolved = wrong & ~ever
    assert [wrong.sum(),right.sum(),unresolved.sum()] == [983,5343,372]
    return wrong,right,unresolved


def coordinates(frame, train):
    # Numeric scales and category membership depend on training attributes alone.
    distance, slopes = [], []
    for field in frame:
        values = frame[field]
        if not pd.api.types.is_numeric_dtype(values):
            for category in values.iloc[train].unique():
                distance.append((values == category).to_numpy(float) / np.sqrt(2))
            continue
        x = values.to_numpy(dtype=float)
        observed = np.isfinite(x)
        positive = observed & (x > 0) if field in BILLS else observed
        reference = np.sort(x[train][positive[train]])
        if not len(reference):
            continue
        ranks = np.zeros(len(x))
        ranks[positive] = (np.searchsorted(reference,x[positive],side='left') +
                           np.searchsorted(reference,x[positive],side='right')) / (2*len(reference))
        distance.extend([ranks,.5*(~observed)])
        slopes.append(ranks)
        if field in BILLS:
            distance.append(.65*positive)
            slopes.append(positive.astype(float))
    return np.column_stack(distance), np.column_stack(slopes)


def balanced_weights(delta, initial):
    """Entropy-regularized calibration toward the query, with a ridge on dual coefficients."""
    n,_,d = delta.shape
    log_initial = np.log(initial)
    theta = np.zeros((n,d))
    identity = np.eye(d)
    for _ in range(7):
        weights = softmax(log_initial + (delta @ theta[:,:,None])[:,:,0],axis=1)
        mean = (weights[:,None,:] @ delta)[:,0]
        centered = delta - mean[:,None,:]
        hessian = centered.transpose(0,2,1) @ (weights[:,:,None]*centered) + .035*identity
        gradient = mean + .035*theta
        step = np.linalg.solve(hessian,gradient[:,:,None])[:,:,0]
        theta -= step / np.maximum(1., np.linalg.norm(step,axis=1,keepdims=True)/2.)
    weights = softmax(log_initial + (delta @ theta[:,:,None])[:,:,0],axis=1)
    return weights


def local_logit(delta, initial, labels):
    n,k,d = delta.shape
    x = np.concatenate([np.ones((n,k,1)),delta],axis=2)
    reg = np.r_[1.,np.full(d,2.)]
    w = np.zeros((n,d+1))
    for _ in range(7):
        p = expit((x @ w[:,:,None])[:,:,0])
        h = k*initial*p*(1-p)
        hessian = x.transpose(0,2,1) @ (h[:,:,None]*x) + np.diag(reg)
        gradient = (x.transpose(0,2,1) @ (k*initial*(p-labels))[:,:,None])[:,:,0] + w*reg
        step = np.linalg.solve(hessian,gradient[:,:,None])[:,:,0]
        w -= step / np.maximum(1.,np.linalg.norm(step,axis=1,keepdims=True)/3.)
    return expit(w[:,0])


def fold_fit(frame,groups,y,t,q):
    assert not np.intersect1d(groups[t],groups[q]).size
    distance,z = coordinates(frame,t)
    pred, quality = {}, []
    for batch in np.array_split(q,int(np.ceil(len(q)/64))):
        raw = cdist(distance[batch],distance[t],'sqeuclidean')
        near = np.argsort(raw,axis=1,kind='stable')[:,:96]
        d = np.take_along_axis(raw,near,axis=1)
        d -= d[:,:1]
        weights = softmax(-d/np.maximum(.025,d[:,[47]]),axis=1)
        donors = t[near]
        delta = z[donors] - z[batch,None]
        target = y[donors].astype(float)
        tilted = balanced_weights(delta,weights)
        score = {
            'ordinary': (96*(weights*target).sum(1)+1)/98,
            'balanced': (96*(tilted*target).sum(1)+1)/98,
            'intercept': local_logit(delta[:,:,:0],weights,target),
            'local_slopes': local_logit(delta,weights,target),
        }
        measures = []
        for w in (weights,tilted):
            measures.extend([np.linalg.norm((w[:,None,:]@delta)[:,0],axis=1),1/np.square(w).sum(1)])
        quality.append(np.column_stack(measures))
        for k,p in score.items(): pred.setdefault(k,[]).append(p)
    return {k:np.concatenate(v) for k,v in pred.items()}, np.vstack(quality)


def main():
    row,y,frame,g,_ = load()
    wrong,right,hard=cohorts(row,y)
    output=ART/'nov23_local_profile_fit'
    output.mkdir(exist_ok=True)
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        bank, diagnostics = {}, np.full((len(y),4),np.nan)
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            scores,diagnostics[q]=fold_fit(frame,g,y,t,q)
            for k,p in scores.items(): bank.setdefault(k,np.full(len(y),np.nan))[q]=p
        decisions={'reference':ref}
        for k,p in bank.items():
            assert np.isfinite(p).all()
            decisions[k]=p>=.5
            decisions['quarter_'+k]=np.where(ref==(op[:,0]>=.5),.75*op[:,0]+.25*p>=.5,ref)
        for k,c in [('balanced','ordinary'),('local_slopes','intercept')]:
            odds = logit(op[:,0].clip(.002,.998))+logit(bank[k].clip(.002,.998))-logit(bank[c].clip(.002,.998))
            decisions['change_'+k]=np.where(ref==(op[:,0]>=.5),odds>=0,ref)
        hits={k:v==y for k,v in decisions.items()}
        results={k:paired(v,ref==y) for k,v in hits.items()}
        plans[seed]=dict(models=results,mean_profile_gap_before_after=diagnostics.mean(0)[[0,2]].tolist(),
                          mean_effective_donors_before_after=diagnostics.mean(0)[[1,3]].tolist())
        if seed!='v131':
            for k,v in hits.items():history.setdefault(k,[]).append(v)
        np.savez_compressed(output/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=op[:,0],**bank)
        print(seed,{k:d['fixed']-d['harmed'] for k,d in results.items()},flush=True)
    r=np.stack(history['reference'])
    stats={k:dict(common_correct=int((np.all(v,axis=0)&wrong).sum()),hard_stable=int((np.all(v,axis=0)&hard).sum()),
                  hard_ever=int((np.any(v,axis=0)&hard).sum()),right_missed=int((~np.any(v,axis=0)&right).sum()),
                  gain95_pp=interval((np.stack(v).astype(float)-r).mean(0),g)) for k,v in history.items()}
    write(REPORT/'comparison.json',dict(plans=plans,stable=stats))


if __name__=='__main__':
    with threadpool_limits(1):main()
