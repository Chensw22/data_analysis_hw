"""Learn how much conditional field evidence is useful over the incumbent on unseen bookings."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_class_transport_n23 import encode
from broad_dependence_anchors_n23 import REPORT, likelihoods
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, SEEDS, write


def fit(x, offset, y):
    # Zero coefficients exactly reproduce the incumbent. Regularize the intercept
    # and field coefficients equally, after centering and scaling on training bookings.
    def objective(w):
        margin = offset+x@w
        return np.sum(np.logaddexp(0, margin)-y*margin)+4*(w@w), x.T@(expit(margin)-y)+8*w
    result = minimize(objective,np.zeros(x.shape[1]),method='L-BFGS-B',jac=True)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def run_seed(x,y,g,fold,base,ref):
    widths = x.max(0)+1
    anchor = logit(base.clip(.002,.998))
    adjustable = ref==(base>=.5)
    output = {}
    contribution = np.full_like(x,np.nan,dtype=float)
    for f in np.unique(fold):
        q = np.flatnonzero(fold==f)
        t = np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
        assert not np.intersect1d(g[t],g[q]).size
        arms = [likelihoods(x[t[y[t]==label]],x[q],widths,field_conditionals=True) for label in (False,True)]
        marginal, cond = [arms[1][k]-arms[0][k] for k in ('marginal_fields','conditional_fields')]
        difference = cond-marginal
        contribution[q] = difference
        inputs = {'anchor':np.empty((len(q),0)), 'marginal':marginal, 'conditional':cond,
                  'dependence_only':difference, 'both':np.column_stack([marginal,difference])}
        for a,b in GroupKFold(3,shuffle=True,random_state=589).split(q,groups=g[q]):
            assert not np.intersect1d(g[q[a]],g[q[b]]).size
            a = a[adjustable[q[a]]]
            for k,fields in inputs.items():
                raw = np.column_stack([anchor[q],fields])
                xfit = (raw-raw[a].mean(0))/np.maximum(raw[a].std(0),.10)
                design = np.column_stack([np.ones(len(q)),xfit])
                weights = fit(design[a],anchor[q[a]],y[q[a]])
                p = expit(anchor[q[b]]+design[b]@weights)
                output.setdefault(k,np.full(len(y),np.nan))[q[b]] = np.where(adjustable[q[b]],p,ref[q[b]].astype(float))
    assert np.isfinite(contribution).all() and all(np.isfinite(p).all() for p in output.values())
    return output,contribution


def main():
    row,y,frame,g,_ = load()
    wrong,right,hard = cohorts(row,y)
    x = encode(frame)
    directory = ART/'nov23_conditional_field_values'
    directory.mkdir(exist_ok=True)
    plans,history = {},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as data:
            np.testing.assert_array_equal(row,data['row'])
            np.testing.assert_array_equal(y,data['y'])
            fold,base,ref = data['fold'],data['base'],data['ref']
        output,delta = run_seed(x,y,g,fold,base,ref)
        hits = {k:(v>=.5)==y for k,v in output.items()}
        hits['reference'] = ref==y
        plans[seed] = {k:paired(v,ref==y) for k,v in hits.items()}
        if seed!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        np.savez_compressed(directory/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=base,
                            field_names=np.array(frame.columns,dtype=str),contribution=delta,**output)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in plans[seed].items()},flush=True)
    baseline = np.stack(history['reference'])
    stable = {}
    for k,h in history.items():
        h = np.stack(h)
        stable[k] = dict(hard_stable=int((hard & h.all(0)).sum()),hard_ever=int((hard & h.any(0)).sum()),
                         common_stable=int((wrong & h.all(0)).sum()),stable_harms=int((right & ~h.any(0)).sum()),
                         gain95_pp=interval((h.astype(float)-baseline).mean(0),g))
    write(REPORT/'conditional_fields.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
