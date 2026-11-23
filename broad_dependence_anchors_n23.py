"""Integrate one- and two-field conditioning models of a passenger's whole record."""
from itertools import combinations
import numpy as np
from scipy.special import expit, logit, logsumexp
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_class_transport_n23 import encode
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write

REPORT = ROOT/'reports/nov23_dependence_anchors'


def likelihoods(x, query, widths, smooth=12., field_conditionals=False):
    """One normalized joint density per anchor or unordered anchor pair for one outcome."""
    n,d = x.shape
    assert len(x) and (x>=0).all() and (query>=0).all()
    cols = np.arange(d)
    singles = [np.bincount(x[:,i],minlength=w) for i,w in enumerate(widths)]
    marginal = [(c+1/w)/(n+1) for c,w in zip(singles,widths)]
    unary = np.column_stack([np.log(m[query[:,i]]) for i,m in enumerate(marginal)])
    pairs = {}
    conditional = {}
    anchored = unary.copy()
    for i,j in combinations(cols,2):
        shape = (widths[i],widths[j])
        pair = np.bincount(x[:,i]*widths[j]+x[:,j], minlength=np.prod(shape)).reshape(shape)
        pairs[i,j] = pair
        for parent,child,table in [(i,j,pair),(j,i,pair.T)]:
            probs = (table+smooth*marginal[child])/(singles[parent][:,None]+smooth)
            conditional[parent,child] = probs
            anchored[:,parent] += np.log(probs[query[:,parent],query[:,child]])
    double = []
    if field_conditionals:
        complete = np.full_like(unary, -np.inf)
        withheld = np.full_like(unary, -np.inf)
    for (i,j),counts in pairs.items():
        # Both orders define proper parent-pair probabilities. Average before scoring.
        root = .5*(marginal[i][:,None]*conditional[i,j]+(marginal[j][:,None]*conditional[j,i]).T)
        log_density = np.log(root[query[:,i],query[:,j]])
        code = x[:,i]*widths[j]+x[:,j]
        children = cols[(cols!=i)&(cols!=j)]
        factors = {}
        for k in children:
            w = widths[k]
            triples = np.bincount(code*w+x[:,k],minlength=counts.size*w).reshape(*counts.shape,w)
            observed = triples[query[:,i],query[:,j],query[:,k]]
            prior = .5*(conditional[i,k][query[:,i],query[:,k]]+conditional[j,k][query[:,j],query[:,k]])
            prob = (observed+smooth*prior)/(counts[query[:,i],query[:,j]]+smooth)
            factor = np.log(prob)
            log_density += factor
            if field_conditionals:
                factors[k] = factor
        if field_conditionals:
            for k,factor in factors.items():
                # The field is a child in every included model; summing out its
                # possible values removes exactly this normalized likelihood factor.
                complete[:,k] = np.logaddexp(complete[:,k],log_density)
                withheld[:,k] = np.logaddexp(withheld[:,k],log_density-factor)
        double.append(log_density)
    result = {'independent':unary.sum(1)[:,None],'single':anchored,'double':np.column_stack(double)}
    if field_conditionals:
        result.update(marginal_fields=unary,conditional_fields=complete-withheld)
    return result


def predictions(x, y, t, q, shuffle=False):
    widths = x.max(0)+1
    by_class = []
    rng = np.random.default_rng(121)
    for label in (False,True):
        train = x[t[y[t]==label]].copy()
        if shuffle:
            # Preserve each class's field histograms exactly, destroy record-level dependence.
            for j in range(train.shape[1]):
                rng.shuffle(train[:,j])
        joint = likelihoods(train,x[q],widths)
        by_class.append({k:v+np.log(len(train)/len(t)) for k,v in joint.items()})
    a,b = by_class
    predictions = {}
    for k in a:
        predictions[k] = expit(logsumexp(b[k],axis=1)-logsumexp(a[k],axis=1))
        if k!='independent':
            predictions[k+'_uniform_vote'] = expit(b[k]-a[k]).mean(1)
    return predictions


def main():
    row,y,frame,groups,_ = load()
    wrong,right,hard = cohorts(row,y)
    x = encode(frame)
    directory = ART/'nov23_dependence_anchors'
    directory.mkdir(exist_ok=True)
    plans,history = {},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as data:
            np.testing.assert_array_equal(row,data['row'])
            np.testing.assert_array_equal(y,data['y'])
            fold,ref,base = data['fold'],data['ref'],data['base']
        bank = {}
        for f in np.unique(fold):
            query = np.flatnonzero(fold==f)
            train = np.flatnonzero((fold!=f)&~np.isin(groups,groups[query]))
            assert not np.intersect1d(groups[train],groups[query]).size
            pred = predictions(x,y,train,query)
            placebo = predictions(x,y,train,query,shuffle=True)
            np.testing.assert_array_equal(pred['independent'],placebo['independent'])
            pred['shuffled_double'] = placebo['double']
            for k,p in pred.items():
                bank.setdefault(k,np.full(len(y),np.nan))[query] = p
        hit = {'reference':ref==y}
        active = ref==(base>=.5)
        for k,p in bank.items():
            assert np.isfinite(p).all()
            hit[k] = (p>=.5)==y
            hit['quarter_'+k] = np.where(active,.75*base+.25*p>=.5,ref)==y
        for source,other in [('double','single'),('single','independent')]:
            increment = logit(bank[source].clip(.002,.998))-logit(bank[other].clip(.002,.998))
            hit[source+'_increment'] = np.where(active,logit(base.clip(.002,.998))+.25*increment>=0,ref)==y
        plans[seed] = {k:paired(v,ref==y) for k,v in hit.items()}
        if seed!='v131':
            for k,v in hit.items():
                history.setdefault(k,[]).append(v)
        np.savez_compressed(directory/f'{seed}.npz',row=row,y=y,fold=fold,base=base,ref=ref,**bank)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in plans[seed].items()},flush=True)
    baseline = np.stack(history['reference'])
    stable = {}
    for k,h in history.items():
        h = np.stack(h)
        stable[k] = dict(hard_stable=int((hard & h.all(0)).sum()),hard_ever=int((hard & h.any(0)).sum()),
                         common_stable=int((wrong & h.all(0)).sum()),stable_harms=int((right & ~h.any(0)).sum()),
                         gain95_pp=interval((h.astype(float)-baseline).mean(0),groups))
    write(REPORT/'results.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
