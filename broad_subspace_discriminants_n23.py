"""Classify passengers by combining regularized discriminants on random sets of fields."""
import numpy as np
import pandas as pd
from scipy.linalg import cholesky,solve_triangular
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS,load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired,read_opinions
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write

KINDS = ['logistic','shared','diagonal','qda']
REPORT = ROOT/'reports/nov23_subspace_discriminants'
OUT = ART/'nov23_subspace_discriminants'


def field_blocks(frame,t):
    blocks = []
    for name,col in frame.items():
        if pd.api.types.is_numeric_dtype(col):
            v = col.to_numpy(dtype=float)
            if name in BILLS:
                v = np.log1p(v)
            ok = np.isfinite(v)
            fit = v[t][ok[t]]
            v = np.where(ok,v,np.median(fit))
            observed = fit[fit>0] if name in BILLS else fit
            block = [v,~ok]
            if name in BILLS:
                block.append(v>0)
            for knot in np.unique(np.quantile(observed,[.3,.7])):
                block.append(np.maximum(v-knot,0))
            a = np.column_stack(block)
        else:
            levels = col.iloc[t].value_counts().index
            a = np.column_stack([(col==c).to_numpy(float) for c in levels])
        # Sparsely observed categories need a variance floor before applying a ridge.
        mean,sd = a[t].mean(0),a[t].std(0)
        blocks.append(((a-mean)/np.maximum(sd,.18))[:,sd>1e-8].clip(-6,6))
    return blocks


def spaces(n):
    rng = np.random.default_rng(747)
    # Nested widths give the learned combination both narrow and broad alternatives.
    groups = [np.arange(n)]
    for width in [4,7,10]:
        groups.extend([np.sort(rng.choice(n,width,replace=False)) for _ in range(10)])
    return groups


def density(x,mu,cov):
    factor = cholesky(cov,lower=True)
    diff = solve_triangular(factor,(x-mu).T,lower=True)
    return -.5*(diff**2).sum(0)-np.log(np.diag(factor)).sum()


def discriminants(frame,y,t,q):
    blocks = field_blocks(frame,t)
    scores = {k:[] for k in KINDS}
    labels = y[t]
    prior = logit(labels.mean())
    for subset in spaces(len(blocks)):
        x = np.column_stack([blocks[i] for i in subset])
        fit,query = x[t],x[q]
        lm = LogisticRegression(C=.2,max_iter=180).fit(fit,labels)
        scores['logistic'].append(lm.decision_function(query))
        mean = [fit[labels==c].mean(0) for c in [0,1]]
        cov = [np.cov(fit[labels==c],rowvar=False) for c in [0,1]]
        pooled = (1-labels.mean())*cov[0]+labels.mean()*cov[1]
        eye = np.eye(pooled.shape[0])*.15
        for kind in ['shared','diagonal','qda']:
            likelihood = []
            for c in [0,1]:
                actual = pooled if kind=='shared' else .7*cov[c]+.3*pooled
                if kind=='diagonal':
                    actual = np.diag(np.diag(actual))
                likelihood.append(density(query,mean[c],actual+eye))
            scores[kind].append(prior+likelihood[1]-likelihood[0])
    return {k:np.column_stack(v).clip(-12,12) for k,v in scores.items()}


def forecast(frame,g,y,t,q):
    assert not np.intersect1d(g[t],g[q]).size
    calib = {}
    for a,b in GroupKFold(3,shuffle=True,random_state=74).split(t,groups=g[t]):
        assert not np.intersect1d(g[t[a]],g[t[b]]).size
        for kind,v in discriminants(frame,y,t[a],t[b]).items():
            calib.setdefault(kind,np.empty((len(t),v.shape[1])))[b] = v
    scores = discriminants(frame,y,t,q)
    output = {}
    for kind in KINDS:
        for mode in ['average','learned']:
            fit,query = calib[kind],scores[kind]
            if mode=='average':
                fit,query = fit.mean(1)[:,None],query.mean(1)[:,None]
            # Standardize on the out-of-fold calibration predictions only.
            mu,sd = fit.mean(0),fit.std(0).clip(.25)
            model = LogisticRegression(C=.25,max_iter=300).fit((fit-mu)/sd,y[t])
            output[kind+'_'+mode] = model.predict_proba((query-mu)/sd)[:,1]
    return output


def agreement_audit():
    row,y,_,g,_ = load()
    _,right,hard = cohorts(row,y)
    groups,summary = {},{}
    for seed in [*SEEDS,'v131']:
        with np.load(OUT/f'{seed}.npz') as z:
            np.testing.assert_array_equal(row,z['row'])
            ref = z['ref']
            p,d = z['qda_average'],z['diagonal_average']
        agree = (p>=.5)==(d>=.5)
        conf = np.minimum(np.maximum(p,1-p),np.maximum(d,1-d))
        hit = {}
        for cutoff in [.5,.65,.8]:
            active = agree&(conf>=cutoff)
            vote = np.where(active,p>=.5,ref)
            h = vote==y
            hit[str(cutoff)] = h
            summary.setdefault(str(cutoff),{})[str(seed)] = dict(**paired(h,ref==y),hard=int(h[hard].sum()))
        hit['qda'] = (p>=.5)==y
        hit['diagonal'] = (d>=.5)==y
        hit['ref'] = ref==y
        groups[str(seed)] = hit
    group = {k:np.stack([groups[str(s)][k] for s in SEEDS]) for k in hit}
    summary['stable_overlap'] = int((group['qda'].all(0)&group['diagonal'].all(0)&hard).sum())
    for cutoff in [.5,.65,.8]:
        key = str(cutoff)
        h = group[key]
        summary[key]['three_groups'] = dict(stable_hard=int((h.all(0)&hard).sum()),
            ever_hard=int((h.any(0)&hard).sum()),stable_wrong_among_right=int((~h.any(0)&right).sum()),
            ci_vs_reference=interval((h.astype(float)-group['ref']).mean(0),g))
    write(REPORT/'agreement_audit.json',summary)


def main():
    row,y,frame,g,_ = load()
    _,right,hard = cohorts(row,y)
    solo = frame.BookingSize.to_numpy()==1
    REPORT.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    panel,summary = {},dict(folds={},across={})
    for seed in [*SEEDS,'v131']:
        fold,ref,op,_ = read_opinions(seed,row,y)
        scores = {}
        for f in np.unique(fold):
            q = np.flatnonzero(fold==f)
            t = np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            for key,v in forecast(frame,g,y,t,q).items():
                scores.setdefault(key,np.full(len(y),np.nan))[q] = v
        assert all(np.isfinite(v).all() for v in scores.values())
        np.savez_compressed(OUT/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,**scores)
        hits = {'reference':ref==y}
        for key,v in scores.items():
            hits[key] = (v>=.5)==y
            hits[key+'_quarter'] = np.where(ref==(op[:,0]>=.5),.75*op[:,0]+.25*v>=.5,ref)==y
        panel[str(seed)] = hits
        summary['folds'][str(seed)] = {
            key:dict(**paired(h,ref==y),hard=int(h[hard].sum()),solo_hard=int((h&hard&solo).sum()),
                     right_missed=int((~h&right).sum())) for key,h in hits.items()}
        print(seed,[(k,round(100*v.mean(),3),int(v[hard].sum())) for k,v in hits.items() if k in ['reference','qda_learned','qda_learned_quarter']],flush=True)
    cells = {k:np.stack([panel[str(s)][k] for s in SEEDS]) for k in hits}
    for k,h in cells.items():
        shared = k.replace('qda','shared').replace('diagonal','shared')
        summary['across'][k] = dict(hard_all=int((h.all(0)&hard).sum()),hard_any=int((h.any(0)&hard).sum()),
            solo_hard_all=int((h.all(0)&hard&solo).sum()),solo_hard_any=int((h.any(0)&hard&solo).sum()),
            right_wrong_all=int((~h.any(0)&right).sum()),ci_vs_ref=interval((h.astype(float)-cells['reference']).mean(0),g),
            ci_vs_shared=interval((h.astype(float)-cells[shared]).mean(0),g) if k.startswith(('qda','diagonal')) else None)
    write(REPORT/'results.json',summary)
    agreement_audit()


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
