"""Compare shared, separate and sparsely differing error models for each direction of the frozen release."""
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from target_mix_expert_routing_n26 import ROOT, batches, sources, ticker
from release_field_contributions_n26 import blocks


def fit_pair(x,y,side,z,qside,penalty,pooled=False):
    # Each direction has coefficient common +/- deviation. Only deviations incur the sparsity penalty.
    width=x.shape[1]
    dim=width if pooled else 2*width
    def objective(w):
        common=w[:width]
        raw=x@common
        if not pooled:
            delta=w[width:]
            raw=raw+side*(x@delta)
        error=expit(raw)-y
        value=np.sum(np.logaddexp(0.,raw)-y*raw) + 4.*(common[1:]@common[1:])
        gradient=x.T@error
        gradient[1:]+=8.*common[1:]
        if not pooled:
            smooth=np.sqrt(delta**2+.0025)
            value+=4.*(delta@delta)+penalty*np.sum(smooth-.05)
            gradient=np.r_[gradient,x.T@(error*side)+8.*delta+penalty*delta/smooth]
        return value,gradient
    solution=minimize(objective,np.zeros(dim),jac=True,method='L-BFGS-B',options={'maxiter':450,'ftol':1e-10})
    if not solution.success:
        raise RuntimeError(solution.message)
    w=solution.x
    return expit(z@w[:width] if pooled else z@w[:width]+qside*(z@w[width:]))


def transform(x,z):
    # A fixed fill plus explicit missing indicators makes absent service shares distinguishable from zero.
    mask=~np.isfinite(x)
    missing_columns=mask.any(0)
    def assemble(a):
        return np.column_stack([np.nan_to_num(a,nan=0.),(~np.isfinite(a))[:,missing_columns]])
    a,b=assemble(x),assemble(z)
    scale=StandardScaler().fit(a)
    def finish(data):
        return np.column_stack([np.ones(len(data)),scale.transform(data).clip(-6,6)])
    return finish(a),finish(b)


def placebo(side,fields,seed):
    rng=np.random.default_rng(seed)
    out=side.copy()
    risk=fields[:,2]
    cells=np.digitize(risk,np.quantile(risk,[.25,.5,.75]))+4*(fields[:,1]>.5)
    for cell in np.unique(cells):
        ix=np.flatnonzero(cells==cell)
        out[ix]=rng.permutation(side[ix])
    return out


def scores(x,z,train,query,truth,side,qside,sham,qsham):
    xt,xq=transform(x[train],z[query])
    target=truth[train].astype(float)
    s,sq=side[train],qside[query]
    strength=np.sqrt(len(train))*.4
    columns=[fit_pair(xt,target,s,xq,sq,0,pooled=True),
             fit_pair(xt,target,s,xq,sq,0),
             fit_pair(xt,target,s,xq,sq,strength),
             fit_pair(xt,target,sham[train],xq,qsham[query],strength)]
    return np.column_stack(columns)


def run():
    raw,_,g,_=sources()
    physical=np.column_stack(blocks(raw,g))
    methods=['pooled','separate','partial_pool','shuffled_direction']
    results={}
    for b in batches(True):
        s=2*(b.train_cell%2)-1
        sq=2*(b.query_cell%2)-1
        x=np.column_stack([b.train_fields,physical[b.train_ids]])
        z=np.column_stack([b.query_fields,physical[b.ids]])
        sham=placebo(s,b.train_fields,3721)
        qsham=placebo(sq,b.query_fields,4721)
        t,q=np.arange(len(x)),np.arange(len(z))
        predicted=scores(x,z,t,q,b.train_error,s,sq,sham,qsham)
        # Choose the degree of sharing inside each historical calibration set, using only other orders.
        inner=np.full((len(x),4),np.nan)
        for a,c in GroupKFold(3,shuffle=True,random_state=5156).split(x,groups=b.groups):
            assert not np.isin(b.groups[a],b.groups[c]).any()
            inner[c]=scores(x,x,a,c,b.train_error,s,s,sham,sham)
        assert np.isfinite(inner).all()
        best=int(np.argmin([log_loss(b.train_error,inner[:,j],labels=[0,1]) for j in range(3)]))
        data={name:predicted[:,j] for j,name in enumerate(methods)}
        data['selected']=predicted[:,best]
        data['two_reviewers']=np.minimum(data['separate'],data['partial_pool'])
        results.setdefault(b.tag,[]).append((b.ids,sq,b.query_error,b.strict,b.query_cell,data))
    report={}
    for tag,items in results.items():
        ids,side,wrong,strict,cell=[np.concatenate([item[j] for item in items]) for j in range(5)]
        assert len(ids)==len(np.unique(ids))
        scopes={'all':np.ones(len(ids),bool),'predicted_no':side<0,'predicted_yes':side>0,
                'original_common':strict,'added_common':(cell>=4)&~strict}
        baseline=np.concatenate([v[-1]['pooled'] for v in items])>.5
        r={}
        for name in items[0][-1]:
            p=np.concatenate([v[-1][name] for v in items])
            action=p>.5
            r[name]={}
            for label,mask in scopes.items():
                r[name][label]=dict(**ticker(mask,wrong,action),log_loss=float(log_loss(wrong[mask],p[mask],labels=[0,1])),
                                    added=ticker(mask,wrong,action&~baseline),withdrawn=ticker(mask,wrong,~action&baseline))
        report[tag]=r
        print(tag,[(n,v['all']['fixed'],v['all']['harmed'],round(v['all']['log_loss'],4)) for n,v in r.items()],flush=True)
    dest=ROOT/'reports/nov23_matched_learning_curve/paired_error_directions.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
