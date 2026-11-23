"""Fit corrections to v108 only where the strong v131 release inherits its decisions."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import write
from specialist_v131_transfer_n23 import OUT,REPORT,read


def fit(offset,x,y):
    def objective(w):
        z=offset+x@w
        return np.sum(np.logaddexp(0,z)-y*z)+12*(w@w), x.T@(expit(z)-y)+24*w
    model=minimize(objective,np.zeros(x.shape[1]),method='L-BFGS-B',jac=True,
                   options=dict(ftol=1e-12))
    if not model.success:
        raise RuntimeError(model.message)
    return model.x


def features(p108,v80,experts):
    margin=logit(p108.clip(.002,.998))
    opinions=logit(np.c_[v80,experts].clip(.002,.998))
    simple=np.c_[np.ones(len(p108)),margin]
    diff=opinions-margin[:,None]
    votes=(opinions>=0).astype(float)-(margin>=0)[:,None]
    # Average scales across families before computing the total regularization.
    return margin,dict(confidence=simple, residual=np.c_[simple,diff/np.sqrt(diff.shape[1]),
                                                       votes/np.sqrt(votes.shape[1])])


def main():
    row,y,g,fold,v80,p108,v131,cold,experts=read()
    offset,models=features(p108,v80,experts)
    locked=v131!=(p108>=.5)
    inc=v131==y
    methods=['seed481','seed179','ordered']
    allpreds={}
    records={}
    for name,x in models.items():
        output={k:np.full(len(y),np.nan) for k in methods}
        for f in np.unique(fold):
            ix=np.flatnonzero(fold==f)
            splits=[list(GroupKFold(5,shuffle=True,random_state=s).split(ix,groups=g[ix])) for s in [481,179]]
            ordered=[]
            for part in np.array_split(np.unique(g[ix]),5):
                flag=np.isin(g[ix],part)
                ordered.append((np.flatnonzero(~flag),np.flatnonzero(flag)))
            for mode,outer in zip(methods,[*splits,ordered]):
                for a,b in outer:
                    t=ix[a][~locked[ix[a]]]
                    q=ix[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    w=fit(offset[t],x[t],y[t])
                    output[mode][q]=expit(offset[q]+x[q]@w)
        output['random_mean']=.5*(output['seed481']+output['seed179'])
        records[name]={}
        for mode,p in output.items():
            assert np.isfinite(p).all()
            v=np.where(locked,v131,p>=.5)
            allpreds[name+'_'+mode]=v
            hit=v==y
            gain=hit.astype(float)-inc
            records[name][mode]=dict(versus_v131=paired(hit,inc),cold=paired(hit[cold],inc[cold]),
                                      by_fold=[int(gain[fold==f].sum()) for f in np.unique(fold)],
                                      cold_by_fold=[int(gain[(fold==f)&cold].sum()) for f in np.unique(fold)],
                                      ci_pp=interval(gain,g),cold_ci_pp=interval(gain[cold],g[cold]))
        print(name,records[name]['random_mean'],flush=True)
    for mode in [*methods,'random_mean']:
        reference=allpreds['confidence_'+mode]==y
        hit=allpreds['residual_'+mode]==y
        records['residual'][mode]['versus_confidence']=paired(hit,reference)
    np.savez_compressed(OUT/'residual_oof.npz',row=row,y=y,fold=fold,cold=cold,v131=v131,**allpreds)
    write(REPORT/'residual.json',records)


if __name__=='__main__':
    with threadpool_limits(2):
        main()
