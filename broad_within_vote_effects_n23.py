"""Can continuous scores and passenger records distinguish identical sets of 50 model votes?"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel, predictions
from broad_bill_exception_n23 import load, strata
from broad_local_odds_comparator_n23 import REPORT, manifest_coordinates
from broad_partition_evidence_n23 import EXTRA, SOURCES, SEEDS, write
from broad_release_miss_coverage_n23 import summarize
from released_correction_audit_n23 import development


def within(train, query, index, x):
    n = np.bincount(index[train], minlength=index.max()+1)
    center = np.zeros((len(n),x.shape[1]))
    np.add.at(center,index[train],x[train])
    center /= np.maximum(n[:,None],1)
    a = x[train]-center[index[train]]
    b = x[query]-center[index[query]]
    # At least two training observations are needed for a within-code contrast.
    weight = n/(n+5.)*(n>=2)
    return a*weight[index[train],None], b*weight[index[query],None], n


def fit_predict(train,query,index,z,y,a,b,n):
    cells,ci = np.unique(index[train],return_inverse=True)
    def objective(params):
        intercept,beta = params[:len(cells)],params[len(cells):]
        eta = z[train]+intercept[ci]+a@beta
        error = expit(eta)-y[train]
        loss = np.sum(np.logaddexp(0,eta)-y[train]*eta)
        penalty = 6*(intercept@intercept)+12*(beta@beta)
        grad_int = np.bincount(ci,weights=error,minlength=len(cells))+12*intercept
        return loss+penalty,np.r_[grad_int, a.T@error+24*beta]
    answer = minimize(objective,np.zeros(len(cells)+a.shape[1]),jac=True,method="L-BFGS-B",
                      options={"maxiter":350,"ftol":1e-11})
    assert answer.success,answer.message
    intercept = np.zeros(len(n))
    intercept[cells] = answer.x[:len(cells)]
    return expit(z[query]+intercept[index[query]]+b@answer.x[len(cells):])


def run():
    row,y,base,old,_,_,_ = panel()
    ids,labels,frame,g,_ = load()
    np.testing.assert_array_equal(ids,row)
    np.testing.assert_array_equal(labels,y)
    labels,release,_ = development()
    np.testing.assert_array_equal(labels,y)
    miss = np.logical_and.reduce([p!=y for p in release.values()])
    bill = strata(frame)
    masks = dict(all=miss,paid=miss&y&(bill==2),zero=miss&~y&(bill==0))
    extra = {**EXTRA,**dict(list(SOURCES.items())[:4])}
    raw = manifest_coordinates(frame)
    embedding = RandomTreesEmbedding(n_estimators=28,max_depth=3,random_state=712,sparse_output=False).fit_transform(raw)
    roster = np.c_[np.sqrt(len(frame.columns))*raw,embedding/np.sqrt(28)]
    history,coverage = {},{}
    for i,seed in enumerate(SEEDS):
        _,others,fold = predictions(row,y,seed,extra)
        probs = np.column_stack([old[i],others])
        codes = np.column_stack([probs>=.5,base[i]>=.5,bill])
        _,idx = np.unique(codes,axis=0,return_inverse=True)
        dist = logit(probs.clip(.002,.998))/np.sqrt(50)
        z = logit(base[i].clip(.002,.998))
        p = {k:np.full(len(y),np.nan) for k in ("cell","scores","roster","both","shuffled")}
        seen = np.zeros(len(y),bool)
        for f in np.unique(fold):
            ids = np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
                t,q = ids[a],ids[b]
                assert not np.intersect1d(g[t],g[q]).size
                train_score,query_score,n = within(t,q,idx,dist)
                train_raw,query_raw,_ = within(t,q,idx,roster)
                seen[q] = n[idx[q]]>=2
                full_train = np.c_[train_score,train_raw]
                full_query = np.c_[query_score,query_raw]
                order = np.arange(len(t))
                rng = np.random.default_rng(821)
                for c in np.unique(idx[t]):
                    sel = np.flatnonzero(idx[t]==c)
                    order[sel] = rng.permutation(sel)
                choices = {"cell":(train_raw[:,:0],query_raw[:,:0]),"scores":(train_score,query_score),
                           "roster":(train_raw,query_raw),"both":(full_train,full_query),
                           "shuffled":(full_train[order],full_query)}
                for k,(u,v) in choices.items():
                    p[k][q] = fit_predict(t,q,idx,z,y,u,v,n)
        coverage[str(seed)] = {k:int((seen&m).sum()) for k,m in {"all_rows":np.ones(len(y),bool),**masks}.items()}
        for k,score in p.items():
            assert np.isfinite(score).all()
            history.setdefault(k,[]).append((score>=.5)==y)
    results = {k:summarize(np.stack(v),(base>=.5)==y,masks) for k,v in history.items()}
    write(REPORT/"within_vote_effects.json",dict(matched_at_least_twice=coverage,results=results))
    print(coverage,results)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
