"""Combine two independently fitted alternatives and check their incremental population coverage."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold

from archive_error_recovery_n23 import panel, predictions
from broad_booking_policy_n23 import EXTRA
from broad_booking_policy_replay_n23 import interval
from broad_error_placement_control_n23 import SOURCES
from broad_geographic_transfer_n23 import load
from broad_weak_disagreement_n23 import ART, OUT, REPORT, SEEDS, archive
from relational_error_corrector import write


def combine(base, alternatives, y, learn, query):
    odds = logit(base.clip(.002,.998))
    contrasts = logit(alternatives.clip(.002,.998))-odds[:,None]
    x = np.column_stack([np.ones(len(y)), odds, contrasts])
    penalty = np.array([4.,10.,*([20.]*alternatives.shape[1])])
    def objective(beta):
        z = odds[learn]+x[learn]@beta
        value = np.sum(np.logaddexp(0,z)-y[learn]*z) + .5*penalty@(beta*beta)
        return value, x[learn].T@(expit(z)-y[learn])+penalty*beta
    result = minimize(objective,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return expit(odds[query]+x[query]@result.x)


def matched_order(scores, base, eligible, errors):
    risks = np.where(base>=.5,1-scores,scores)
    conf_risk = np.minimum(base,1-base)
    cells = (np.floor(conf_risk/.02).astype(int)*2 + (base>=.5))
    pairs = favorable = 0
    for b in np.unique(cells[eligible]):
        group = eligible & (cells==b)
        positives, negatives = risks[group & errors],risks[group & ~errors]
        diff = positives[:,None]-negatives
        pairs += diff.size
        favorable += (diff>0).sum()+.5*(diff==0).sum()
    return dict(pairs=int(pairs),accuracy=float(favorable/pairs))


def run():
    row,y,_,groups,_=load()
    rows,labels,bases,set1594,masks,_,_=archive()
    np.testing.assert_array_equal(rows,row)
    np.testing.assert_array_equal(labels,y)
    _,_,_,old,_,_,_=panel()
    methods={**EXTRA,**dict(list(SOURCES.items())[:4])}
    adds=np.stack([predictions(row,y,s,methods)[1] for s in SEEDS])
    old_stable=((np.concatenate([old,adds],axis=2)>=.5)==y[None,:,None]).all(0).any(1)
    previous_unresolved=masks["repeat"]&~old_stable
    assert previous_unresolved.sum()==509
    stats,history={},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(OUT/f"objective_{name}.npz") as f, np.load(ART/f"nov23_broad_service_transport/{name}.npz") as h:
            for key in ("row","y","fold"):
                np.testing.assert_array_equal(f[key],h[key])
            base,ref,fold=f["base"],f["ref"],f["fold"]
            active=np.column_stack([f["direct_smooth_count"],h["learned"]])
            control=np.column_stack([f["direct_logistic"],h["star"]])
        pred={"boundary":active[:,0],"services":active[:,1]}
        for method,family in (("base_only",active[:,:0]),("combined",active),("matched_models",control)):
            scores=np.full(len(y),np.nan)
            for k in np.unique(fold):
                outer=np.flatnonzero(fold==k)
                for a,b in GroupKFold(3,shuffle=True,random_state=667).split(outer,groups=groups[outer]):
                    t,q=outer[a],outer[b]
                    assert not np.intersect1d(groups[t],groups[q]).size
                    scores[q]=combine(base,family,y,t,q)
            pred[method]=scores
        stats[name]={}
        for method,score in pred.items():
            assert np.isfinite(score).all()
            fix_old=(ref!=(base>=.5)) & (method not in ("boundary","services"))
            hit=np.where(fix_old,ref,score>=.5)==y
            gain=hit.astype(int)-(ref==y)
            stats[name][method]=dict(accuracy=float(hit.mean()),fixed=int((gain>0).sum()),harmed=int((gain<0).sum()),
                                     net_by_fold=[int(gain[fold==k].sum()) for k in np.unique(fold)])
            if name!="v131":
                history.setdefault(method,[]).append((hit,gain))
                stats[name][method]["matched_mirror_order"]=matched_order(score,base,set1594,masks["weak409"])
        np.savez_compressed(OUT/f"complement_{name}.npz",row=row,y=y,fold=fold,**pred)
    stable={}
    for method,data in history.items():
        correct=np.logical_and.reduce([v[0] for v in data])
        stable[method]={k:int((correct&m).sum()) for k,m in masks.items()}
        stable[method]["new_vs_50"]=int((correct&previous_unresolved).sum())
        stable[method]["paired_gain_ci95_pp"]=interval(np.mean([v[1] for v in data],0),groups)
    union=np.logical_or.reduce([np.logical_and.reduce([v[0] for v in history[k]]) for k in ("services","boundary")])
    summary={k:int((m&union).sum()) for k,m in masks.items()}
    summary["new_vs_50"]=int((union&previous_unresolved).sum())
    write(REPORT/"complement.json",dict(plans=stats,stable=stable,oracle_union_of_direct_models=summary))
    print(stable,flush=True)


if __name__=="__main__":
    run()
