"""Find exceptions inside unanimous predictions using magnitudes of independent model opinions."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from scipy.stats import multivariate_t
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART,FAMILIES,ROOT,SEEDS,read_panel
from relational_error_corrector import write
from transported_subclasses_n23 import stats


def conditional_opinions(prior, opinions, learn, ids):
    s=logit(prior.clip(.001,.999))
    basis=np.column_stack([np.ones(len(s)),s,np.maximum(s+2.,0),np.maximum(s+1.,0)])
    x=logit(opinions.clip(.002,.998))
    a=basis[learn]
    mean=np.linalg.solve(a.T@a+2*np.eye(a.shape[1]),a.T@x[learn])
    residual=x-basis@mean
    scale=residual[learn].std(0).clip(.15)
    return (residual[ids]/scale).clip(-5,5)


def ratio(train, target, query, diagonal):
    global_cov=np.cov(train,rowvar=False)+.15*np.eye(train.shape[1])
    densities=[]
    for label in [False,True]:
        part=train[target==label]
        n=len(part)
        center=part.sum(0)/(n+10)
        deviation=part-center
        cov=(deviation.T@deviation+50*global_cov)/(n+50)+.15*np.eye(train.shape[1])
        if diagonal:
            cov=np.diag(cov.diagonal())
        densities.append(multivariate_t.logpdf(query,loc=center,shape=cov,df=6))
    return np.atleast_1d(densities[1]-densities[0]).clip(-6,6)


def calibrate(prior,evidence,target,test_prior,test_evidence):
    z=logit(prior.clip(.001,.999))
    def objective(v):
        score=z+v[0]+v[1]*evidence
        r=expit(score)-target
        return (np.logaddexp(0.,score).sum()-score@target+v[0]**2+3*v[1]**2,
                np.array([r.sum()+2*v[0],r@evidence+6*v[1]]))
    fit=minimize(objective,[0.,0.],jac=True,method="L-BFGS-B",bounds=[(-3,3),(0,2)])
    if not fit.success:
        raise RuntimeError(fit.message)
    return expit(logit(test_prior.clip(.001,.999))+fit.x[0]+fit.x[1]*test_evidence)


def run():
    rows=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    g=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.str.split("_").str[0]
    single=g.map(g.value_counts()).to_numpy()[rows]==1
    group=g.to_numpy()[rows]
    y,old,others=read_panel()
    persistent=~old.any(0)
    hard=persistent&~others.any(axis=(0,1))
    out=ART/"nov23_consensus_opinions"
    out.mkdir(exist_ok=True)
    result={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,rows,y,fold)
        members=[]
        for experiment,key in FAMILIES.values():
            with np.load(ART/f"nov23_{experiment}/{seed}.npz") as z:
                verify(z,rows,y,fold)
                members.append(z[key])
        members=np.column_stack(members)
        base_decision=base>=.5
        same=np.all((members>=.5)==base_decision[:,None],axis=1)
        errors=y!=base_decision
        scores={k:base.copy() for k in ["prior_calibration","individual_opinions","joint_opinions"]}
        for k in np.unique(fold):
            held=np.flatnonzero(fold==k)
            for direction in [False,True]:
                pool=held[same[held] & (base_decision[held]==direction)]
                prior=1-base if direction else base
                opinions=1-members if direction else members
                for a,b in GroupKFold(3,shuffle=True,random_state=151).split(pool,groups=group[pool]):
                    t,q=pool[a],pool[b]
                    assert not np.intersect1d(group[t],group[q]).size
                    estimates={n:[] for n in scores}
                    for d,c in GroupKFold(2,shuffle=True,random_state=322).split(t,groups=group[t]):
                        learn,cal=t[d],t[c]
                        assert not np.intersect1d(group[learn],group[cal]).size
                        indices=np.r_[learn,cal,q]
                        residual=conditional_opinions(prior,opinions,learn,indices)
                        x,z=residual[:len(learn)],residual[len(learn):]
                        for name in scores:
                            signal=(np.zeros(len(z)) if name=="prior_calibration" else
                                    ratio(x,errors[learn],z,name=="individual_opinions"))
                            value=calibrate(prior[cal],signal[:len(cal)],errors[cal],prior[q],signal[len(cal):])
                            estimates[name].append(value)
                    for name,m in estimates.items():
                        risk=np.mean(m,axis=0)
                        scores[name][q]=1-risk if direction else risk
        metrics={}
        for name,p in dict(base=base,**scores).items():
            assert np.isfinite(p).all()
            m=stats(p>=.5,base_decision,y,persistent,hard,single)
            for direction in [False,True]:
                selected=same&(base_decision==direction)
                risk=1-p if direction else p
                label="false_alarms" if direction else "missed_transports"
                m[label]=dict(rows=int(selected.sum()),errors=int(errors[selected].sum()),
                              auc=float(roc_auc_score(errors[selected],risk[selected])),
                              max_error_probability=float(risk[selected].max()))
            metrics[name]=m
        result[seed]=metrics
        np.savez_compressed(out/f"{seed}.npz",row=rows,y=y,fold=fold,base=base,unanimous=same,**scores)
        print(seed,{k:(m["fixed"],m["harmed"],round(m["missed_transports"]["auc"],3)) for k,m in metrics.items()},flush=True)
    write(ROOT/"reports/nov23_consensus_opinions/result.json",dict(results=result))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
