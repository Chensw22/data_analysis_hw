"""Separate outcome ambiguity from parameter uncertainty under a fitted passenger GP."""
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART,ROOT,SEEDS,panel
from disagreement_pattern_n23 import fit_offset
from grouped_five_way_coverage_n23 import verify
from relational_error_corrector import write


def separate(prob,variance):
    pts,w=np.polynomial.hermite.hermgauss(12)
    offset=np.sqrt(2*variance[:,None])*pts
    w=w/np.sqrt(np.pi)
    # Invert the same monotone quadrature used by the GP, avoiding a retrain
    # or rounding-sensitive reconstruction of variational parameters.
    lo=np.full(len(prob),-40.)
    hi=-lo
    for _ in range(52):
        mid=(hi+lo)/2
        vals=expit(mid[:,None]+offset)@w
        lo,hi=np.where(vals<prob,mid,lo),np.where(vals<prob,hi,mid)
    draws=expit((.5*(lo+hi))[:,None]+offset)
    np.testing.assert_allclose(draws@w,prob,atol=1e-8)
    uncertainty=np.square(draws-prob[:,None])@w
    ambiguity=(draws*(1-draws))@w
    np.testing.assert_allclose(ambiguity+uncertainty,prob*(1-prob),atol=1e-8)
    return ambiguity,uncertainty


def conditional_auc(labels,score,bands):
    weighted=0.
    pairs=0
    for group in np.unique(bands):
        m=bands==group
        n=int(labels[m].sum())*int((~labels[m]).sum())
        if n:
            weighted+=n*roc_auc_score(labels[m],score[m])
            pairs+=n
    return weighted/pairs


def report_cohort(mask,wrong,u,a):
    return dict(rows=int(mask.sum()),errors=int((mask&wrong).sum()),ambiguity=float(a[mask].mean()),
                parameter_variance=float(u[mask].mean()),parameter_share=float(u[mask].sum()/(u[mask]+a[mask]).sum()))


def run():
    row,y,b,bank,_,_,_=panel()
    ref=(b>=.5)==y
    hits=(bank>=.5)==y[None,:,None]
    strong=(hits.mean(1)>=ref.mean(1)[:,None]-.01).all(0)
    repeated=~ref.any(0)
    weak=repeated&hits.all(0).any(1)&~hits.all(0)[:,strong].any(1)
    deep=repeated&~hits.any(axis=(0,2))
    assert (weak.sum(),deep.sum())==(335,252)
    group=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.str.split("_").str[0].to_numpy()[row]
    out=ART/"nov23_error_variance"
    out.mkdir(exist_ok=True)
    report={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_variational_gp/{seed}.npz") as z:
            fold,base,prob,v=(z[k] for k in ("fold","base","p","latent_variance"))
            verify(z,row,y,fold)
        a,u=separate(prob,v)
        wrong=(base>=.5)!=y
        risk=np.minimum(prob,1-prob)
        error=(prob>=.5)!=y
        bands=np.searchsorted(np.unique(np.quantile(risk,np.arange(1,11)/11)),risk)
        auc={}
        for key,mask in [("strong_reference_error",wrong),("gp_error",error)]:
            auc[key]={name:dict(marginal=float(roc_auc_score(mask,score)),
                                within_gp_risk=float(conditional_auc(mask,score,bands)))
                      for name,score in [("ambiguity",a),("parameter_variance",u),("risk",risk)]}
        prior=logit(base.clip(.001,.999))
        gp_level=logit(prob.clip(.001,.999))
        both=np.column_stack([np.ones(len(y)),prior,gp_level-prior])
        uncertainty=np.log(np.maximum(u,1e-8))
        candidates={k:np.full(len(y),np.nan) for k in ("probabilities","variance","shuffled")}
        for f in np.unique(fold):
            held=np.flatnonzero(fold==f)
            for aa,bb in GroupKFold(3,shuffle=True,random_state=115).split(held,groups=group[held]):
                t,q=held[aa],held[bb]
                assert not np.intersect1d(group[t],group[q]).size
                real=(uncertainty-uncertainty[t].mean())/max(.1,uncertainty[t].std())
                placebo=real.copy()
                rng=np.random.default_rng(492)
                for part in (t,q):
                    for band in np.unique(bands[part]):
                        ix=part[bands[part]==band]
                        placebo[ix]=real[rng.permutation(ix)]
                for name,more in [("probabilities",None),("variance",real),("shuffled",placebo)]:
                    design=both if more is None else np.column_stack([both,more,more*prior,more*(gp_level-prior)])
                    penalties=np.r_[[8.,30.,30.],np.full(design.shape[1]-3,50.)]
                    candidates[name][q]=fit_offset(t,q,design,base,y,penalties)
        metrics={}
        for name,p in candidates.items():
            assert np.isfinite(p).all()
            hit=(p>=.5)==y
            metrics[name]=dict(accuracy=float(hit.mean()),fixed=int((hit&wrong).sum()),harmed=int((~hit&~wrong).sum()),
                               weak335=int(hit[weak].sum()),deep252=int(hit[deep].sum()))
        cohorts={name:report_cohort(mask,wrong,u,a) for name,mask in [("all",np.ones(len(y),bool)),("always_correct",ref.all(0)),
                ("always_wrong",repeated),("only_weaker_stable",weak),("all40_wrong",deep)]}
        report[seed]=dict(auc=auc,cohorts=cohorts,metrics=metrics)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,ambiguity=a,parameter_variance=u,**candidates)
        print(seed,auc["gp_error"],metrics["variance"],flush=True)
    write(ROOT/"reports/nov23_error_variance/result.json",dict(results=report))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
