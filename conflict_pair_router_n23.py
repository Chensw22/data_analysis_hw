"""Train a feature-dependent ensemble on cases known to demand incompatible weights."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from exaone_validation_data import CATEGORICAL,compact_features
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART,FAMILIES,ROOT,SEEDS,read_panel
from portfolio_threshold_certificate_n23 import pair_bound
from relational_error_corrector import write
from spend_simplex_likelihood_n23 import BILLS
from transported_subclasses_n23 import stats


def routing(prob,features,y,t,q,edges):
    x=features[t]
    p=prob[t]
    midpoint=p-.5
    target=y[t].astype(float)
    margin=midpoint*(2*target[:,None]-1)
    prior=np.full(p.shape[1],.5/(p.shape[1]-1))
    prior[0]=.5
    offset=np.log(prior)
    width=(x.shape[1],p.shape[1])
    if edges is not None:
        a,b=edges
        difference=margin[a]-margin[b]
    def objective(theta):
        pars=theta.reshape(width)
        w=softmax(x@pars+offset,axis=1)
        m=(w*p).sum(1).clip(1.e-6,1-1.e-6)
        loss=-np.sum(target*np.log(m)+(1-target)*np.log1p(-m))+3*np.square(theta).sum()
        grad_w=(m-target)[:,None]*p/(m*(1-m))[:,None]
        if edges is not None:
            # Each training pair should receive weights that improve the sum
            # of its own signed margins relative to swapping the two weight vectors.
            z=((w[a]-w[b])*difference).sum(1)/.1
            loss+=.5*np.logaddexp(0,-z).sum()
            derivative=-5*expit(-z)[:,None]*difference
            np.add.at(grad_w,a,derivative)
            np.add.at(grad_w,b,-derivative)
        deriv=w*(grad_w-(grad_w*w).sum(1)[:,None])
        return loss,(x.T@deriv+6*pars).ravel()
    fit=minimize(objective,np.zeros(np.prod(width)),jac=True,method="L-BFGS-B",
                 options=dict(maxiter=450,ftol=1.e-11))
    if not fit.success:
        raise RuntimeError(fit.message)
    weights=softmax(features[q]@fit.x.reshape(width)+offset,axis=1)
    return (weights*prob[q]).sum(1)


def run():
    rows=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"})
    book=raw.PassengerId.str.split("_").str[0]
    size=book.map(book.value_counts()).to_numpy()[rows]
    group=book.to_numpy()[rows]
    x=compact_features(raw.iloc[rows]).reset_index(drop=True)
    bills=x[BILLS].to_numpy(float)
    design=np.column_stack([pd.get_dummies(x[CATEGORICAL],dtype=float),np.nan_to_num(np.log1p(bills)),
                            np.isnan(bills),np.log1p(size),x.Age.fillna(28)/30,x.Age.isna(),
                            x.CabinNumber.fillna(900)/900,x.CabinNumber.isna()])
    y,old,others=read_panel()
    persist=~old.any(0)
    hard=persist&~others.any(axis=(0,1))
    out=ART/"nov23_conflict_router"
    out.mkdir(exist_ok=True)
    result={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            base,fold=z["base"],z["fold"]
            verify(z,rows,y,fold)
        members=[base]
        for source,key in [*FAMILIES.values(),("simplex_likelihood","budget_prior"),("bart_long","bart"),
                           ("tensor_train","tensor_train"),("registration_rank","matched_registration")]:
            with np.load(ART/f"nov23_{source}/{seed}.npz") as z:
                verify(z,rows,y,fold)
                members.append(z[key])
        p=np.column_stack(members)
        inputs=np.column_stack([design,base,p.std(1)])
        estimates={name:np.full(len(y),np.nan) for name in ["constant","ordinary","conflict","shuffled_pairs"]}
        pairs=[]
        for k in np.unique(fold):
            held=np.flatnonzero(fold==k)
            for it,iq in GroupKFold(3,shuffle=True,random_state=592).split(held,groups=group[held]):
                t,q=held[it],held[iq]
                assert not np.intersect1d(group[t],group[q]).size
                z=(inputs-inputs[t].mean(0))/inputs[t].std(0).clip(.3)
                z=np.column_stack([np.ones(len(y)),z.clip(-5,5)])
                margin=(p[t]-.5)*np.where(y[t],1.,-1.)[:,None]
                _,a,b,_=pair_bound(margin,(base[t]>=.5)==y[t])
                pairs.append(len(a))
                rng=np.random.default_rng(seed+int(k)+int(iq[0]))
                for name,features,edges in [("constant",z[:,:1],None),("ordinary",z,None),
                                            ("conflict",z,(a,b)),("shuffled_pairs",z,(a,rng.permutation(b)))]:
                    estimates[name][q]=routing(p,features,y,t,q,edges)
        m={}
        for name,score in estimates.items():
            assert np.isfinite(score).all()
            m[name]=stats(score>=.5,base>=.5,y,persist,hard,size==1)
        np.savez_compressed(out/f"{seed}.npz",row=rows,y=y,fold=fold,base=base,**estimates)
        result[seed]=dict(metrics=m,pairs_per_calibration_fit=pairs)
        write(ROOT/"reports/nov23_conflict_router/result.json",dict(results=result))
        print(seed,[(n,round(v["accuracy"]*100,3),v["fixed"],v["harmed"]) for n,v in m.items()],flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
