"""Learn weights for independent whole-booking outcome models on separate calibration bookings."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART,ROOT,SEEDS,hard_rows
from grouped_five_way_coverage_n23 import EXPERIMENTS,verify
from relational_error_corrector import write


def pool(prob,context,y,t,q):
    # The prior gives half the mass to the established baseline, distributed
    # equally among the challengers otherwise. Only calibration labels update
    # weights, including the (regularized) constant column.
    prior=np.full(prob.shape[1],.5/(prob.shape[1]-1))
    prior[0]=.5
    offset=np.log(prior)
    centered=context.copy()
    if context.shape[1]>1:
        centered[:,1:]=(context[:,1:]-context[t,1:].mean(0))/context[t,1:].std(0).clip(.15)
    shape=(centered.shape[1],prob.shape[1])
    def objective(params):
        weight=softmax(centered[t]@params.reshape(shape)+offset,axis=1)
        mean=(weight*prob[t]).sum(1)
        loss=-np.sum(y[t]*np.log(mean)+(1-y[t])*np.log1p(-mean))+2*np.square(params).sum()
        derivative=(mean-y[t])/(mean*(1-mean))
        grad=centered[t].T@(derivative[:,None]*weight*(prob[t]-mean[:,None]))
        return loss,grad.ravel()+4*params
    fit=minimize(objective,np.zeros(np.prod(shape)),jac=True,method="L-BFGS-B",options={"maxiter":300})
    if not fit.success:
        raise RuntimeError(fit.message)
    weights=softmax(centered[q]@fit.x.reshape(shape)+offset,axis=1)
    return (weights*prob[q]).sum(1)


def run():
    catalog=pd.read_csv(ART/'nov23_global_error_inventory.csv')
    dev,y=catalog.row.to_numpy(),catalog.target.to_numpy(bool)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=["PassengerId","CryoSleep","RoomService","Spa","VRDeck","FoodCourt","ShoppingMall"])
                   for s in ["train","test"]],ignore_index=True)
    group=raw.PassengerId.str.split("_").str[0]
    size=group.map(group.value_counts()).to_numpy()[dev]
    hard=hard_rows(dev,y)
    frame=raw.iloc[dev]
    out=ART/"nov23_group_pool"
    out.mkdir(exist_ok=True)
    all_metrics={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            base,fold=z["base"],z["fold"]
            verify(z,dev,y,fold)
        members=[base]
        for folder,key in EXPERIMENTS.items():
            with np.load(ART/f"nov23_{folder}/{seed}.npz") as z:
                verify(z,dev,y,fold)
                members.append(z[key])
        p=np.clip(np.column_stack(members),.003,.997)
        context=np.column_stack([np.ones(len(dev)),np.abs(base-.5),p.std(1),np.log(size),
                                 frame.CryoSleep.eq(True),frame.drop(columns=["CryoSleep","PassengerId"]).isna().any(axis=1)])
        predictions={name:np.full(len(y),np.nan) for name in ["constant","contextual"]}
        for k in np.unique(fold):
            valid=np.flatnonzero(fold==k)
            groups=group.to_numpy()[dev[valid]]
            for a,b in GroupKFold(3,shuffle=True,random_state=323).split(valid,groups=groups):
                assert not np.intersect1d(groups[a],groups[b]).size
                t,q=valid[a],valid[b]
                for name,c in [("constant",context[:,:1]),("contextual",context)]:
                    predictions[name][q]=pool(p,c,y.astype(float),t,q)
        result={}
        for name,score in predictions.items():
            assert np.isfinite(score).all()
            ok=(score>=.5)==y
            bad=(base>=.5)!=y
            result[name]=dict(accuracy=float(ok.mean()),fixed=int((ok&bad).sum()),harmed=int((~ok&~bad).sum()),
                              hard447_correct=int(ok[hard].sum()),fold_accuracy=[float(ok[fold==k].mean()) for k in np.unique(fold)])
        all_metrics[str(seed)]=result
        np.savez_compressed(out/f"{seed}.npz",row=dev,y=y,fold=fold,**predictions)
        print(seed,result,flush=True)
    write(ROOT/"reports/nov23_group_pool/result.json",all_metrics)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
