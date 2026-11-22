"""Measure error coverage and learn when to trust models on unseen passenger groups."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from error_overlap_diagnostic import load_authenticated
from exact_fold_error_selector import cached, counts
from relational_error_corrector import write

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
OUT = ROOT / "reports/global_error_models_nov23"


def arrays(path, dev, fold, y, columns):
    if path.suffix == ".csv":
        archive = pd.read_csv(path)
    else:
        with np.load(path,allow_pickle=False) as data:
            archive = {k:data[k] for k in ["row","fold","y",*columns.values()]}
    for key, value in [("row",dev),("fold",fold),("y",y)]:
        np.testing.assert_array_equal(archive[key],value)
    out = {name:np.asarray(archive[key],float) for name,key in columns.items()}
    for value in out.values():
        assert value.shape == y.shape and np.isfinite(value).all() and ((value>=0)&(value<=1)).all()
    return out


def residual_fit(x, offset, label, train, query):
    # Fit an offset logistic model with a Gaussian prior centered on no adjustment.
    def objective(w):
        z = offset[train]+x[train]@w
        loss = np.logaddexp(0,z).sum() - label[train]@z + 5*(w@w)
        grad = x[train].T@(expit(z)-label[train])+10*w
        return loss, grad
    fitted = minimize(objective,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",
                      options={"maxiter":300,"ftol":1e-11})
    if not fitted.success:
        raise RuntimeError(fitted.message)
    return expit(offset[query]+x[query]@fitted.x)


def run():
    inp, _, fold, old, _ = load_authenticated()
    dev, group, y = inp["dev"], np.asarray(inp["groups"]), inp["y"].astype(bool)
    champion = pd.read_csv(ART/"nov21_case_checks/v136_final_oof.csv")
    for key, v in [("row",dev),("fold",fold),("true",y)]:
        np.testing.assert_array_equal(champion[key],v)
    anchor = champion.v131.to_numpy(bool)
    hard = champion.hard.to_numpy(bool)
    persistent = champion.persistent.to_numpy(bool)
    raw = cached("v108_exact_oof/oof.npz","v108",dev,fold)
    prior = dict(old, v108=raw)
    sources = [
        ("global_quantile_profiles_n23/oof.csv",{"quantile":"p","raw_catboost":"p_raw"}),
        ("nov23_honest_neighbors/oof.npz",{"honest_neighbors":"p","forest":"p_forest"}),
        ("nov23_manifest_denoiser/oof.npz",{"denoiser":"masked","supervised_mlp":"supervised"}),
        ("global_tensor_nov23.npz",{"tensor":"p","marginals":"marginal"})]
    new = {}
    for filename,columns in sources:
        new.update(arrays(ART/filename,dev,fold,y,columns))
    historical = np.column_stack([p>=.5 for p in prior.values()])
    broad = np.column_stack([p>=.5 for p in new.values()])
    was_wrong = anchor!=y
    def coverage(mask):
        return dict(rows=int(mask.sum()),anchor_wrong=int((mask&was_wrong).sum()),
                    fixed_by_some_new=int((mask&was_wrong&(broad==y[:,None]).any(1)).sum()),
                    all_old_wrong_new_can_fix=int((mask&was_wrong&(historical!=y[:,None]).all(1)&(broad==y[:,None]).any(1)).sum()),
                    all_old_and_new_and_anchor_wrong=int((mask&was_wrong&(historical!=y[:,None]).all(1)&(broad!=y[:,None]).all(1)).sum()))
    reference = prior["rawMitra"]
    scores = np.column_stack(list(prior.values())+list(new.values()))
    logits = logit(np.clip(scores,.005,.995))
    # The offset defines a reproducible comparison. No additive information is derived from the true error mask.
    offset = logit(np.clip(reference,.005,.995))
    contrasts = np.clip(logits-offset[:,None],-6,6)
    simple = np.column_stack([np.ones(len(y)),offset,contrasts])
    manifest = inp["features"].iloc[dev]
    expense = manifest[["FoodCourt","ShoppingMall","RoomService","Spa","VRDeck"]]
    context = np.column_stack([np.abs(offset)/3,
        np.std(logits,axis=1),manifest.CryoSleep.eq("True"),expense.isna().any(axis=1),
        expense.eq(0).all(axis=1),np.log1p(expense.sum(axis=1))/8,manifest.HomePlanet.eq("Earth")])
    conditioned = np.column_stack([simple,context,(contrasts[:,:,None]*context[:,None,:]).reshape(len(y),-1)])
    pred = {"fixed_equal_families":(.5*(reference+raw)*.5+
            .5*np.mean(np.column_stack([new[k] for k in ["quantile","honest_neighbors","denoiser","tensor"]]),axis=1)),
            "learned_global":np.full(len(y),np.nan), "learned_conditional":np.full(len(y),np.nan)}
    # Calibrate only within the same base-evaluation fold. Therefore every meta-training
    # row used the same base fits as the query; the base fits never saw query outcomes.
    for k in np.unique(fold):
        outer = np.flatnonzero(fold==k)
        for t,q in GroupKFold(3,shuffle=True,random_state=1123).split(outer,groups=group[outer]):
            t,q=outer[t],outer[q]
            assert not np.intersect1d(group[t],group[q]).size
            for key, x in [("learned_global",simple),("learned_conditional",conditioned)]:
                pred[key][q]=residual_fit(x,offset,y.astype(float),t,q)
    for p in pred.values():
        assert np.isfinite(p).all()
    metrics={k:counts(p>=.5,anchor,y,fold,hard,persistent) for k,p in {**prior,**new,**pred}.items()}
    wrong_new = was_wrong[:,None]&(broad==y[:,None])
    result=dict(anchor_correct=int((~was_wrong).sum()), metrics=metrics, new_names=list(new),
                coverage={label:coverage(mask) for label,mask in [("all",np.ones(len(y),bool)),("hard546",hard),
                         ("persistent510",persistent),("false_positive",anchor),("false_negative",~anchor)]},
                paired_corrected_overlap=(wrong_new.astype(int).T@wrong_new.astype(int)).tolist(),
                calibration="3 group-separated meta folds within each original base validation fold. Historical base fits use the original random folds. Historical error cohorts only used after scoring.")
    write(OUT/"fusion.json",result)
    np.savez_compressed(ART/"global_error_fusion_n23.npz",row=dev,fold=fold,y=y,**pred)
    print({k:metrics[k] for k in pred},result["coverage"],flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
