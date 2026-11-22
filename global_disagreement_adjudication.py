"""Decide broad model disagreements on separate bookings inside each original validation fold."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from error_overlap_diagnostic import load_authenticated
from exact_fold_error_selector import counts
from global_error_fusion_n23 import arrays
from relational_error_corrector import write

ROOT = Path(__file__).resolve().parent
ART = ROOT/"artifacts"


def cases(inputs,fold,old):
    y,dev = inputs["y"].astype(bool),inputs["dev"]
    models={}
    for file,keys in [
        ("nov23_joint_booking/oof.npz",{"joint":"p"}),
        ("nov23_honest_neighbors/oof.npz",{"honest_forest":"p"}),
        ("nov23_manifest_denoiser/oof.npz",{"denoiser":"masked","mlp":"supervised"}),
        ("global_tensor_nov23.npz",{"tensor":"p"}),
        ("global_quantile_profiles_n23/oof.csv",{"qboost":"p"}),
        ("reverse_manifest_likelihood_n23/oof.csv",{"reverse":"p","forward_reverse":"bidirectional"})]:
        models.update(arrays(ART/file,dev,fold,y,keys))
    evidence = np.column_stack([*models.values(),*old.values()])
    return models, np.clip(logit(np.clip(evidence,.003,.997)),-5,5)


def run():
    inp,_,fold,old,_=load_authenticated()
    group,dev,y=np.asarray(inp["groups"]),inp["dev"],inp["y"].astype(bool)
    anchor=pd.read_csv(ART/"nov21_case_checks/v136_final_oof.csv")
    for key,val in [("row",dev),("true",y),("fold",fold)]:
        np.testing.assert_array_equal(anchor[key],val)
    models, raw = cases(inp,fold,old)
    # Two strong, distinct historical families give the referee a stable point of comparison.
    reference = (old["v93"] + old["v44"]) / 2
    base = reference >= .5
    opposite = np.column_stack([(p>=.5)!=base for p in models.values()])
    frame=inp["features"].iloc[dev]
    bills=frame[["RoomService","Spa","FoodCourt","ShoppingMall","VRDeck"]]
    by_direction=1-2*base.astype(float)
    confidence=by_direction[:,None]*raw
    strength=np.abs(logit(np.clip(reference,.001,.999)))
    profile=np.column_stack([frame.CryoSleep.eq("True"),frame.HomePlanet.eq("Earth"),
                             frame.HomePlanet.eq("Mars"),np.log1p(bills.sum(axis=1))/8,
                             bills.isna().any(axis=1),frame.Age.fillna(28)/40])
    features=np.column_stack([confidence,strength,opposite.mean(1),np.std(confidence,axis=1),base,
                              by_direction[:,None]*profile,(confidence*base[:,None])])
    eligible=opposite.any(axis=1)
    point=reference.copy()
    weighted=reference.copy()
    # Within each original validation fold, all probabilities come from the SAME
    # out-of-sample model. Split the referee's examples by booking as well.
    for k in np.unique(fold):
        pool=np.flatnonzero(fold==k)
        for t,q in GroupKFold(3,shuffle=True,random_state=1123).split(pool,groups=group[pool]):
            t,q=pool[t],pool[q]
            t,q=t[eligible[t]],q[eligible[q]]
            assert not np.intersect1d(group[t],group[q]).size
            scale=StandardScaler().fit(features[t])
            x,z=scale.transform(features[t]),scale.transform(features[q])
            error=(base[t]!=y[t])
            for target,w in [(point,np.ones(len(t))), (weighted,1/(1+strength[t]))]:
                # The weighted fit targets the small accuracy margins; both fit the same binary error label.
                clf=LogisticRegression(C=.03,max_iter=600).fit(x,error,sample_weight=w/w.mean())
                chance=clf.predict_proba(z)[:,1]
                target[q]=np.where(base[q],1-chance,chance)
    pred={"strong_pair":reference,"error_referee":point,"boundary_weighted":weighted}
    scores={k:counts(p>=.5,anchor.v131.to_numpy(),y,fold,anchor.hard.to_numpy(),
                     anchor.persistent.to_numpy()) for k,p in pred.items()}
    all_scores=np.column_stack([p>=.5 for p in models.values()])
    need=(anchor.v131.to_numpy()!=y)
    covered=need&(all_scores==y[:,None]).any(1)
    remaining=need&~covered
    results=dict(rows=len(y),eligible=int(eligible.sum()),scores=scores,
                 new_experts=list(models),realizable_gain_needed_for_83=int(np.floor(.83*len(y))+1-(~need).sum()),
                 original_errors_covered=int(covered.sum()),original_errors_uncovered=int(remaining.sum()),
                 historical_common_new_coverage=int((anchor.hard&covered).sum()),
                 remaining_missed_positive=int((remaining & y).sum()),remaining_false_alarm=int((remaining & ~y).sum()))
    np.savez_compressed(ART/"global_disagreement_adjudication.npz",row=dev,fold=fold,y=y,**pred)
    write(ROOT/"reports/global_error_models_nov23/adjudication.json",results)
    print(results)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
