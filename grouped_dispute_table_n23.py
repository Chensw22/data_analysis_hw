"""Group-held-out calibration of manifest-wide disputes, within each fixed base validation fold."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from exact_fold_error_selector import counts
from relational_error_corrector import write


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
SEEDS = [20261112, 20261115, 20261116]


def scores_for_seed(seed, rows):
    components = {}
    for key, path in [("tree","refs/v44"), ("prior","refs/mitra"),
                      ("context","candidates/context"), ("mirror","candidates/mirror")]:
        with np.load(ART / f"top5_group_phase/{path}/group_{seed}.npz") as a:
            np.testing.assert_array_equal(a["dev_ids"], rows.row)
            if components:
                np.testing.assert_array_equal(fold, a["fold_id"])
                np.testing.assert_array_equal(group, a["group_ids"])
            fold, group = a["fold_id"].copy(), a["group_ids"].copy()
            components[key] = a["prob"].copy()
    with np.load(ART / f"nov23_group_kernel/{seed}.npz") as kernel:
        for name, expected in [("row", rows.row), ("fold",fold), ("y",rows.target)]:
            np.testing.assert_array_equal(kernel[name],expected)
        for key in ["base","fixed","learned_twice"]:
            components[key] = kernel[key].copy()
    np.testing.assert_allclose(components["base"],(components["tree"]+components["prior"])/2)
    assert pd.DataFrame({"group":group, "fold":fold}).groupby("group").fold.nunique().max()==1
    return components, fold, group


def cell_posterior(key, errors, train, query, prior, strength):
    n = np.bincount(key[train], minlength=int(key.max())+1)
    success = np.bincount(key[train], weights=errors[train], minlength=len(n))
    return (success[key[query]] + strength * prior) / (n[key[query]] + strength)


def calibrated(c, fold, group, y):
    base = c["base"]
    label = base >= .5
    uncertainty = np.minimum(base, 1-base)
    scores = np.column_stack([c[k] for k in ["context","mirror","fixed","learned_twice"]])
    opposition = np.where(label[:,None], 1-scores, scores)
    # The prior/context and metric families each cast one vote, regardless of how many variants are present.
    prior_vote = opposition[:,:2].mean(1)
    metric_vote = opposition[:,2:].mean(1)
    votes = (prior_vote>=.5).astype(int)+2*(metric_vote>=.5)
    strength = np.digitize(uncertainty, [.10,.25,.40])
    parent = 4*label.astype(int)+strength
    child = 4*parent+votes
    matrix = np.column_stack([uncertainty,label,prior_vote,metric_vote,np.abs(c["prior"]-c["tree"])])
    errors = y != label
    prob = {key: np.full(len(y),np.nan) for key in ["two_level_table","shallow_tree"]}
    for outer in np.unique(fold):
        pool = np.flatnonzero(fold==outer)
        for t,q in GroupKFold(3,shuffle=True,random_state=723).split(pool,groups=group[pool]):
            t,q = pool[t],pool[q]
            assert not np.intersect1d(group[t],group[q]).size
            # Coarse score calibration uses the reported error probability as its prior.
            # Child cells shrink to the coarse rate using eight pseudo-passengers.
            coarse = cell_posterior(parent,errors,t,q,uncertainty[q],20)
            posterior = cell_posterior(child,errors,t,q,coarse,8)
            # Deployment decides which class is more likely with no query outcomes.
            prob["two_level_table"][q] = np.where(label[q],1-posterior,posterior)
            tree=DecisionTreeClassifier(max_depth=3,min_samples_leaf=55,criterion="gini",random_state=23)
            tree.fit(matrix[t],errors[t])
            risk=tree.predict_proba(matrix[q])[:,1]
            prob["shallow_tree"][q] = np.where(label[q],1-risk,risk)
    assert all(np.isfinite(p).all() for p in prob.values())
    return dict(anchor=base, soft_mean=(base+(c["context"]+c["mirror"])/2+
                                        (c["fixed"]+c["learned_twice"])/2)/3, **prob)


def run():
    rows=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    y=rows.target.to_numpy(bool)
    one_time_wrong=rows.v131_wrong.to_numpy(bool) & rows.number_new_correct.eq(0).to_numpy()
    seed_inputs={seed:scores_for_seed(seed,rows) for seed in SEEDS}
    stubborn=one_time_wrong.copy()
    for c,_,_ in seed_inputs.values():
        for k in ["base","fixed","learned_twice"]:
            stubborn &= (c[k]>=.5)!=y
    assert stubborn.sum()==447
    results={}
    out=ART/"nov23_group_disputes"
    out.mkdir(exist_ok=True)
    hit_sets={}
    for seed,(c,fold,group) in seed_inputs.items():
        variants=calibrated(c,fold,group,y)
        results[str(seed)]={}
        for k,p in variants.items():
            pred,anchor=p>=.5,c["base"]>=.5
            metrics=counts(pred,anchor,y,fold,rows.historical_546.to_numpy(bool),stubborn)
            metrics["stubborn447_correct"]=int((stubborn & (pred==y)).sum())
            results[str(seed)][k]=metrics
            hit_sets.setdefault(k,[]).append(stubborn & (pred==y))
        np.savez_compressed(out/f"{seed}.npz",row=rows.row,y=y,fold=fold,**variants)
    write(ROOT/"reports/nov23_group_disputes/result.json",dict(scores=results,stubborn447=447,
        stubborn_rescued_by_seed_count={k:np.bincount(np.sum(masks,axis=0)[stubborn],minlength=4).tolist()
                                       for k,masks in hit_sets.items()},
        note="Nested whole-booking calibration within a single original evaluation fold. The 447-row mask is used only for final scoring; persistent fields refer to this mask."))
    print(results)


if __name__=="__main__":
    run()
