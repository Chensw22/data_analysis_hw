"""Reconcile five experiments, retaining genuine candidate methods across all three group assignments."""
from pathlib import Path

import numpy as np
import pandas as pd

from grouped_dispute_table_n23 import SEEDS
from relational_error_corrector import write


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
EXPERIMENTS = {
    "boarding_sequence": "sequence", "projection_shapes": "projection_shapes",
    "peer_attr_surprise": "self_peer_surprise", "manifest_tree": "manifest_tree",
}


def verify(cache, row, y, fold):
    for k, target in [("row",row),("y",y),("fold",fold)]:
        np.testing.assert_array_equal(cache[k],target)


def run():
    rows=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    y=rows.target.to_numpy(bool)
    row=rows.row.to_numpy()
    stubborn=rows.v131_wrong.to_numpy(bool)&rows.number_new_correct.eq(0).to_numpy()
    base_hits, all_hits=[], []
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            base=ref["base"]
            fold=ref["fold"]
            verify(ref,row,y,fold)
            for k in ["base","fixed","learned_twice"]:
                stubborn &= (ref[k]>=.5)!=y
        base_hits.append((base>=.5)==y)
        predictions={}
        for folder,method in EXPERIMENTS.items():
            with np.load(ART/f"nov23_{folder}/{seed}.npz") as z:
                verify(z,row,y,fold)
                if "base" in z:
                    np.testing.assert_allclose(z["base"],base)
                candidate=z[method]
                blend=.75*base+.25*candidate
                if method+"_blend" in z:
                    np.testing.assert_allclose(z[method+"_blend"],blend)
                predictions[method]=candidate
                predictions[method+"_blend"]=blend
        with np.load(ART/f"nov23_group_disputes/{seed}.npz") as z:
            verify(z,row,y,fold)
            np.testing.assert_array_equal(z["anchor"],base)
            for k in ["two_level_table","shallow_tree"]:
                predictions[k]=z[k]
        matrix=np.column_stack(list(predictions.values()))
        assert np.isfinite(matrix).all() and np.all((0<=matrix)&(matrix<=1))
        all_hits.append((matrix>=.5)==y[:,None])
    hits=np.stack(all_hits)
    baseline=np.stack(base_hits)
    triple_wrong=~baseline.any(0)
    triple_right=baseline.all(0)
    assert stubborn.sum()==447 and np.all(triple_wrong[stubborn])
    summary={}
    for name,sel in [("stubborn447",stubborn),("all_three_baselines_wrong",triple_wrong)]:
        h=hits[:,sel,:]
        summary[name]=dict(size=int(sel.sum()),
            any_method_any_seed=int(h.any(axis=(0,2)).sum()),
            every_seed_some_method=int(h.any(2).all(0).sum()),
            one_method_all_seeds=int(h.all(0).any(1).sum()),
            never_correct=int((~h.any(axis=(0,2))).sum()))
    summary["always_baseline_correct"]=int(triple_right.sum())
    summary["fixed_methods"]={}
    for i,key in enumerate(predictions):
        h=hits[:,:,i]
        summary["fixed_methods"][key]=dict(
            all3_rescue_stubborn=int((stubborn&h.all(0)).sum()),
            all3_rescue_baseline=int((triple_wrong&h.all(0)).sum()),
            all3_harm_baseline=int((triple_right&~h.any(0)).sum()),
            accuracies=h.mean(1).tolist())
    write(ROOT/"reports/nov23_five_way_coverage/result.json",dict(
        assignments=SEEDS,candidates=list(predictions),
        exclusions=["raw_peers","single_shapes","basic","placebo","independent","permuted_tree","soft_mean",
                    "their associated blends"],
        interpretation="A rowwise best across multiple methods is a diagnostic ceiling. The same passenger and true label recur in all three assignments.",results=summary))
    print(summary)


if __name__=="__main__":
    run()
