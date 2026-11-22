"""Quantify the accuracy that repeating the baseline partitions can and cannot attain on development."""
import numpy as np
import pandas as pd

from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART, ROOT, SEEDS, read_panel
from relational_error_corrector import write


def run():
    rows = pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    y, historical, others = read_panel()
    repeated = {name:[] for name in ["base","fixed_quarter","joint"]}
    bookings = pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.str.split("_").str[0].iloc[rows]
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as anchor, np.load(ART/f"nov23_joint_risk_surface/{seed}.npz") as z:
            fold=anchor["fold"]
            verify(anchor,rows,y,fold)
            verify(z,rows,y,fold)
            np.testing.assert_array_equal(z["base"],anchor["base"])
            assert pd.DataFrame(dict(g=bookings.to_numpy(),fold=fold)).groupby("g").fold.nunique().max()==1
            for name in repeated:
                repeated[name].append(z[name])
    stable_wrong=~historical.any(0)
    strict_target=int(np.floor(.85*len(y)))+1
    result={}
    for name,probs in repeated.items():
        scores=np.asarray(probs)
        pred=scores.mean(0)>=.5
        correct=pred==y
        some=((scores>=.5)==y).any(0)
        # A convex combination cannot cross the decision boundary if every input
        # is strictly on the same side, including when the combination varies by person.
        assert not (correct&~some).any()
        result[name]=dict(equal_average_correct=int(correct.sum()),accuracy=float(correct.mean()),
                          actual_more_correct_needed_for_85=strict_target-int(correct.sum()),
                          number_correct_if_always_selecting_a_correct_member=int(some.sum()),
                          optimistic_upper_accuracy=float(some.mean()),wrong_under_every_member=int((~some).sum()),
                          wrong_under_every_member_true_positive_class=int((~some&y).sum()),
                          original_1070_correct=int((correct&stable_wrong).sum()),
                          original_480_persistent_false_negatives_correct=int((correct&stable_wrong&y).sum()),
                          original_590_persistent_false_positives_correct=int((correct&stable_wrong&~y).sum()))
    trusted=historical.any(0)
    np.testing.assert_array_equal(trusted,((np.asarray(repeated["base"])>=.5)==y).any(0))
    all_families=(trusted|others.any(axis=(0,1)))
    write(ROOT/"reports/nov23_split_ensemble_capacity/result.json",
          dict(rows=len(y),target_correct=strict_target,results=result,
               minimum_repairs_outside_baseline_members_at_85=strict_target-int(trusted.sum()),
               oracle_all_eight_families=int(all_families.sum()),
               caveat="Only fixed, known out-of-fold predictions on these same people are compared. The best-member bound permits looking at the label for diagnosis and is not a bound on new models or on Kaggle."))
    print(result,strict_target,flush=True)


if __name__=="__main__":
    run()
