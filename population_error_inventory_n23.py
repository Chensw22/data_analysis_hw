"""Count unique rescued and unresolved incumbent errors across the new outcome models."""
from pathlib import Path

import numpy as np
import pandas as pd

from error_overlap_diagnostic import load_authenticated
from global_error_fusion_n23 import arrays
from relational_error_corrector import write

ROOT = Path(__file__).resolve().parent
ART = ROOT/"artifacts"


def run():
    inp,_,fold,old,_=load_authenticated()
    dev,y=inp["dev"],inp["y"].astype(bool)
    anchor=pd.read_csv(ART/"nov21_case_checks/v136_final_oof.csv")
    for key,v in [("row",dev),("fold",fold),("true",y)]:
        np.testing.assert_array_equal(anchor[key],v)
    original=anchor.v131.to_numpy(bool)
    proposals={}
    for file,keys in [
        ("global_quantile_profiles_n23/oof.csv",{"raw_tree":"p_raw","quantile_tree":"p"}),
        ("nov23_honest_neighbors/oof.npz",{"forest":"p_forest","honest_leaves":"p"}),
        ("nov23_manifest_denoiser/oof.npz",{"network":"supervised","denoiser":"masked"}),
        ("global_tensor_nov23.npz",{"marginal":"marginal","tensor":"p"}),
        ("nov23_joint_booking/oof.npz",{"independent_pair":"independent","joint_pair":"joint"}),
        ("reverse_manifest_likelihood_n23/oof.csv",{"prior":"prior","reverse":"reverse","direct":"direct","bidirectional":"bidirectional"}),
        ("nov23_booking_mixture/random_purged.npz",{"shared_booking":"shared","individual_booking":"independent"})]:
        proposals.update(arrays(ART/file,dev,fold,y,keys))
    new=np.column_stack(list(proposals.values()))>=.5
    hits=(new==y[:,None])
    wrong=original!=y
    older=np.column_stack(list(old.values()))>=.5
    groups={"all":np.ones(len(y),bool),"false_negatives":y,"false_positives":~y,
            "historical_546":anchor.hard.to_numpy(bool),"persistent_510":anchor.persistent.to_numpy(bool)}
    result={}
    for label,scope in groups.items():
        target=scope&wrong
        covered=target&hits.any(1)
        result[label]=dict(v131_wrong=int(target.sum()),new_can_repair=int(covered.sum()),
                           still_all_new_wrong=int((target&~hits.any(1)).sum()),
                           all_seven_historical_also_wrong_but_new_can_repair=int((covered&(older!=y[:,None]).all(1)).sum()),
                           still_wrong_with_all_new_and_historical=int((target&~hits.any(1)&(older!=y[:,None]).all(1)).sum()))
    ledger=ART/"nov23_global_error_inventory.csv"
    pd.DataFrame({"row":dev,"fold":fold,"target":y,"v131":original,"v131_wrong":wrong,
                  "number_new_correct":hits.sum(1),"historical_546":anchor.hard,**proposals}).to_csv(ledger,index=False)
    write(ROOT/"reports/global_error_models_nov23/coverage.json",dict(models=list(proposals),rows=len(y),counts=result,
        definition="Sixteen predictions from the broad model experiments and their controls. Diagnostic coverage uses observed labels for evaluation only."))
    print(result)


if __name__=="__main__":
    run()
