"""Count corroboration from four structurally different classifiers on every proposed flip."""
import numpy as np
import pandas as pd

from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART, ROOT, SEEDS, read_panel
from relational_error_corrector import write


FAMILIES = {"class_analogues":"class_linear", "spend_copula":"conditional_copula",
            "masked_views":"structured_full", "spend_exceptions":"two_experts"}


def run():
    row=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    y,old,earlier=read_panel()
    hard=~old.any(0)&~earlier.any(axis=(0,1))
    results={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_teacher_field_offsets/{seed}.npz") as z:
            base,fold,score=z["base"],z["fold"],z["offset3"]
            verify(z,row,y,fold)
        proposed=(score>=.5)!=(base>=.5)
        correct=(score>=.5)==y
        votes=np.zeros(len(y),int)
        for name,method in FAMILIES.items():
            with np.load(ART/f"nov23_{name}/{seed}.npz") as z:
                verify(z,row,y,fold)
                np.testing.assert_allclose(z["base"],base)
                votes += (z[method]>=.5)==(score>=.5)
        groups={}
        for n in range(5):
            mask=proposed&(votes==n)
            groups[n]=dict(proposals=int(mask.sum()),fixed=int((mask&correct).sum()),
                           harmed=int((mask&~correct).sum()),fixed_of466=int((mask&hard&correct).sum()))
        assert sum(v["proposals"] for v in groups.values())==proposed.sum()
        # A strict majority of corroborating families, plus the original residual prediction.
        switch=proposed&(votes>=3)
        hit=np.where(switch,score,base)>=.5
        results[seed]=dict(by_corroborating_families=groups,strict_majority_accuracy=float((hit==y).mean()),
                           fixed=int((correct&switch).sum()),harmed=int((~correct&switch).sum()),
                           fixed_of466=int((correct&switch&hard).sum()))
    write(ROOT/"reports/nov23_model_corroboration/result.json",dict(families=FAMILIES,results=results))
    print(results)


if __name__=="__main__":
    run()
