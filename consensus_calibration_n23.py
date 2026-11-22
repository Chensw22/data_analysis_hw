"""Compare expected and observed error counts in cohorts defined before looking at labels."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART, FAMILIES, ROOT, SEEDS, read_panel
from relational_error_corrector import write


def measure(mask, positive, y, p, register, bookings):
    observed = (y if positive else ~y)[mask].astype(float)
    risk = (p if positive else 1-p)[mask]
    other = (register if positive else 1-register)[mask]
    _,g = np.unique(bookings[mask],return_inverse=True)
    size = np.bincount(g)
    delta = np.bincount(g,weights=observed-risk)
    rng=np.random.default_rng(528)
    diff=[]
    for _ in range(80):
        draws=rng.integers(0,len(size),size=(25,len(size)))
        diff.extend(100*delta[draws].sum(1)/size[draws].sum(1))
    def auc(score):
        return float(roc_auc_score(observed,score)) if len(np.unique(observed))==2 else None
    return dict(rows=int(mask.sum()),bookings=len(size),observed=int(observed.sum()),expected=float(risk.sum()),
                observed_rate=float(observed.mean()),mean_risk=float(risk.mean()),
                actual_minus_expected_95_pp=np.quantile(diff,[.025,.975]).tolist(),
                base_within_group_auc=auc(risk),registration_within_group_auc=auc(other))


def run():
    row=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    y,_,_=read_panel()
    group=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.str.split("_").str[0].to_numpy()[row]
    p,register,votes=[],[],[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold=z["fold"]
            verify(z,row,y,fold)
            p.append(z["base"])
        with np.load(ART/f"nov23_bill_matched_register/{seed}.npz") as z:
            verify(z,row,y,fold)
            register.append(z["register"])
        members=[]
        for experiment,key in FAMILIES.values():
            with np.load(ART/f"nov23_{experiment}/{seed}.npz") as z:
                verify(z,row,y,fold)
                members.append(z[key]>=.5)
        votes.append(members)
    p=np.asarray(p)
    avg=p.mean(0)
    reg=np.mean(register,axis=0)
    votes=np.asarray(votes)
    result={}
    for positive,side in [(True,"missed_transports"),(False,"false_alarms")]:
        score=p if positive else 1-p
        directions=(p<.5) if positive else (p>=.5)
        other_agrees=(~votes if positive else votes).all(axis=(0,1))
        same=directions.all(0)
        high=same & (score.max(0)<=.2)
        scopes={"agreement_across_baseline_fits":same,"baseline_confidence_at_least_80":high,
                "all_eight_families_agree":same&other_agrees,
                "all_eight_agree_and_baseline_confidence_at_least_80":high&other_agrees}
        result[side]={name:measure(mask,positive,y,avg,reg,group) for name,mask in scopes.items()}
    write(ROOT/"reports/nov23_consensus_calibration/result.json",dict(results=result,
        interpretation="Cohorts use predictions only. Expected is the summed probability assigned to the contrary outcome. AUC evaluates ranking rather than selection at threshold 0.5. Intervals resample whole bookings with fixed predictors."))
    print(result,flush=True)


if __name__=="__main__":
    run()
