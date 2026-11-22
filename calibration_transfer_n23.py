"""Check the same conditional recalibration on a separate fixed sparse-rule ensemble."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from conditional_calibration_n23 import ART, ROOT, SEEDS, BILLS, attributes, corrected
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write


def run():
    inventory = pd.read_csv(ART/"nov23_global_error_inventory.csv")
    row = inventory.row.to_numpy()
    y, anchor_hits, challengers = read_panel()
    hard = ~anchor_hits.any(0) & ~challengers.any(axis=(0,1))
    raw = pd.read_csv(ROOT/"data/raw/train.csv", usecols=["PassengerId","Cabin",*BILLS]).iloc[row]
    group = raw.PassengerId.str.split("_").str[0].to_numpy()
    _, idx = np.unique(group, return_inverse=True)
    sizes = np.bincount(idx)
    context = attributes(raw)
    results, differences = {}, []
    out = ART/"nov23_calibration_transfer"
    out.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_sparse_rules/{seed}.npz") as z:
            fold, base, rule = z["fold"], z["base"], z["rules"]
            verify(z,row,y,fold)
        anchor = .75*base+.25*rule
        original = (base>=.5)==y
        before = (anchor>=.5)==y
        candidates = corrected(anchor, fold, group, context, y)
        metrics = {}
        for name, prob in candidates.items():
            hit = (prob>=.5)==y
            assert np.isfinite(prob).all()
            metrics[name] = dict(accuracy=float(hit.mean()), fixed=int((hit&~original).sum()),
                                 harmed=int((~hit&original).sum()), versus_rules=int(hit.sum()-before.sum()),
                                 unresolved466=int(hit[hard].sum()))
        joint = ((candidates["context"]>=.5)==y).astype(float)
        differences.append(np.column_stack([joint-original, joint-before,
                                             joint-((candidates["placebo"]>=.5)==y)]))
        results[seed] = metrics
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,anchor=anchor,**candidates)
    # Repeated splits share labels. Resample each whole booking with all its fold results together.
    delta = np.mean(differences, axis=0)
    totals = np.zeros((len(sizes),delta.shape[1]))
    np.add.at(totals, idx, delta)
    rng = np.random.default_rng(743)
    draws = []
    for _ in range(80):
        freq = rng.multinomial(len(sizes),np.full(len(sizes),1/len(sizes)),size=50)
        draws.append(100*(freq@totals)/(freq@sizes)[:,None])
    gains = 100*delta.mean(axis=0)
    ci = np.quantile(np.concatenate(draws),[.025,.975],axis=0)
    comparisons = {name:dict(mean_gain_pp=float(gains[i]),ci95_pp=ci[:,i].tolist())
                   for i,name in enumerate(["baseline","rule_ensemble","shuffled_context"])}
    write(ROOT/"reports/nov23_calibration_transfer/result.json",dict(
        results=results, comparisons=comparisons, bookings=len(sizes),
        interpretation="Descriptive paired uncertainty conditional on the fixed fitted development predictions."))
    print(results, comparisons, flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
