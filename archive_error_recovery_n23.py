"""Count recoveries and collateral errors across archived booking-held-out experiments."""
import numpy as np
import pandas as pd

from convex_portfolio_bound_n23 import read_scores
from grouped_five_way_coverage_n23 import verify
from population_error_accounting_n23 import ADDITIONS, ART, FAMILIES, ROOT, SEEDS
from relational_error_corrector import write

# One declared prediction per additional archived experiment. Exclude placebos,
# uncertainty measures, always-positive predictors, and variants of the same trial.
EXTRA = {
    "across_deck_field": "cabin_direct", "apple_semantics": "semantic",
    "booking_mixup": "neighbor", "booking_scoring": "fit_marginal",
    "class_analogues": "class_linear", "conditional_view_density": "conditional",
    "exact_booking_field": "joint", "field_factors": "order3", "gibbs_manifest": "adaptive",
    "group_disputes": "two_level_table", "masked_views": "structured_marginal",
    "ordered_states": "cabin", "pair_pretext": "integrity", "party_hotdeck": "multiple",
    "passenger_sampler": "sequential", "registration_rank": "matched_registration",
    "residual_discovery": "residual_rules", "residual_registration": "unexpected",
    "rule_context_pool": "global_local", "selective_fields": "all_orders",
    "spend_copula": "conditional_copula", "spend_exceptions": "two_experts",
    "symbolic_residuals": "evolved", "teacher_field_offsets": "penalized_offset3",
    "training_cohort_sensitivity": "named_cohorts", "transported_subclasses": "teacher_difficulty",
}


def predictions(rows, labels, seed, methods):
    with np.load(ART / f"nov23_group_kernel/{seed}.npz") as ref:
        baseline, folds = ref["base"], ref["fold"]
        verify(ref, rows, labels, folds)
    scores = []
    for folder, key in methods.values():
        with np.load(ART / f"nov23_{folder}/{seed}.npz") as model:
            verify(model, rows, labels, folds)
            p = model[key]
            assert p.shape == labels.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(), (folder, key)
            scores.append(p)
    return baseline, np.column_stack(scores), folds


def panel():
    rows, y, means = read_scores()
    old = {**FAMILIES, "budget_prior": ("simplex_likelihood","budget_prior"), **ADDITIONS}
    extra = {name: (name,key) for name,key in EXTRA.items()}
    methods = {**old, **extra}
    assert len(methods) == len(old)+len(extra)
    probs, base = [], []
    for seed in SEEDS:
        b, p, _ = predictions(rows,y,seed,methods)
        base.append(b)
        probs.append(p)
    bank = np.stack(probs)
    hits = (bank>=.5) == y[None,:,None]
    agreed_wrong = ((means>=.5) != y[:,None]).all(axis=1)
    unresolved = agreed_wrong & ~hits[:,:,:len(old)].any(axis=(0,2))
    assert unresolved.sum() == 365
    return rows, y, np.asarray(base), bank, unresolved, list(methods), len(old)


def run():
    rows, y, baseline, p, hard, names, start = panel()
    base = (baseline>=.5) == y
    hits = (p>=.5) == y[None,:,None]
    stable = hits.all(axis=0)
    sporadic = hits.any(axis=0)
    spread = stable[:,start:].sum(axis=1)
    exams = {}
    for col, name in enumerate(names[start:],start):
        mask = hard & stable[:,col]
        exams[name] = dict(stable_recoveries=int(mask.sum()),positive=int((mask&y).sum()),
                           negative=int((mask&~y).sum()), exclusive=int((mask & (spread==1)).sum()),
                           newly_wrong_on_all_three=int((~sporadic[:,col] & base.all(axis=0)).sum()),
                           full_accuracy=hits[:,:,col].mean(axis=1).tolist())
    raw = pd.read_csv(ROOT/"data/raw/train.csv", usecols=["HomePlanet","CryoSleep","PassengerId"])
    groups = raw.PassengerId.str.split("_").str[0].to_numpy()[rows]
    sets = {"originally_unresolved":hard, "still_unresolved":hard & ~sporadic[:,start:].any(axis=1),
            "at_least_one_stable":hard & (spread>0), "at_least_one_hit":hard & sporadic[:,start:].any(axis=1)}
    buckets = {}
    for name,m in sets.items():
        buckets[name] = dict(rows=int(m.sum()), transported=int((m&y).sum()), not_transported=int((m&~y).sum()),
                             bookings=int(np.unique(groups[m]).size),earth=int((m&raw.HomePlanet.eq("Earth").to_numpy()[rows]).sum()))
    # Row masks are exclusively retrospective. Never expose these to a candidate as features or training labels.
    cache = ART/"nov23_archive_recovery"
    cache.mkdir(exist_ok=True)
    np.savez_compressed(cache/"masks.npz",row=rows,**sets)
    write(ROOT/"reports/nov23_archive_recovery/result.json",dict(rows=len(y),original_methods=names[:start],
          additional_experiments=exams,cohorts=buckets))
    print(buckets, sorted([(v["stable_recoveries"],k,v["newly_wrong_on_all_three"]) for k,v in exams.items()],reverse=True)[:8])


if __name__=="__main__":
    run()
