"""Measure complementary corrections, then learn their eligibility on other bookings."""
import numpy as np
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, historical, evaluate, write
from class_analogue_experts_n23 import design

REPORT = ROOT / "reports/nov23_broad_cross_family_rescue"
FAMILIES = ("boosting_path", "neural_path", "training_membership")


def read(name, row, y):
    bank = {}
    for family in FAMILIES:
        with np.load(ART / f"nov23_broad_{family}/{name}.npz") as z:
            for field, truth in (("row", row), ("y", y)):
                np.testing.assert_array_equal(z[field], truth)
            if not bank:
                base, ref, fold = z["base"], z["ref"], z["fold"]
            for key, value in (("base", base), ("ref", ref), ("fold", fold)):
                np.testing.assert_array_equal(z[key], value)
            if family == "boosting_path":
                bank[family] = expit(z["paths"].mean(2))
            elif family == "neural_path":
                bank[family] = z["trajectories"].mean(1)
            else:
                bank[family] = np.column_stack([z[k] for k in ("ordinary","favor_shared","favor_members")])
    return base, ref, fold, bank


def predictions(y, base, ref, fold, bank, frame, group, locked):
    probabilities = np.column_stack(list(bank.values()))
    # During incumbent replay only the original probability is used as a covariate.
    # Locked incumbent revisions are preserved when scoring the resulting policy.
    original = base >= .5
    alternatives = (probabilities >= .5) != original[:,None]
    eligible = alternatives.any(1)
    assert np.array_equal(ref[~locked], original[~locked])
    scores = {}
    for key in ("risk", "families", "manifest"):
        scores["update_"+key] = np.where(original,.75,.25)
    for f in np.unique(fold):
        ids = np.flatnonzero(fold == f)
        for a,b in GroupKFold(3,shuffle=True,random_state=749).split(ids, groups=group[ids]):
            t,q = (part[eligible[part] & ~locked[part]] for part in (ids[a],ids[b]))
            assert not np.intersect1d(group[t],group[q]).size
            summary = np.column_stack([base, *[(p>=.5).mean(1) for p in bank.values()]])
            _, manifest = design(frame,t)
            for key, matrix in (("risk",summary), ("families",np.c_[summary,probabilities]),
                                ("manifest",np.c_[summary, probabilities, manifest, frame.BookingSize])):
                # Fit directly to the quantity that matters when changing a binary decision.
                model = RandomForestClassifier(n_estimators=120,min_samples_leaf=24,
                          max_features=1.,n_jobs=2,random_state=698).fit(matrix[t], y[t]!=original[t])
                flip = model.predict_proba(matrix[q])[:,1] > .5
                scores["update_"+key][q] = np.where(original[q]^flip,.75,.25)
    return scores


def main():
    row,y,frame,group,_ = load()
    original,labels,masks = historical()
    np.testing.assert_array_equal(row,original)
    np.testing.assert_array_equal(y,labels)
    records, stacked, policies = {}, [], {}
    for name in [*map(str,SEEDS),"v131"]:
        base,ref,fold,bank = read(name,row,y)
        locked = ref!=(base>=.5)
        if name!="v131":
            assert not locked.any()
        scores = predictions(y,base,ref,fold,bank,frame,group,locked)
        result,hits = evaluate(y,base,ref,scores,fold,name=="v131",masks=masks)
        records[name] = result
        if name!="v131":
            stacked.append({k:(v>=.5)==y[:,None] for k,v in bank.items()})
            for key, (hit,_) in hits.items():
                policies.setdefault(key,[]).append(hit)
    # For each family one and the same checkpoint/recipe must succeed on the
    # same passenger across all three assignments to count them as covered.
    stable = np.column_stack([np.logical_and.reduce([s[k] for s in stacked]).any(1) for k in FAMILIES])
    scopes = {**masks, "all":np.ones(len(y),bool)}
    coverage = {}
    for key,mask in scopes.items():
        data = {}
        for label in (False,True):
            sel = mask & (y==label)
            data[str(label)] = dict(n=int(sel.sum()), any=int((sel&stable.any(1)).sum()),
                    none=int((sel&~stable.any(1)).sum()),
                    families={k:int((sel&stable[:,i]).sum()) for i,k in enumerate(FAMILIES)},
                    unique={k:int((sel&(stable.sum(1)==1)&stable[:,i]).sum()) for i,k in enumerate(FAMILIES)})
        coverage[key] = data
    learned = {k:{n:int((np.logical_and.reduce(v)&m).sum()) for n,m in scopes.items()} for k,v in policies.items()}
    # Partition all repeat errors using one broad everyday distinction. Include the
    # whole population denominator and the share left after all model families.
    bills = frame[["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]]
    observed_positive = bills.gt(0).any(axis=1).to_numpy()
    unknown = bills.isna().any(axis=1).to_numpy()
    burden = {}
    for name,mask in (("positive_bill",observed_positive),("all_five_zero",~observed_positive&~unknown),
                      ("no_positive_incomplete",~observed_positive&unknown)):
        repeat = masks["repeat"] & mask
        burden[name] = dict(passengers=int(mask.sum()),repeat=int(repeat.sum()),
                  repeat_actual_true=int((repeat&y).sum()),stable_alternative=int((repeat&stable.any(1)).sum()))
    assert sum(v["repeat"] for v in burden.values())==int(masks["repeat"].sum())
    write(REPORT/"results.json",dict(plans=records,stable_policy=learned,coverage=coverage,bill_population=burden))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
