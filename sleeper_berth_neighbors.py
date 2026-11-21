"""Estimate Earth/G sleeper outcomes from other parties in the same local berth range."""
import numpy as np
import pandas as pd

from passenger_neighbor_checks import ART, ROOT, OUT, INCUMBENT, vote, ref, routes, counts, compact_features, confirm
from sleeper_consensus_scope import cohort
from stable_consensus_selector import persistent_mask
from sklearn.model_selection import GroupKFold


def local_views(x):
    cabin = x.CabinNumber.fillna(850) / 100
    age = x.Age.fillna(28) / 12
    port = x.Side.eq("P")
    unknown_side = x.Side.eq("__MISSING__")
    trappist = x.Destination.eq("TRAPPIST-1e")
    return [np.column_stack([cabin, 4 * port.astype(int), 2 * unknown_side.astype(int)]),
            np.column_stack([cabin * .45, age, x.Age.lt(13), port, trappist])]


def estimate(matrices, groups, target_rows, target_y, query_rows):
    small = vote(matrices[0], groups, target_rows, query_rows, target_y, 35)
    broad = vote(matrices[1], groups, target_rows, query_rows, target_y, 75)
    return small, broad


def apply(anchor, eligible, small, broad):
    return {"v135d": anchor & ~(eligible & ((small + broad) / 2 < .52)),
            "v135c": anchor & ~(eligible & (small < .5) & (broad < .5)),
            "v136c": anchor & ~(eligible & ((small + broad) / 2 < .5))}


def run():
    inp, _, fold, models, _ = ref.overlap.load_authenticated()
    dev, y = inp["dev"], inp["y"].astype(bool)
    train = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c not in ["Name", "Transported"])
    test = pd.read_csv(ROOT / "data/raw/test.csv", usecols=lambda c: c != "Name")
    roster = pd.concat([train,test], ignore_index=True)
    x = compact_features(roster)
    groups = roster.PassengerId.str.split("_").str[0].to_numpy()
    source = cohort(x) & x.CabinNumber.notna().to_numpy()
    matrices = local_views(x)
    small, broad = np.ones(len(dev)), np.ones(len(dev))
    for part in np.unique(fold):
        t = (fold != part) & source[dev]
        q = (fold == part) & source[dev]
        small[q], broad[q] = estimate(matrices, groups, dev[t], y[t], dev[q])
    with np.load(ART / "sleeper_consensus_scope.npz") as scoped, np.load(ART / "sleeper_manifest_risk.npz") as witness:
        for z in (scoped,witness):
            np.testing.assert_array_equal(z["dev"],dev)
            np.testing.assert_array_equal(z["fold"],fold)
        anchor = confirm(scoped["base"],scoped["original"],witness["manifest_berth_party_a"])
        eligible = scoped["eligible"] & source[dev]
    decisions = apply(anchor,eligible,small,broad)
    rows = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(rows.row_id, dev)
    hard = rows.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp, models, hard)
    summary = {name:counts(p,anchor,y,fold,hard,persistent) for name,p in decisions.items()}
    other_folds = np.zeros(len(y),int)
    for k, (_, q) in enumerate(GroupKFold(3).split(dev,y,groups[dev])):
        other_folds[q] = k
    boarding = np.searchsorted(np.quantile(groups[dev].astype(int), [1/3,2/3]),groups[dev].astype(int))
    repeats = {}
    for partition, alt_fold in [("whole_parties", other_folds), ("boarding_blocks", boarding)]:
        sx,bx = np.ones(len(dev)),np.ones(len(dev))
        for part in range(3):
            t = (alt_fold != part) & source[dev]
            q = (alt_fold == part) & source[dev]
            sx[q],bx[q]=estimate(matrices,groups,dev[t],y[t],dev[q])
        repeats[partition] = {n:counts(p,anchor,y,alt_fold,hard,persistent)
                              for n,p in apply(anchor,eligible,sx,bx).items()}
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(dict(PassengerId=train.PassengerId.iloc[dev].to_numpy(), fold=fold, target=y, incumbent=anchor,
                      hard=hard, persistent=persistent, eligible=eligible, small=small, broad=broad,
                      **decisions)).to_csv(OUT/"berth_local_rows.csv", index=False)
    labels = pd.read_csv(ROOT/"data/raw/train.csv", usecols=["Transported"]).Transported.to_numpy(bool)
    test_idx=np.arange(len(train),len(roster))
    train_idx=np.flatnonzero(source[:len(train)])
    q=source[test_idx]
    small,broad=np.ones(len(test)),np.ones(len(test))
    small[q],broad[q]=estimate(matrices,groups,train_idx,labels[train_idx],test_idx[q])
    ids=test.PassengerId.to_numpy(str)
    anchor=routes.official_decisions(INCUMBENT,ids)[0]
    eligible=q & routes.official_decisions(routes.BASE,ids)[0]
    for family in routes.HISTORICAL:
        eligible &= routes.official_decisions(family,ids)[0]
    for name,p in apply(anchor,eligible,small,broad).items():
        pd.DataFrame(dict(PassengerId=ids, Transported=p)).to_csv(OUT/(name+".csv"), index=False)
        summary[name]["test_changes_vs_v128"]=int((p!=anchor).sum())
    ref.write(ROOT/"reports/nov21_case_checks/berth.json",dict(results=summary, alternate_donor_partitions=repeats, description=
        "Only observed Earth/G zero-bill cryosleep with available cabin number and unanimous v108 and five historical positive decisions. "
        "Predict from all other-fold parties inside this cohort, k35 by cabin and side; k75 by cabin, age, side and destination. "
        "v135c both probabilities below half; v136c mean below half. v135d mean below .52, chosen following a development bin inspection. Same labels reused on alternate partitions, v128 fixed. Cohort derived from previous error analysis."))
    print(summary)


if __name__=="__main__":
    run()
