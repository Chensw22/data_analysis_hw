"""Test whether a zero-spend sleeper's party composition identifies false positives."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exact_fold_error_selector import ART, ROOT, cached, counts
from peer_bill_correction import BILLS
from replicate_error_committee import partitions
from route_consensus_errors import FAMILIES
from stable_consensus_selector import persistent_mask
import relational_error_corrector as ref


def roster_design():
    cols = ["PassengerId", "Cabin", "Age", "HomePlanet", "CryoSleep", *BILLS]
    roster = pd.concat([pd.read_csv(ROOT/"data/raw"/f, usecols=cols)
                        for f in ("train.csv", "test.csv")], ignore_index=True)
    party = roster.PassengerId.str.split("_").str[0].to_numpy()
    cab = roster.Cabin.str.split("/", expand=True)
    cabin = roster.Cabin.fillna("").to_numpy()
    side = cab[2].to_numpy()
    number = pd.to_numeric(cab[1]).fillna(-1).to_numpy()
    money = roster[BILLS].to_numpy(float)
    # Unmeasured bills stay distinguishable from reported zeros.
    sleep = roster.CryoSleep.astype(float).to_numpy()
    eligible = ((roster.HomePlanet == "Earth") & (cab[0] == "G") & (sleep == 1)
                & np.all(money == 0, axis=1)).to_numpy()
    size = pd.Series(party).map(pd.Series(party).value_counts()).to_numpy()
    detail = np.zeros((len(roster), 11))
    for group in roster.groupby(party).indices.values():
        for i in group[eligible[group] & (size[group] > 1)]:
            peers = group[group != i]
            same = (cabin[peers] == cabin[i])
            spending = money[peers]
            # These measurements describe shared rooms, awake companions and two
            # service families. Exclude the passenger's own values throughout.
            paid = (spending > 0).any(axis=1)
            protective = np.log1p(np.nansum(spending[:, [1, 2]], axis=1))
            leisure = np.log1p(np.nansum(spending[:, [0, 3, 4]], axis=1))
            detail[i] = [same.mean(), (sleep[peers] == 0).mean(), np.isnan(sleep[peers]).mean(),
                         paid.mean(), (paid & same).mean(), (paid & ~same).mean(),
                         protective.mean(), leisure.mean(), (protective * same).mean(),
                         (leisure * same).mean(), np.isnan(spending).mean()]
    actual = detail.copy()
    faux = detail.copy()
    # Negative control exchanges the whole set of peer measurements between
    # comparable sleepers with similar cabin positions and equal parties' sizes.
    rng = np.random.default_rng(20261121)
    for i in np.flatnonzero(eligible & (size > 1)):
        pool = np.flatnonzero(eligible & (size == size[i]) & (side == side[i]) & (party != party[i]))
        if len(pool) == 0:
            pool = np.flatnonzero(eligible & (size > 1) & (party != party[i]))
        nearby = pool[np.argsort(np.abs(number[pool]-number[i]), kind="stable")[:16]]
        faux[i] = detail[rng.choice(nearby)]
    return roster, party, size, eligible, actual, faux


def run():
    inp, _, fold, bases, _ = ref.overlap.load_authenticated()
    dev, y = inp["dev"], inp["y"].astype(bool)
    p = cached("v108_exact_oof/oof.npz", "v108", dev, fold)
    base = p >= .5
    hist = np.column_stack([bases[k] for k in FAMILIES])
    roster, party, size, eligible, actual, faux = roster_design()
    np.testing.assert_array_equal(roster.PassengerId.iloc[dev], inp["passengers"])
    groups = party[dev]
    cohort = eligible[dev] & base & (hist >= .5).all(axis=1)
    active = cohort & (size[dev] > 1)
    simple = np.column_stack([ref.logit(p), ref.logit(hist).mean(axis=1), np.log(size[dev])])
    designs = dict(no_peers=simple, actual=np.column_stack([simple, actual[dev]]),
                   matched=np.column_stack([simple, faux[dev]]))
    cat = pd.read_csv(ART/"error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(cat.row_id, dev)
    hard = cat.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp, bases, hard)
    results, cache = {}, {}
    for mode in ("original", "new_party_partitions", "boarding_blocks"):
        risks = {name:np.zeros((len(p), 3)) for name in designs}
        for outer in range(3):
            ids = np.flatnonzero(active & (fold == outer))
            coverage = np.zeros_like(risks["actual"], dtype=int)
            for slot, _, t, q in partitions(ids, groups, mode, outer):
                assert not np.intersect1d(groups[t], groups[q]).size
                coverage[q,slot] += 1
                for name,x in designs.items():
                    fit = make_pipeline(StandardScaler(), LogisticRegression(C=.05,max_iter=500))
                    fit.fit(x[t], ~y[t])
                    risks[name][q,slot] = fit.predict_proba(x[q])[:,1]
            assert (coverage[ids] == 1).all()
        for name, risk in risks.items():
            pred = base ^ (risk.mean(axis=1) >= .5)
            key = mode + "_" + name
            results[key] = counts(pred, base, y, fold, hard, persistent)
            cache[key] = pred
    for label,mask in [("whole_cohort",cohort),("multi",active),("solo",cohort & ~active)]:
        results[label] = dict(rows=int(mask.sum()), errors=int((mask & ~y).sum()),
                              hard=int((hard & mask & ~y).sum()), persistent=int((persistent & mask & ~y).sum()))
    ref.write(ROOT/"reports/sleeper_peer_evidence/result.json", dict(results=results,
        design="Adaptive subgroup identified from previous OOF inspection. Exact v108 positive and all five references positive, "
        "Earth/G/asleep/five observed zero bills. Only passengers with companions are changed. Per outer held-out "
        "fold, fit a ridge logistic error model on other held-out groups. Three partition modes, cutoff 0.5. "
        "Average all-companion attributes compared with matched impostor peer sets and a size and confidence control."))
    np.savez_compressed(ART/"sleeper_peer_evidence.npz",dev=dev,fold=fold,base=base,**cache)
    for k,v in results.items():
        print(k, v)


if __name__ == "__main__":
    run()
