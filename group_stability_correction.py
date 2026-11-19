"""Cross-party-deleted challengers with authentic or matched impostor companions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

import relational_error_corrector as ref
from stable_consensus_selector import persistent_mask
from peer_bill_correction import BILLS

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
OUT = ROOT / "reports/group_stability_correction"
SEEDS = [20261112, 20261115, 20261116]


def companion_design(dev, passengers):
    cols = ["PassengerId", "Cabin", "HomePlanet", "CryoSleep", "Age", "Destination", "VIP", *BILLS]
    roster = pd.concat([pd.read_csv(ROOT / "data/raw" / f, usecols=cols) for f in ("train.csv", "test.csv")], ignore_index=True)
    np.testing.assert_array_equal(roster.PassengerId.iloc[dev], passengers)
    group = roster.PassengerId.str.split("_").str[0]
    cab = roster.Cabin.str.split("/", expand=True)
    raw = pd.DataFrame({"age":roster.Age, "room":pd.to_numeric(cab[1]), "size":group.map(group.value_counts())})
    raw = pd.concat([raw, np.log1p(roster[BILLS]), pd.get_dummies(pd.concat(
        [roster[["HomePlanet","CryoSleep","Destination","VIP"]].astype("string"), cab[[0,2]].astype("string")],axis=1),dummy_na=True)], axis=1)
    nums = np.column_stack([roster.Age.to_numpy(float), roster.CryoSleep.astype(float), np.log1p(roster[BILLS])])
    rooms = roster.Cabin.fillna("").to_numpy()
    dest = roster.Destination.fillna("").to_numpy()
    closest = np.full(len(roster), -1)
    # Same cabin, then smallest age gap, then manifest order for ties.
    for rows in roster.groupby(group).indices.values():
        if len(rows) < 2:
            continue
        for member in rows:
            peers = rows[rows != member]
            same = (rooms[peers] == rooms[member]) & (rooms[member] != "")
            distance = np.nan_to_num(np.abs(nums[peers,0] - nums[member,0]), nan=100)
            closest[member] = peers[np.lexsort((peers,distance,~same))[0]]
    multi = closest >= 0
    matched = closest.copy()
    strata = pd.DataFrame(dict(planet=roster.HomePlanet.fillna("?"), deck=cab[0].fillna("?"),
                               size=raw["size"].clip(upper=4), age=roster.Age.fillna(-1)//15))
    rng = np.random.default_rng(20261119)
    # Exchange actual companion records between similar index passengers, excluding own party.
    # Fall back to the complete multi-party-index pool in sparsely populated strata.
    ids = np.flatnonzero(multi)
    groups = group.to_numpy()
    for rows in strata.iloc[ids].groupby(list(strata),sort=True).indices.values():
        pool = closest[ids[rows]]
        for row in ids[rows]:
            candidates = pool[groups[pool] != groups[row]]
            if len(candidates) == 0:
                fallback = closest[ids]
                candidates = fallback[groups[fallback] != groups[row]]
            matched[row] = rng.choice(candidates)
    x = raw.to_numpy(float)
    designs = {}
    for name, target in [("true",closest), ("impostor",matched)]:
        extra = np.full((len(roster), 2*nums.shape[1]+2), np.nan)
        other = target[multi]
        extra[multi] = np.column_stack([nums[other], nums[multi]-nums[other],
            (rooms[multi] == rooms[other]) & (rooms[multi] != ""),
            (dest[multi] == dest[other]) & (dest[multi] != "")])
        designs[name] = np.column_stack([x, extra])[dev]
    designs["no_companion"] = x[dev]
    return designs, multi[dev]


def summarize(pred, base, y, fold, hard, persistent):
    repair = (pred==y) & (base!=y)
    harm = (pred!=y) & (base==y)
    return dict(accuracy=float(np.mean(pred==y)), repairs=int(repair.sum()), harms=int(harm.sum()),
                fold_net=[int(repair[fold==i].sum())-int(harm[fold==i].sum()) for i in range(3)],
                hard=[int((repair&hard).sum()),int((harm&hard).sum())],
                persistent=[int((repair&persistent).sum()),int((harm&persistent).sum())])


def run():
    inp, _, _, models, _ = ref.overlap.load_authenticated()
    dev, y, groups = inp["dev"], inp["y"].astype(bool), np.asarray(inp["groups"])
    designs, eligible = companion_design(dev, inp["passengers"])
    result = {}
    cache = dict(dev=dev, eligible=eligible)
    saved = []
    for seed in SEEDS:
        summary = json.loads((ROOT/"reports/top5_group_phase/refs/mitra/summary.json").read_text())
        source = next(r for r in summary["results"] if r["seed"]==seed)
        path = ROOT/source["oof_path"]
        assert ref.sha(path)==source["oof_sha256"]
        with np.load(path) as z:
            np.testing.assert_array_equal(z["dev_ids"],dev)
            np.testing.assert_array_equal(z["group_ids"],groups)
            fold, anchor = z["fold_id"], z["prob"]
        outs = {key:np.full((len(y),3),np.nan) for key in designs}
        for j in range(3):
            a,b = np.flatnonzero(fold!=j),np.flatnonzero(fold==j)
            assert not set(groups[a]).intersection(groups[b])
            unique = np.unique(groups[a])
            order = np.random.default_rng(seed+j).permutation(unique)
            for m,omit in enumerate(np.array_split(order,3)):
                support = a[~np.isin(groups[a],omit)]
                for key,x in designs.items():
                    model=HistGradientBoostingClassifier(max_iter=110,max_leaf_nodes=15,learning_rate=.075,
                        min_samples_leaf=20,l2_regularization=3,early_stopping=False,random_state=seed+j)
                    model.fit(x[support],y[support])
                    outs[key][b,m]=model.predict_proba(x[b])[:,1]
        saved.append((seed,fold,anchor,outs))
        print("Fitted", seed, flush=True)
    catalog=pd.read_csv(ART/"error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    persistent=persistent_mask(inp,models,hard)
    for seed,fold,anchor,outs in saved:
        base=anchor>=.5
        result[seed]={"baseline":summarize(base,base,y,fold,hard,persistent)}
        for key,probs in outs.items():
            assert np.isfinite(probs).all()
            sign=np.where(base,-1.,1.)
            margins=(probs-.5)*sign[:,None]
            # For every supporter deletion, rival must oppose the base more strongly than its own margin.
            for mode,signal in [("stable",margins.min(1)),("mean",margins.mean(1))]:
                change=eligible & (signal>np.abs(anchor-.5))
                pred=base^change
                name=key+"_"+mode
                result[seed][name]=summarize(pred,base,y,fold,hard,persistent)
                cache[str(seed)+"_"+name]=pred
    ref.write(OUT/"result.json",dict(reference="Mitra zero-shot, three group-disjoint split seeds, comparator specific to each seed; same development passengers", seeds=result))
    np.savez_compressed(ART/"group_stability_correction.npz", **cache)
    print(result)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
