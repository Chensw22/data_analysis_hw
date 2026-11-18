"""Test short repeating cabin positions against a local position-shuffling control."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

import relational_error_corrector as ref
from peer_bill_correction import load_v96

ROOT = Path(__file__).resolve().parent
PERIODS = [2, 3, 4, 6, 8, 12, 16, 32]


def design(frame, shuffled):
    cabin = frame.Cabin.str.split("/", expand=True)
    cabin.columns = ["deck", "number", "side"]
    number = pd.to_numeric(cabin.number).to_numpy(float)
    if shuffled:
        # Permute unique cabins, preserving co-occupancy, sides and broad position.
        unique = cabin.dropna().drop_duplicates().copy()
        unique["block"] = pd.to_numeric(unique.number) // 32
        unique = unique.sort_values(["deck", "side", "block", "number"])
        rng = np.random.default_rng(20261118)
        unique["replacement"] = unique.groupby(["deck", "side", "block"]).number.transform(rng.permutation)
        index = unique.set_index(["deck", "number", "side"]).replacement
        number = pd.to_numeric(index.reindex(pd.MultiIndex.from_frame(cabin))).to_numpy(float)
    defined = np.isfinite(number)
    phase = np.nan_to_num(number)
    signals = []
    for period in PERIODS:
        angle = 2*np.pi*phase/period
        signals.append(np.cos(angle))
        if period != 2:
            signals.append(np.sin(angle))
    harmonics = np.column_stack(signals)
    strata = pd.get_dummies(cabin.deck.fillna("missing"), dtype=float).to_numpy()
    controls = np.column_stack([np.ones(len(frame)), cabin.side.eq("S"), frame.CryoSleep.eq(True), strata])
    x = (harmonics[:, :, None] * controls[:, None, :]).reshape(len(frame), -1)
    x[~defined] = 0
    return x, defined


def run():
    inp, splits, folds, bases, evidence = ref.overlap.load_authenticated()
    dev, y, raw = inp["dev"], inp["y"], bases["rawMitra"]
    frame = pd.concat([pd.read_csv(ROOT / "data/raw" / f, usecols=["Cabin","CryoSleep","PassengerId"])
                       for f in ["train.csv", "test.csv"]], ignore_index=True)
    np.testing.assert_array_equal(frame.PassengerId.iloc[dev], inp["passengers"])
    catalog = pd.read_csv(ROOT / "artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(dev,catalog.row_id)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    result = {}
    for mode in ["actual", "shuffled"]:
        x, defined = design(frame, mode == "shuffled")
        shift = np.zeros(len(dev))
        for fold, (a,b) in enumerate(splits):
            inner = ref.load_inner(fold,a,b,dev,y,evidence)
            train = defined[dev[a]]
            q = b[defined[dev[b]]]
            weights,_ = ref.fit_corrector(inner[train],y[a][train],x[dev[a[train]]])
            shift[q] = weights[0] + x[dev[q]] @ weights[1:]
        result[mode] = {}
        for name,base in [("raw",raw),("v96",load_v96(dev,folds))]:
            p = expit(ref.logit(base) + shift)
            result[mode][name] = dict(metrics=ref.score(p,y), all=ref.pair(p,base,y),
                hard546=ref.pair(p,base,y,hard), fold_net=[ref.pair(p[b],base[b],y[b])["net_correct"] for _,b in splits])
    ref.write(ROOT/"reports/cabin_periodicity/result.json",result)
    print(result)


if __name__ == "__main__":
    run()
