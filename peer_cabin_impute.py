"""Recover missing cabins using companions and boarding order within deck and side."""
import argparse
import numpy as np
import pandas as pd
import peer_planet_impute as common

PROPOSAL = "proposal_nov18_peer_cabin"
VERSION = "v106_v96_peer_cabin"


def infer():
    frame = pd.concat([pd.read_csv(common.ROOT/"data/raw"/file, usecols=["PassengerId","Cabin"])
                       for file in ["train.csv","test.csv"]], ignore_index=True)
    group = frame.PassengerId.str.split("_").str[0].astype(int)
    location = frame.Cabin.str.split("/", expand=True)
    location.columns = ["Deck","CabinNumber","Side"]
    location.CabinNumber = pd.to_numeric(location.CabinNumber)
    location["group"] = group
    # Each group contributes once per deck and side to the interpolation.
    knots = {key: rows.groupby("group").CabinNumber.median().sort_index()
             for key, rows in location.groupby(["Deck","Side"])}
    filled = {}
    observed = correct = 0
    distances = []
    for indices in frame.groupby(group).indices.values():
        known = location.iloc[indices].dropna(subset=["CabinNumber"])
        for i in indices:
            peers = known.drop(index=i, errors="ignore")
            if peers.empty or peers.Deck.nunique() != 1 or peers.Side.nunique() != 1:
                continue
            key = (peers.Deck.iloc[0], peers.Side.iloc[0])
            curve = knots[key]
            # Exclude the entire target group so masked cabin numbers cannot enter their own estimates.
            train = curve.index != group.iat[i]
            if not train.any():
                continue
            number = float(np.rint(np.interp(group.iat[i],curve.index[train],curve.to_numpy()[train])))
            if pd.notna(frame.Cabin.iat[i]):
                observed += 1
                if key == (location.Deck.iat[i],location.Side.iat[i]):
                    correct += 1
                    distances.append(abs(location.CabinNumber.iat[i] - number))
            else:
                filled[i] = (*key,number)
    updates = pd.DataFrame.from_dict(filled,orient="index",columns=["Deck","Side","CabinNumber"])
    catalog = pd.read_csv(common.ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    hard = catalog.loc[catalog.all_recipes_still_wrong, "row_id"]
    test = updates.loc[updates.index >= 8693].sort_index()
    stats = dict(masked_cabins=observed, correct_deck_and_side=correct,
                 median_number_error_when_deck_side_correct=float(np.median(distances)),
                 test_fillable=len(test), hard546_fillable=int(hard.isin(updates.index).sum()))
    return frame,test,stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export",action="store_true")
    args = parser.parse_args()
    frame, patches, stats = infer()
    print(stats,flush=True)
    if args.export:
        common.export(frame, patches, stats, PROPOSAL, VERSION)
