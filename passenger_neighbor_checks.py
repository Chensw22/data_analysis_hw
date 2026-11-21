"""Check candidate corrections against observed outcomes of unrelated, similar passengers."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import export_consensus_routes as routes
import relational_error_corrector as ref
from confirm_sleeper_scope import NAME as INCUMBENT, confirm
from export_nov21_error_batch import test_probabilities
from exact_fold_error_selector import counts
from exaone_validation_data import compact_features
from peer_bill_correction import BILLS
from route_consensus_errors import FAMILIES
from stable_consensus_selector import persistent_mask

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
OUT = ART / "nov21_case_checks"
CHALLENGER = "proposal_v118_error_union"


def coordinates(x, view):
    columns = []
    for name, weight in {"HomePlanet": 1.2, "Deck": 1.0, "CryoSleep": 1.4, "Side": .55,
                         "Destination": .55, "VIP": .25}.items():
        for value in sorted(x[name].unique()):
            columns.append(weight * x[name].eq(value).to_numpy())
    for bill in BILLS:
        amount = x[bill]
        columns.extend([.45 * amount.isna(), .55 * amount.gt(0),
                        (.55 if view == "billing" else .17) * np.log1p(amount.fillna(0))])
    columns.extend([x.Age.fillna(28) / 33, .5 * x.Age.isna(), .55 * x.Age.lt(13),
                    x.CabinNumber.fillna(800) / (250 if view == "berth" else 1000),
                    .8 * x.CabinNumber.isna()])
    return np.column_stack(columns)


def vote(matrix, groups, train_rows, query_rows, labels, k):
    result = []
    for start in range(0, len(query_rows), 128):
        rows = query_rows[start:start + 128]
        dist = cdist(matrix[rows], matrix[train_rows], metric="cityblock")
        dist[groups[rows, None] == groups[train_rows]] = np.inf
        # Stable tie-break in original roster order, even for identical manifests.
        nearest = np.argsort(dist, axis=1, kind="stable")[:, :k]
        distance = np.take_along_axis(dist, nearest, axis=1)
        assert np.isfinite(distance).all()
        weights = 1 / (1 + distance)
        result.extend((weights * labels[nearest]).sum(1) / weights.sum(1))
    return np.asarray(result)


def decisions(anchor, base, challenger, history, probability, sleep, billing, berth):
    eligible = anchor == base
    disagreement = history != base[:, None]
    opposite_bills = np.where(base, 1 - billing, billing)
    opposite_berth = np.where(base, 1 - berth, berth)
    unanimous = ~disagreement.any(axis=1)
    margin = np.abs(probability - .5)
    # Two correlated neighbor views count as one outcome estimate.
    neighbor_opposition = (opposite_bills + opposite_berth) / 2 - .5
    consistent = (opposite_bills > .5) & (opposite_berth > .5)
    return {
        "v135b": anchor ^ (eligible & unanimous & consistent & (neighbor_opposition > margin + .05)),
        "v136b": anchor ^ (eligible & unanimous & sleep & consistent & (neighbor_opposition > .12)),
        "v135": anchor ^ (eligible & (challenger != base) & (opposite_bills > .5)),
        "v136": anchor ^ (eligible & ((challenger != base) | (disagreement.sum(1) >= 2))
                          & (opposite_bills > .60) & (opposite_berth > .60)),
    }


def run():
    inp, _, fold, models, _ = ref.overlap.load_authenticated()
    dev, y = inp["dev"], inp["y"].astype(bool)
    train = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c not in ["Name", "Transported"])
    test = pd.read_csv(ROOT / "data/raw/test.csv", usecols=lambda c: c != "Name")
    roster = pd.concat([train, test], ignore_index=True)
    groups = roster.PassengerId.str.split("_").str[0].to_numpy()
    x = compact_features(roster)
    matrices = [coordinates(x, view) for view in ["billing", "berth"]]
    oof = np.zeros((2, len(dev)))
    for part in np.unique(fold):
        t, q = fold != part, fold == part
        for arr, matrix, k in zip(oof, matrices, [25, 41]):
            arr[q] = vote(matrix, groups, dev[t], dev[q], y[t], k)
    rows = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(rows.row_id, dev)
    hard = rows.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp, models, hard)
    with np.load(ART / "sleeper_consensus_scope.npz") as scoped, np.load(ART / "sleeper_manifest_risk.npz") as witness, np.load(ART / "replicate_error_committee.npz") as committee:
        for cache in (scoped, witness, committee):
            np.testing.assert_array_equal(cache["dev"], dev)
            np.testing.assert_array_equal(cache["fold"], fold)
        base = scoped["base"]
        anchor = confirm(base, scoped["original"], witness["manifest_berth_party_a"])
        challenger = committee["original_either"]
        assert challenger.dtype == bool
    with np.load(ART / "v108_exact_oof/oof.npz") as cache:
        np.testing.assert_array_equal(cache["dev_ids"], dev)
        probability = cache["v108"]
        np.testing.assert_array_equal(probability >= .5, base)
    sleep = x.CryoSleep.eq("True").to_numpy()
    pred = decisions(anchor, base, challenger, np.column_stack([models[n] >= .5 for n in FAMILIES]),
                     probability, sleep[dev], *oof)
    summary = {}
    OUT.mkdir(exist_ok=True)
    frame = pd.DataFrame(dict(PassengerId=train.PassengerId.iloc[dev].to_numpy(), fold=fold, observed=y,
                              incumbent=anchor, hard=hard, persistent=persistent, billing=oof[0], berth=oof[1]))
    for name, p in pred.items():
        summary[name] = counts(p, anchor, y, fold, hard, persistent)
        frame[name] = p
    frame.to_csv(OUT / "local_rows.csv", index=False)
    labels = pd.read_csv(ROOT / "data/raw/train.csv", usecols=["Transported"]).Transported.to_numpy(bool)
    train_rows, test_rows = np.arange(len(train)), np.arange(len(train), len(roster))
    votes = [vote(matrix, groups, train_rows, test_rows, labels, k) for matrix, k in zip(matrices, [25,41])]
    ids = test.PassengerId.to_numpy(str)
    def official(name):
        return routes.official_decisions(name, ids)[0]
    anchor = official(INCUMBENT)
    final = decisions(anchor, official(routes.BASE), official(CHALLENGER),
                      np.column_stack([official(n) for n in routes.HISTORICAL]),
                      test_probabilities(routes.BASE, ids), sleep[test_rows], *votes)
    for name, p in final.items():
        summary[name]["test_changes_vs_v128"] = int((p != anchor).sum())
        pd.DataFrame(dict(PassengerId=ids, Transported=p)).to_csv(OUT / (name + ".csv"), index=False)
    ref.write(ROOT / "reports/nov21_case_checks/result.json", dict(baseline=INCUMBENT, results=summary,
              validation="Original passenger folds, train-fold targets only, same party excluded from all donor pools. "
              "Each view uses outcome frequencies, with Manhattan distances and inverse distance smoothing. "
              "v135 requires an existing v118 correction and 25 billing neighbors above 50%. "
              "v136 requires v118 or two opposing historical votes and both views above 60%. "
              "v135b compares the average opposing vote margin to the v108 margin plus .05 among unanimous cases. "
              "v136b applies a .12 average opposing margin among unanimously classified, observed cryosleep passengers."))
    print(json.dumps(summary))


if __name__ == "__main__":
    run()
