"""Test whether v115 assigns the right number of positive labels to the wrong party members."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
BILLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def design(frame):
    cabin = frame.Cabin.str.split("/", expand=True)
    cat = frame[["HomePlanet", "CryoSleep", "Destination", "VIP"]].assign(
        deck=cabin[0], side=cabin[2])
    features = pd.get_dummies(cat.astype("string").fillna("missing"), dtype=float)
    features["age"] = frame.Age.fillna(-1)
    features["age_missing"] = frame.Age.isna()
    features["child"] = frame.Age.lt(13)
    for bill in BILLS:
        features[bill] = np.log1p(frame[bill]).fillna(0)
        features[bill + "_positive"] = frame[bill].gt(0)
        features[bill + "_missing"] = frame[bill].isna()
    return features.to_numpy(dtype=float)


def blocks(party, selected):
    by_party = {}
    for i in selected:
        by_party.setdefault(party[i], []).append(i)
    return [np.array(ix) for ix in by_party.values()]


def rank(train_groups, x, y):
    differences = []
    for group in train_groups:
        for i, j in combinations(group, 2):
            if y[i] != y[j]:
                differences.append((x[i] - x[j]) if y[i] else (x[j] - x[i]))
    positive = np.stack(differences)
    train = np.concatenate([positive, -positive])
    labels = np.repeat([1, 0], len(positive))
    scale = StandardScaler(with_mean=False).fit(train)
    model = LogisticRegression(C=1., fit_intercept=False).fit(scale.transform(train), labels)
    return model.decision_function(scale.transform(x)), len(positive)


def oracle(groups, base, y):
    # Each swap consumes one false positive and one false negative in the same party.
    return int(2 * sum(min((base[g] & ~y[g]).sum(), (~base[g] & y[g]).sum()) for g in groups))


def metrics(pred, base, y, folds, hard):
    gain = (pred == y).astype(int) - (base == y).astype(int)
    return dict(correct=int((pred == y).sum()), accuracy=float((pred == y).mean()),
                repairs=int((gain == 1).sum()), harms=int((gain == -1).sum()),
                hard_repairs=int((gain[hard] == 1).sum()), hard_harms=int((gain[hard] == -1).sum()),
                fold_net=[int(gain[folds == f].sum()) for f in np.unique(folds)])


def run():
    art = ROOT / "artifacts"
    with np.load(art / "v108_exact_oof/oof.npz") as z:
        dev, folds, prob, y, ids = (z[k] for k in ["dev_ids", "fold_id", "v108", "true", "PassengerId"])
    y = y.astype(bool)
    with np.load(art / "consensus_error_routes.npz") as z:
        np.testing.assert_array_equal(dev, z["dev"])
        np.testing.assert_array_equal(folds, z["fold"])
        base = (prob >= .5) ^ z["combined_by_historical_agreement"]
    rows = pd.read_csv(art / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(rows.row_id, dev)
    np.testing.assert_array_equal(rows.y, y)
    hard = rows.all_recipes_still_wrong.to_numpy(bool)
    columns = ["PassengerId", "Cabin", "HomePlanet", "CryoSleep", "Destination", "VIP", "Age", *BILLS]
    roster = pd.read_csv(ROOT / "data/raw/train.csv", usecols=columns)
    frame = roster.iloc[dev].reset_index(drop=True)
    np.testing.assert_array_equal(frame.PassengerId, ids)
    test = pd.read_csv(ROOT / "data/raw/test.csv", usecols=["PassengerId", "Cabin"])
    party = frame.PassengerId.str.split("_").str[0].to_numpy()
    x = design(frame)
    answers = {k: base.copy() for k in ["within_party_rank", "v108_order_control"]}
    diagnostics = []
    for fold in np.unique(folds):
        q = np.flatnonzero(folds == fold)
        train = np.flatnonzero((folds != fold) & ~np.isin(party, party[q]))
        assert not set(party[train]) & set(party[q])
        query_groups = blocks(party, q)
        score, pairs = rank(blocks(party, train), x, y)
        for g in query_groups:
            count = int(base[g].sum())
            if 0 < count < len(g):
                for key, s in [("within_party_rank", score), ("v108_order_control", prob)]:
                    # Preserve the incumbent choice on ties.
                    order = np.lexsort((g, ~base[g], -s[g]))
                    answers[key][g] = False
                    answers[key][g[order[:count]]] = True
                    assert answers[key][g].sum() == count
        diagnostics.append(dict(train_passengers=len(train), discordant_fit_pairs=pairs,
                                fixed_count_oracle_net=oracle(query_groups, base, y),
                                eligible_groups=int(sum(0 < base[g].sum() < len(g) for g in query_groups))))
    pooled_ceiling = oracle(blocks(party, np.arange(len(y))), base, y)
    fold_ceiling = sum(r["fixed_count_oracle_net"] for r in diagnostics)
    party_size = pd.concat([roster.PassengerId, test.PassengerId]).str.split("_").str[0].value_counts()
    solo = pd.Series(party).map(party_size).eq(1).to_numpy()
    errors_by_party_size = {}
    for name, selected in [("solo", solo), ("multi", ~solo)]:
        errors = selected & (base != y)
        errors_by_party_size[name] = dict(rows=int(selected.sum()), wrong=int(errors.sum()),
                                           historical_hard_still_wrong=int((hard & errors).sum()))
    result = dict(errors_by_party_size=errors_by_party_size, rows=len(y), correct_base=int((base == y).sum()),
                  required_net=int(np.floor(.85 * len(y)) + 1 - (base == y).sum()),
                  data=dict(all_train_rows=len(roster), previous_holdout_rows=len(roster)-len(dev),
                            test_rows=len(test), labeled_dev_used=len(y),
                            test_rows_sharing_train_party=int(test.PassengerId.str.split("_").str[0].isin(
                                roster.PassengerId.str.split("_").str[0]).sum()),
                            test_rows_sharing_observed_train_cabin=int(test.Cabin.isin(roster.Cabin.dropna()).sum())),
                  fold_diagnostics=diagnostics, pooled_party_oracle_net=pooled_ceiling,
                  fold_query_party_oracle_net=fold_ceiling,
                  models={k: metrics(p, base, y, folds, hard) for k,p in answers.items()})
    out = ROOT / "reports/observation_collision_audit/result.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result)


if __name__ == "__main__":
    run()
