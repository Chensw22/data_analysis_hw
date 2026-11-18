"""Test whether bill allocation within a party explains residual errors of v115."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import GroupKFold

import peer_bill_correction as peer
import relational_error_corrector as ref

ROOT = Path(__file__).resolve().parent


def allocation_features(roster):
    """Conserve each service total while splitting it between likely users."""
    groups = roster.PassengerId.str.split("_").str[0]
    age = roster.Age.to_numpy(float)
    sleeping = roster.CryoSleep.map({True: 1., False: 0.}).to_numpy(float)
    cabin = roster.Cabin.fillna("").to_numpy(str)
    bills = roster[peer.BILLS].to_numpy(float)
    result = {name: np.zeros_like(bills) for name in ["equal_share", "recipient", "rotated_recipient"]}
    eligible = np.zeros(len(roster), bool)
    for rows in roster.groupby(groups, sort=False).indices.values():
        if len(rows) < 2:
            continue
        spend = np.nan_to_num(bills[rows])
        # An unrecorded bill contributes no observed money to redistribute.
        if not spend.any():
            continue
        eligible[rows] = True
        ages, rooms = age[rows], cabin[rows]
        same_cabin = (rooms[:, None] == rooms) & (rooms[:, None] != "")
        age_gap = np.abs(ages[:, None] - ages)
        affinity = np.where(np.isfinite(age_gap), np.exp(-age_gap / 24.), .5)
        awake_weight = np.where(sleeping[rows] == 1, .1, np.where(sleeping[rows] == 0, 1., .55))
        weights = (1 + same_cabin) * (.5 + affinity) * awake_weight[None, :]
        weights /= weights.sum(axis=1, keepdims=True)
        equal = np.broadcast_to(spend.mean(axis=0), spend.shape)
        recipient = weights.T @ spend
        np.testing.assert_allclose(recipient.sum(axis=0), spend.sum(axis=0))
        for key, amount in [("equal_share", equal), ("recipient", recipient),
                            ("rotated_recipient", np.roll(recipient, 1, axis=0))]:
            result[key][rows] = np.log1p(amount) - np.log1p(spend)
    return eligible, result


def run():
    data, splits, fold, _, evidence = ref.overlap.load_authenticated()
    dev, y, groups = data["dev"], data["y"], np.asarray(data["groups"])
    attrs = ["PassengerId", "Age", "Cabin", "CryoSleep", *peer.BILLS]
    roster = pd.concat([pd.read_csv(ROOT / "data/raw" / f, usecols=attrs)
                        for f in ["train.csv", "test.csv"]], ignore_index=True)
    eligible, views = allocation_features(roster)
    passenger, known, _, old_design = peer.context()
    np.testing.assert_array_equal(roster.PassengerId, passenger)
    np.testing.assert_array_equal(passenger.iloc[dev], data["passengers"])
    with np.load(ROOT / "artifacts/v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"], dev)
        np.testing.assert_array_equal(z["fold_id"], fold)
        prob = z["v108"]
    with np.load(ROOT / "artifacts/consensus_error_routes.npz") as z:
        np.testing.assert_array_equal(z["dev"], dev)
        np.testing.assert_array_equal(z["fold"], fold)
        old_flip = z["combined_by_historical_agreement"]
    base = (prob >= .5) ^ old_flip
    available = eligible[dev] & ~old_flip
    decisions = {key: base.copy() for key in views}
    for outer, (a, b) in enumerate(splits):
        teacher = ref.load_inner(outer, a, b, dev, y, evidence)
        # Reconstruct the old bill adjustment on cross-fitted training rows.
        corrected = teacher.astype(float).copy()
        for t, q in GroupKFold(3).split(a, groups=groups[a]):
            t = t[known[dev[a[t]]]]
            q = q[known[dev[a[q]]]]
            fit = peer.fit_offset(old_design[dev[a[t]]], teacher[t], y[a[t]])
            corrected[q] = expit(ref.logit(teacher[q]) + peer.apply(fit, old_design[dev[a[q]]]))
        t = a[eligible[dev[a]]]
        teacher_for_t = corrected[eligible[dev[a]]]
        q = b[available[b]]
        for key, design in views.items():
            model = peer.fit_offset(design[dev[t]], teacher_for_t, y[t])
            decisions[key][q] = (ref.logit(prob[q]) + peer.apply(model, design[dev[q]])) >= 0
    catalog = pd.read_csv(ROOT / "artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id, dev)
    np.testing.assert_array_equal(catalog.y, y)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    summary = dict(rows=len(y), eligible=int(available.sum()), eligible_errors=int((available & (base != y)).sum()),
                   eligible_hard_remaining=int((available & hard & (base != y)).sum()),
                   methods={})
    for key, pred in decisions.items():
        fixed = (pred == y) & (base != y)
        harm = (pred != y) & (base == y)
        summary["methods"][key] = dict(accuracy=float(np.mean(pred == y)), fixed=int(fixed.sum()),
            harmed=int(harm.sum()), hard_fixed=int((fixed & hard).sum()), hard_harmed=int((harm & hard).sum()),
            fold_net=[int(fixed[b].sum()) - int(harm[b].sum()) for _, b in splits])
    np.savez_compressed(ROOT / "artifacts/party_bill_allocation.npz", dev=dev, fold=fold, base=base, **decisions)
    ref.write(ROOT / "reports/party_bill_allocation/result.json", summary)
    print(summary)


if __name__ == "__main__":
    run()
