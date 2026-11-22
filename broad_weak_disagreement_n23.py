"""Count correct passengers sharing the same persistent dissent pattern as every weak-only rescue."""
import numpy as np
import pandas as pd

from archive_error_recovery_n23 import ART, ROOT, SEEDS, panel, predictions
from broad_booking_policy_n23 import EXTRA
from broad_error_placement_control_n23 import SOURCES
from relational_error_corrector import write

REPORT = ROOT / "reports/nov23_broad_weak_disagreement"
OUT = ART / "nov23_broad_weak_disagreement"


def archive():
    row, y, base, old, _, names, _ = panel()
    additions = {**EXTRA, **dict(list(SOURCES.items())[:4])}
    latest = np.stack([predictions(row, y, seed, additions)[1] for seed in SEEDS])
    votes = np.concatenate([old, latest], axis=2) >= .5
    reference = base >= .5
    correct = reference == y
    peer = ((votes == y[None, :, None]).mean(1) >= correct.mean(1)[:, None] - .01).all(0)
    opposite = (votes != reference[:, :, None]).all(0)
    consistent = (reference[0] == reference).all(0)
    weak_only = consistent & opposite[:, ~peer].any(1) & ~opposite[:, peer].any(1)
    historic = weak_only & ~correct.any(0)
    assert historic.sum() == 409 and weak_only.sum() == 1594
    masks = dict(ever=~correct.all(0), repeat=~correct.any(0), weak409=historic,
                 mirror1185=weak_only & correct.all(0), never172=~(votes == y[None, :, None]).any((0, 2)) & ~correct.any(0))
    assert masks["never172"].sum() == 172
    return row, y, base, weak_only, masks, opposite[:, ~peer], np.array(names + list(additions))[~peer]


def run():
    row, y, base, eligible, masks, dissent, names = archive()
    raw = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c not in ("Name", "Transported")).iloc[row]
    bills = raw[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]]
    groups = {"all": np.ones(len(y), bool), "predict_true": base[0] >= .5, "predict_false": base[0] < .5,
              "sleep": raw.CryoSleep.eq(True).to_numpy(), "awake": raw.CryoSleep.eq(False).to_numpy(),
              "sleep_unknown": raw.CryoSleep.isna().to_numpy(), "observed_positive_spend": bills.gt(0).any(axis=1).to_numpy(),
              "no_observed_positive_spend": ~bills.gt(0).any(axis=1).to_numpy(),
              "all_input_fields_present": raw.notna().all(axis=1).to_numpy()}
    confidence = np.maximum(base, 1-base).mean(0)
    groups.update({f"confidence_{lo:.1f}_{hi:.1f}": (confidence >= lo) & (confidence < hi)
                   for lo, hi in ((.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.01))})
    def counts(m):
        w = int((eligible & m & masks["weak409"]).sum())
        c = int((eligible & m & masks["mirror1185"]).sum())
        return dict(passengers=w+c, correct_to_flip=w, incorrect_to_flip=c, net=w-c)
    write(REPORT / "archive.json", dict(passengers=len(y), groups={k: counts(m) for k,m in groups.items()},
          models={str(name): counts(d) for name,d in zip(names,dissent.T)}))


if __name__ == "__main__":
    run()
