"""Export the exact marginal partition correction over the verified v131 competition roster."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from threadpoolctl import threadpool_limits

from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, Partitions, partition_axes, teachers, write
from exaone_validation_data import compact_features
from export_budget_quarter_n23 import source
from kaggle_workflow import sha256, validate_submission

VERSION = "n23_partition_residual"
NAME = "proposal_" + VERSION


def input_frame(roster):
    frame = compact_features(roster)
    bookings = roster.PassengerId.str.split("_").str[0].astype(int)
    frame["BookingSize"] = bookings.map(bookings.value_counts())
    return frame, bookings.to_numpy()


def partition_score(frame, t, q, y, teacher, base):
    a = logit(teacher.clip(.002, .998))
    b = logit(base.clip(.002, .998))
    scores = [Partitions(axis).infer(t, q, y, a, b, .8)[0] for axis in partition_axes(frame).values()]
    return np.mean(scores, axis=0)


def local_replay(frame, g, truth):
    plans = {}
    stable = []
    with np.load(ART / "nov23_release_vote_information/panel.npz") as z:
        shared = (z["votes"] != z["y"][None, :, None]).all((0, 2))
        common_right = (z["votes"] == z["y"][None, :, None]).all((0, 2))
        rows = z["row"]
        np.testing.assert_array_equal(truth[rows], z["y"])
    for key in [*map(str, SEEDS), "v131"]:
        with np.load(ART / f"nov23_broad_partition_evidence/{key}.npz") as z:
            y, ref, base, fold = [z[k] for k in ("y","ref","base","fold")]
            np.testing.assert_array_equal(z["row"], rows)
            np.testing.assert_array_equal(truth[rows], y)
            p = z["update"].copy()
            if key == "v131":
                local, groups = frame.iloc[rows].reset_index(drop=True), g[rows]
                recomputed = np.empty(len(y))
                for f in np.unique(fold):
                    q = np.flatnonzero(fold == f)
                    t = np.flatnonzero((fold != f) & ~np.isin(groups, groups[q]))
                    pt, _ = teachers(local, y, groups, t, q, 1171+int(f))
                    recomputed[q] = partition_score(local, t, q, y, pt, base[q])
                np.testing.assert_allclose(recomputed, p, atol=2e-12, rtol=0)
                p = np.where(ref != (base >= .5), ref, recomputed)
            hit = (p >= .5) == y
            if key != "v131":
                stable.append(hit)
            plans[key] = dict(**paired(hit, ref == y), common_wrong_correct=int(hit[shared].sum()),
                              common_right_harmed=int((~hit[common_right]).sum()))
    return dict(plans=plans, stable_common_wrong_correct=int((np.all(stable, axis=0) & shared).sum()),
                stable_common_right_harmed=int((~np.any(stable, axis=0) & common_right).sum()))


def export():
    train_file = ROOT/"data/raw/train.csv"
    test_file = ROOT/"data/raw/test.csv"
    train, test = [pd.read_csv(p, usecols=lambda c: c != "Name") for p in (train_file, test_file)]
    y = train.Transported.to_numpy(bool)
    assert "Transported" not in test
    frame, groups = input_frame(pd.concat([train.drop(columns=["Transported"]), test], ignore_index=True))
    t, q = np.arange(len(train)), np.arange(len(train), len(frame))
    assert not np.intersect1d(groups[t], groups[q]).size
    checks = local_replay(frame, groups, y)
    base, _, base_hash = source("proposal_v108_party_bills", test)
    _, ref, ref_hash = source("proposal_v131_party_consensus", test)
    lock = ref != (base >= .5)
    train_teacher, _ = teachers(frame, y, groups, t, q, 1171)
    raw_score = partition_score(frame, t, q, y, train_teacher, base)
    p = np.where(lock, ref, raw_score)

    directory = ART/NAME
    report = ROOT/"reports"/NAME
    directory.mkdir(exist_ok=True)
    file = directory/"teacher_context.npz"
    np.savez_compressed(file, passenger_id=train.PassengerId.to_numpy(dtype=str), oof_teacher=train_teacher)
    with np.load(file, allow_pickle=False) as saved:
        np.testing.assert_array_equal(train.PassengerId, saved["passenger_id"])
        restored = partition_score(frame, t, q, y, saved["oof_teacher"], base)
    repeat = np.where(lock, ref, restored)
    np.testing.assert_array_equal(repeat, p)
    assert ((0 <= p) & (p <= 1)).all()
    csv = directory/"submission.csv"
    pd.DataFrame(dict(PassengerId=test.PassengerId, Transported=p>=.5)).to_csv(csv, index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId, probability=p, reload_probability=repeat)).to_csv(
        directory/"test_probabilities.csv", index=False)
    validation = validate_submission(csv, test_file)
    distances = {}
    for receipt in (ROOT/"reports").glob("*/online_attempts/*.json"):
        if receipt.parent.parent == report:
            continue
        record = json.loads(receipt.read_text())
        digest = record.get("file_sha256")
        assert digest != validation["sha256"], receipt
        prior = ART/receipt.parent.parent.name/"submission.csv"
        if prior.is_file() and sha256(prior) == digest:
            batch = pd.read_csv(prior)
            np.testing.assert_array_equal(batch.PassengerId, test.PassengerId)
            distances[receipt.parent.parent.name] = int(((p >= .5) != batch.Transported).sum())
    near = min(distances.values())
    assert near > 0
    metrics = dict(**validation, version=VERSION, model_sha256=sha256(file), reload_difference=0,
                   anchors=dict(v108=base_hash, v131=ref_hash), train_sha256=sha256(train_file),
                   test_sha256=sha256(test_file), training_people=len(train), teacher_seed=1171,
                   previous_decisions_locked=int(lock.sum()), differences_from_v131=int(((p>=.5)!=ref).sum()),
                   nearest_prior_changes=near, prior_files_compared=len(distances),
                   nearest_prior_files=[k for k, v in distances.items() if v==near], local_replay=checks)
    for folder in (directory, report):
        write(folder/"manifest.json", metrics)
        write(folder/"selection.json", dict(version=VERSION, sha256=validation["sha256"]))
    print(metrics, flush=True)


if __name__ == "__main__":
    with threadpool_limits(2):
        export()
