"""Describe all jointly missed development passengers and the cost of reversing consensus."""
import numpy as np
import pandas as pd

from released_correction_audit_n23 import ART, REPORT, ROOT, development, write


BILLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
RAW = ["PassengerId", "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", *BILLS]


def passenger_attributes():
    raw = pd.read_csv(ROOT / "data/raw/train.csv", usecols=RAW)
    passenger_ids = pd.concat([raw.PassengerId, pd.read_csv(
        ROOT / "data/raw/test.csv", usecols=["PassengerId"]).PassengerId], ignore_index=True)
    assert passenger_ids.is_unique
    booking_sizes = passenger_ids.str.split("_").str[0].value_counts()
    row_table = pd.read_csv(ART / "nov21_party_negative/oof.csv", usecols=["row", "target"])
    assert row_table.row.is_unique
    frame = raw.iloc[row_table.row].reset_index(drop=True)
    bills = frame[BILLS]
    spending = np.select([bills.gt(0).any(axis=1), bills.eq(0).all(axis=1)],
                         ["有正额消费", "五项明确为零"], default="未见正额且有缺项")
    size = frame.PassengerId.str.split("_").str[0].map(booking_sizes)
    cabin = frame.Cabin.str.split("/", expand=True)
    age = pd.cut(frame.Age, [-np.inf, 13, 18, np.inf], right=False,
                 labels=["未满13岁", "13–17岁", "18岁及以上"])
    axes = pd.DataFrame({
        "消费记录": spending,
        "出发星球": frame.HomePlanet,
        "休眠": frame.CryoSleep.map({True: "是", False: "否"}),
        "舱层": cabin[0], "舷侧": cabin[2],
        "目的地": frame.Destination,
        "年龄": age.astype("string"),
        "同行预订": np.where(size > 1, "多名乘客", "单人"),
        "是否缺消费记录": np.where(bills.isna().any(axis=1), "有缺项", "完整"),
    }).fillna("未知")
    return axes, row_table.target.to_numpy(bool)


def main():
    label, predictions, _ = development()
    axes, row_labels = passenger_attributes()
    np.testing.assert_array_equal(label, row_labels)
    prediction_matrix = np.column_stack(list(predictions.values()))
    consensus = prediction_matrix.min(axis=1) == prediction_matrix.max(axis=1)
    direction = prediction_matrix[:, 0]
    missed = consensus & (direction != label)
    assert missed.sum() == 941 and len(label) == 6954 and prediction_matrix.shape[1] == 24
    totals = []
    for target in (False, True):
        same = label == target
        totals.append(dict(target=target, population=int(same.sum()), missed=int((same & missed).sum()),
                           other=int((same & ~missed).sum())))
    profiles = []
    for axis in axes:
        by_group = []
        for value in axes[axis].unique():
            group = axes[axis].eq(value).to_numpy()
            for total in totals:
                target = total["target"]
                same_label = label == target
                in_cell = group & same_label
                # An observable intervention uses only the characteristic and the
                # unanimous prediction. The actual target is used for scoring only.
                flip = group & consensus & (direction != target)
                repaired = int((flip & same_label).sum())
                harmed = int((flip & ~same_label).sum())
                other = int((in_cell & ~missed).sum())
                assert repaired == (in_cell & missed).sum()
                by_group.append(dict(value=value, target=target, group_size=int(group.sum()),
                    cell_size=int(in_cell.sum()), missed=repaired, other=other,
                    share_of_all_missed=repaired / int(missed.sum()),
                    share_of_all_other=other / int((~missed).sum()),
                    share_of_same_label_missed=repaired / total["missed"],
                    share_of_same_label_other=other / total["other"],
                    unanimous_flip_harm=harmed, unanimous_flip_net=repaired-harmed))
        assert sum(row["cell_size"] for row in by_group) == len(label)
        assert sum(row["missed"] for row in by_group) == missed.sum()
        profiles.append(dict(axis=axis, rows=by_group))
    payment_exception = (label & axes["消费记录"].eq("有正额消费").to_numpy()) | (
        ~label & axes["消费记录"].eq("五项明确为零").to_numpy())
    earth = axes["出发星球"].eq("Earth").to_numpy()
    overlap = {}
    for description, group in {"both": payment_exception & earth, "spending_only": payment_exception & ~earth,
                               "earth_only": earth & ~payment_exception, "neither": ~earth & ~payment_exception}.items():
        overlap[description] = dict(missed=int((group & missed).sum()), other=int((group & ~missed).sum()))
    assert sum(count["missed"] for count in overlap.values()) == missed.sum()
    write(REPORT / "population.json", dict(totals=totals, earth_and_payment_exceptions=overlap,
          unanimously_classified=int(consensus.sum()),
          correct_unanimous=int((consensus & ~missed).sum()), profiles=profiles))
    for profile in profiles:
        print(profile["axis"], [(r["value"], r["target"], r["missed"], r["other"], r["unanimous_flip_harm"])
                                for r in profile["rows"]])


if __name__ == "__main__":
    main()
