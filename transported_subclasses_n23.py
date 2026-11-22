"""Learn separate within-outcome subclasses, then marginalize to transported probability."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from class_analogue_experts_n23 import design
from exaone_validation_data import CATEGORICAL, compact_features
from group_crossfit_residual_discovery_n23 import teachers
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART, ROOT, SEEDS, read_panel
from relational_error_corrector import write


def predict(frame, targets, query, binary):
    model = CatBoostClassifier(iterations=320, depth=5, learning_rate=.055,
                               thread_count=2, verbose=False, random_seed=605,
                               allow_writing_files=False)
    model.fit(frame, targets, cat_features=CATEGORICAL)
    scores = model.predict_proba(query)
    positive = model.classes_ == 1 if binary else model.classes_ >= 2
    return scores[:, positive].sum(axis=1)


def partition_labels(frame, y, train, train_prob):
    targets = y[train].astype(int)
    difficult = (train_prob >= .5) != targets
    groups = np.zeros(len(train), dtype=int)
    matrix, _ = design(frame, train)
    for label in [0, 1]:
        member = np.flatnonzero(targets == label)
        groups[member] = KMeans(n_clusters=2, n_init=10, random_state=905).fit_predict(matrix[train[member]])
    rng = np.random.default_rng(1021)
    shuffled = difficult.copy()
    for label in [0, 1]:
        idx = targets == label
        shuffled[idx] = rng.permutation(difficult[idx])
    return dict(binary=targets, teacher_difficulty=2*targets+difficult,
                passenger_clusters=2*targets+groups, shuffled_difficulty=2*targets+shuffled)


def stats(decision, baseline, target, persistent, unresolved, single):
    now = decision == target
    before = baseline == target
    fixed = now & ~before
    harmed = ~now & before
    return dict(accuracy=float(now.mean()), fixed=int(fixed.sum()), harmed=int(harmed.sum()),
                fixed_false_negatives=int((fixed & target).sum()),
                new_false_positives=int((harmed & ~target).sum()),
                fixed_false_positives=int((fixed & ~target).sum()),
                new_false_negatives=int((harmed & target).sum()),
                persistent_false_negatives=int((fixed & target & persistent).sum()),
                eight_family_false_negatives=int((fixed & target & unresolved).sum()),
                solo_persistent_false_negatives=int((fixed & target & persistent & single).sum()))


def run():
    rows = pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    y, old, others = read_panel()
    persistent = ~old.any(axis=0)
    unresolved = persistent & ~others.any(axis=(0, 1))
    raw = pd.read_csv(ROOT/"data/raw/train.csv", usecols=lambda c: c not in {"Transported","Name"})
    bookings = raw.PassengerId.str.split("_").str[0]
    party_size = bookings.map(bookings.value_counts()).to_numpy()[rows]
    group = bookings.to_numpy()[rows]
    frame = compact_features(raw.iloc[rows]).reset_index(drop=True)
    folder = ART/"nov23_transported_subclasses"
    folder.mkdir(exist_ok=True)
    results = {}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as saved:
            fold, base = saved["fold"], saved["base"]
            verify(saved, rows, y, fold)
        predictions = {}
        for k in np.unique(fold):
            train = np.flatnonzero(fold != k)
            query = np.flatnonzero(fold == k)
            assert not np.intersect1d(group[train], group[query]).size
            train_prob, _ = teachers(frame, y, group, train, query, seed+int(k))
            for name, labels in partition_labels(frame, y, train, train_prob).items():
                predictions.setdefault(name, np.full(len(y), np.nan))[query] = predict(
                    frame.iloc[train], labels, frame.iloc[query], name == "binary")
        baseline = base >= .5
        results[seed] = {}
        for name, p in predictions.items():
            assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
            results[seed][name] = {
                kind: stats(decision, baseline, y, persistent, unresolved, party_size == 1)
                for kind, decision in {
                    "direct": p >= .5, "quarter_blend": .75*base+.25*p >= .5,
                    "restore_only": baseline | (p >= .5)}.items()}
        np.savez_compressed(folder/f"{seed}.npz", row=rows, y=y, fold=fold, base=base, **predictions)
        write(ROOT/"reports/nov23_transported_subclasses/result.json", dict(results=results))
        print(seed, {k:(round(v["direct"]["accuracy"]*100, 3), v["restore_only"]["fixed"], v["restore_only"]["harmed"]) for k,v in results[seed].items()}, flush=True)


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
