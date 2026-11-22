"""Find stable complementary classifiers and test a booking-held-out policy for choosing one."""
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from scipy.special import logit

from archive_error_recovery_n23 import ART, ROOT, SEEDS, panel, predictions
from exaone_validation_data import compact_features, CATEGORICAL
from fresh_family_rescue_n23 import NEW
from relational_error_corrector import write


EXTRA = {**NEW, "native_carte": ("carte", "p")}
# The routing experiment only uses direct fits with training labels outside the
# entire first-stage validation fold. The retrospective census has a wider scope.
ACTIONS = {
    "feature_projections": ("projection_shapes", "projection_shapes"),
    "orion": EXTRA["orion"], "variational_gp": EXTRA["variational_gp"],
    "native_carte": EXTRA["native_carte"], "manifest_likelihood": ("manifest_tree", "manifest_tree"),
    "class_analogues": ("class_analogues", "class_linear"), "field_factors": ("field_factors", "order3"),
    "registration_only": ("bill_matched_register", "register"), "budget_prior": ("simplex_likelihood", "budget_prior"),
    "bayesian_forest": ("bart_long", "bart"),
}
OUT = ROOT / "reports/nov23_broad_booking_policy"


def census(y, base, bank, names):
    hit = (bank >= .5) == y[None, :, None]
    ref = (base >= .5) == y
    near = (hit.mean(1) >= ref.mean(1)[:, None] - .01).all(0)
    stable = hit.all(0)
    scope = {}
    for title, mask in {"ever_wrong": ~ref.all(0), "persistent": ~ref.any(0)}.items():
        section = {}
        for pool, available in {"all": np.arange(len(names)), "within_one_point": np.flatnonzero(near)}.items():
            bits = [int.from_bytes(np.packbits(stable[mask, j]).tobytes(), "big") for j in available]
            top = {}
            for k in (1, 2, 3):
                def covered(group):
                    union = 0
                    for j in group:
                        union |= bits[j]
                    return union.bit_count()
                best = max(combinations(range(len(bits)), k), key=covered)
                top[k] = dict(models=[names[available[i]] for i in best], people=covered(best))
            top["any"] = int(stable[mask][:, available].any(1).sum())
            section[pool] = top
        scope[title] = dict(people=int(mask.sum()), pools=section)
    # The triple is selected retrospectively. Count how many initially correct
    # passengers present the very same opportunity to override the reference.
    trio = scope["persistent"]["pools"]["all"][3]["models"]
    cols = [names.index(n) for n in trio]
    dissent = ((bank[:, :, cols] >= .5) != (base >= .5)[:, :, None]).any(2)
    opportunity = dict(models=trio, recoverable=(dissent & ~ref).sum(1).tolist(),
                       at_risk=(dissent & ref).sum(1).tolist())
    return dict(scope=scope, dissent_opportunity=opportunity, within_one_point=[names[i] for i in np.flatnonzero(near)])


def fit_partitions(x, reward, rows, depth=2):
    gain = reward[rows].sum(0)
    score = gain.max()
    if depth == 0:
        return None
    chosen = None
    for j in range(x.shape[1]):
        values = x[rows, j]
        for cut in np.unique(np.quantile(values, [.2, .4, .5, .6, .8])):
            side = values <= cut
            if min(side.sum(), (~side).sum()) < 75:
                continue
            left = reward[rows[side]].sum(0)
            value = left.max() + (gain-left).max()
            if value > score:
                chosen, score = (j, cut, side), value
    if chosen is None:
        return None
    j, cut, side = chosen
    return (j, cut, fit_partitions(x, reward, rows[side], depth-1),
            fit_partitions(x, reward, rows[~side], depth-1))


def assign(node, x, ids, leaf=1):
    if node is None:
        return [(leaf, ids)]
    j, cut, left, right = node
    side = x[ids, j] <= cut
    return assign(left, x, ids[side], 2*leaf) + assign(right, x, ids[~side], 2*leaf+1)


def apply_policy(x, reward, votes, grow, estimate, test):
    tree = fit_partitions(x, reward, grow)
    decision = np.empty(len(test), bool)
    action_counts = Counter()
    for (_, a), (_, b) in zip(assign(tree, x, estimate), assign(tree, x, test)):
        diff = reward[a] - reward[a, :1]
        # One standard error of the paired net gain favors retaining the reference.
        score = diff.sum(0) - np.sqrt((diff**2).sum(0))
        action = int(score.argmax())
        decision[np.searchsorted(test, b)] = votes[b, action]
        action_counts[action] += len(b)
    return decision, action_counts


def summarize(hit, ref, repeat):
    return dict(accuracy=hit.mean(1).tolist(), fixed=(hit & ~ref).sum(1).tolist(),
                harmed=(~hit & ref).sum(1).tolist(), stable_ever=int((hit.all(0) & ~ref.all(0)).sum()),
                stable_persistent=int((hit.all(0) & repeat).sum()))


def run():
    row, y, base, old, _, names, _ = panel()
    raw = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c not in ("Name", "Transported")).iloc[row]
    groups = raw.PassengerId.str.split("_").str[0].to_numpy()
    attrs = pd.get_dummies(compact_features(raw), columns=CATEGORICAL).fillna(-1).to_numpy(float)
    extra = np.stack([predictions(row, y, s, EXTRA)[1] for s in SEEDS])
    archive = census(y, base, np.concatenate([old, extra], axis=2), names + list(EXTRA))
    write(OUT / "coverage.json", archive)
    print(archive["scope"], flush=True)
    hits = {}
    selected = Counter()
    baseline = (base >= .5) == y
    for i, seed in enumerate(SEEDS):
        b, p, folds = predictions(row, y, seed, ACTIONS)
        np.testing.assert_array_equal(b, base[i])
        scores = np.column_stack([b, p])
        votes = scores >= .5
        reward = (votes == y[:, None]).astype(int)
        x = np.column_stack([attrs, scores, abs(b-.5), p.std(1), votes[:, 1:].mean(1)])
        logits = logit(scores.clip(.001, .999))
        decisions = {key: np.empty(len(y), bool) for key in ("honest_policy", "pooled_policy", "linear_scores")}
        seen = np.zeros(len(y), int)
        for outer in np.unique(folds):
            q = np.flatnonzero(folds == outer)
            thirds = [q[j] for _, j in GroupKFold(3, shuffle=True, random_state=123).split(q, groups=groups[q])]
            for j, test in enumerate(thirds):
                a, c = thirds[(j+1)%3], thirds[(j+2)%3]
                assert not (set(groups[a]) & set(groups[c]) or set(groups[test]) & set(groups[np.r_[a,c]]))
                seen[test] += 1
                outs = []
                for grow, estimate in [(a,c), (c,a)]:
                    decision, counts = apply_policy(x, reward, votes, grow, estimate, test)
                    selected.update(counts)
                    outs.append(decision)
                # If the independent leaf choices disagree, use the incumbent.
                decisions["honest_policy"][test] = np.where(outs[0] == outs[1], outs[0], votes[test, 0])
                learn = np.r_[a,c]
                decisions["pooled_policy"][test] = apply_policy(x, reward, votes, learn, learn, test)[0]
                fit = LogisticRegression(C=.1, max_iter=400).fit(logits[learn], y[learn])
                decisions["linear_scores"][test] = fit.predict(logits[test])
        assert (seen == 1).all()
        for name, d in decisions.items():
            hits.setdefault(name, []).append(d == y)
    result = {k:summarize(np.stack(v), baseline, ~baseline.any(0)) for k,v in hits.items()}
    actions = ["baseline", *ACTIONS]
    write(OUT / "policy.json", dict(n=len(row), seeds=list(SEEDS), scores=result,
          honest_branch_passengers={actions[j]:n for j,n in selected.items()}))
    print(result, flush=True)


if __name__ == "__main__":
    run()
