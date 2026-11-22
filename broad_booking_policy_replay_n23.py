"""Re-evaluate the complementary trio and full stack on two fresh routing partitions."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from broad_booking_policy_n23 import ACTIONS, EXTRA, OUT, ROOT, SEEDS, panel, predictions
from relational_error_corrector import write


def interval(gains, groups):
    _, code = np.unique(groups, return_inverse=True)
    sizes = np.bincount(code)
    deltas = np.bincount(code, weights=gains)
    rng = np.random.default_rng(345)
    res = []
    for _ in range(50):
        bag = rng.integers(len(sizes), size=(80, len(sizes)))
        res.extend(100*deltas[bag].sum(1)/sizes[bag].sum(1))
    return np.quantile(res,[.025,.975]).tolist()


def run():
    row, y, base, old, _, old_names, _ = panel()
    groups = pd.read_csv(ROOT / "data/raw/train.csv", usecols=["PassengerId"]).PassengerId.iloc[row].str.split("_").str[0].to_numpy()
    extras = np.stack([predictions(row,y,s,EXTRA)[1] for s in SEEDS])
    bank = np.concatenate([old, extras],axis=2)
    names = old_names + list(EXTRA)
    trio = ["feature_projections","registration_only","field_factors"]
    stable = ((bank>=.5)==y[None,:,None]).all(0)
    baseline = (base>=.5)==y
    persistent = ~baseline.any(0)
    triple = persistent & stable[:,[names.index(j) for j in trio]].any(1)
    shared = persistent & ~((bank>=.5)==y[None,:,None]).any((0,2))
    outer_predictions = [predictions(row, y, s, ACTIONS) for s in SEEDS]
    output = {}
    for key, members in {"reference_calibration":{}, "trio":{n:ACTIONS[n] for n in trio},
                         "complete":ACTIONS}.items():
        scores = []
        for i, seed in enumerate(SEEDS):
            b,p,folds = outer_predictions[i]
            np.testing.assert_array_equal(b,base[i])
            columns = [list(ACTIONS).index(n) for n in members]
            x = logit(np.column_stack([b,p[:,columns]]).clip(.001,.999))
            reps = []
            for split_seed in (513, 819):
                probs = np.full(len(y),np.nan)
                for f in np.unique(folds):
                    ids = np.flatnonzero(folds==f)
                    for a,c in GroupKFold(3,shuffle=True,random_state=split_seed).split(ids,groups=groups[ids]):
                        t,q = ids[a],ids[c]
                        assert not np.intersect1d(groups[t],groups[q]).size
                        learner = LogisticRegression(C=.1,max_iter=400).fit(x[t],y[t])
                        probs[q]=learner.predict_proba(x[q])[:,1]
                assert np.isfinite(probs).all()
                reps.append(probs)
            scores.append(reps)
        score = np.asarray(scores)
        hit = (score>=.5)==y
        gain = hit.astype(float)-baseline[:,None,:]
        output[key]=dict(accuracy=hit.mean(2).tolist(),fixed=(gain>0).sum(2).tolist(),
                         harmed=(gain<0).sum(2).tolist(), mean_gain_pp=float(100*gain.mean()),
                         paired_booking_ci95_pp=interval(gain.mean((0,1)),groups),
                         stable_in_all_six_on_trio314=int((triple&hit.all((0,1))).sum()),
                         stable_in_all_six_on_shared212=int((shared&hit.all((0,1))).sum()),
                         stable_in_all_six_on_persistent=int((persistent&hit.all((0,1))).sum()))
    write(OUT/"replay.json",dict(scores=output,trio_stable_pool=int(triple.sum()),
          historical_shared=int(shared.sum()),calibration_seeds=[513,819]))
    print(output)


if __name__ == "__main__":
    run()
