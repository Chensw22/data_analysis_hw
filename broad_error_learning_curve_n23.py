"""Compare nested training sizes on the same unseen bookings and historical error cohorts."""
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load, strata
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from exaone_validation_data import CATEGORICAL
from released_correction_audit_n23 import development

REPORT = ROOT / "reports/nov23_error_learning_curve"
SIZES = (0.25, 0.5, 1.0)


def paired_changes(small, large, groups, masks):
    """Average repeated predictions per person before resampling whole bookings."""
    delta = (large.astype(float) - small).mean(0)
    _, code = np.unique(groups, return_inverse=True)
    counts = np.column_stack([np.bincount(code, weights=m) for m in masks.values()])
    changes = np.column_stack([np.bincount(code, weights=m * delta) for m in masks.values()])
    means = 100 * changes.sum(0) / counts.sum(0)
    rng = np.random.default_rng(4231)
    samples = []
    for _ in range(40):
        ids = rng.integers(len(counts), size=(50, len(counts)))
        samples.append(100 * changes[ids].sum(1) / counts[ids].sum(1))
    draws = np.concatenate(samples)
    def summary(value, sim):
        return dict(change_pp=float(value), interval95_pp=np.quantile(sim, [.025, .975]).tolist())
    out = {k: summary(means[i], draws[:, i]) for i, k in enumerate(masks)}
    i, j = list(masks).index("core"), list(masks).index("other")
    out["core_minus_other"] = summary(means[i]-means[j], draws[:, i]-draws[:, j])
    return out


def run():
    row, y, frame, groups, _ = load()
    labels, release, historical = development()
    np.testing.assert_array_equal(y, labels)
    miss = np.logical_and.reduce([p != y for p in release.values()])
    state = strata(frame)
    paid = miss & y & (state == 2)
    zero = miss & ~y & (state == 0)
    core = paid | zero
    masks = dict(all=np.ones(len(y), bool), miss=miss, core=core, other=~core,
                 paid=paid, zero=zero, repeat=historical["repeat"])
    assert [masks[k].sum() for k in ("miss","paid","zero","repeat")] == [941,415,281,1070]
    scores = np.full((len(SIZES),len(SEEDS),len(y)),np.nan)
    sizes, bases = [], []
    for plan, seed in enumerate(SEEDS):
        with np.load(ART / f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            fold, base = z["fold"], z["base"]
        bases.append(base >= .5)
        for f in np.unique(fold):
            q, t = np.flatnonzero(fold==f), np.flatnonzero(fold!=f)
            assert not np.intersect1d(groups[t],groups[q]).size
            order = np.random.default_rng(seed+int(f)).permutation(np.unique(groups[t]))
            previous = np.array([],int)
            for i,frac in enumerate(SIZES):
                part = t[np.isin(groups[t],order[:round(frac*len(order))])]
                assert np.isin(previous,part).all()
                previous = part
                if frac==1:
                    np.testing.assert_array_equal(part,t)
                m = CatBoostClassifier(iterations=460,depth=6,learning_rate=.06,thread_count=2,
                                       random_seed=1211,verbose=False,allow_writing_files=False)
                m.fit(frame.iloc[part], y[part], cat_features=CATEGORICAL)
                scores[i,plan,q] = m.predict_proba(frame.iloc[q])[:,1]
                sizes.append(dict(seed=int(seed),fold=int(f),fraction=frac,
                                  train_people=len(part),train_bookings=len(np.unique(groups[part]))))
        print(seed, np.round(((scores[:,plan]>=.5)==y).mean(1),4),flush=True)
    assert np.isfinite(scores).all()
    hit = (scores>=.5)==y
    ref = np.stack(bases)==y
    per_size = {}
    for i, frac in enumerate(SIZES):
        per_size[frac] = {k:dict(n=int(m.sum()),correct_by_plan=hit[i][:,m].sum(1).tolist(),
                                 stable=int(hit[i][:,m].all(0).sum()),
                                 logloss_by_plan=[float(log_loss(y[m],p[m],labels=[False,True])) for p in scores[i]],
                                 fixed_vs_base=(hit[i][:,m]&~ref[:,m]).sum(1).tolist(),
                                 harmed_vs_base=(~hit[i][:,m]&ref[:,m]).sum(1).tolist())
                          for k,m in masks.items()}
    changes = {f"{SIZES[i]}_to_{SIZES[j]}":paired_changes(hit[i],hit[j],groups,masks)
               for i,j in ((0,1),(1,2),(0,2))}
    # An existing eight-fold experiment supplies another comparison closer to
    # the full data size, with a different training recipe and fold definition.
    extra = {}
    for family in ("v44","tabicl"):
        with np.load(ART/f"nov23_matched_learning_curve/{family}.npz") as z:
            np.testing.assert_array_equal(row,z["row"])
            np.testing.assert_array_equal(y,z["y"])
            small,large = [((z[k]>=.5)==y)[None,:] for k in ("small","large")]
            extra[family] = dict(changes=paired_changes(small,large,groups,masks),sizes=z["sizes"].tolist(),
                                 core_correct=[int(small[:,core].sum()),int(large[:,core].sum())],
                                 all_correct=[int(small.sum()),int(large.sum())])
    write(REPORT/"results.json",dict(sizes=sizes,per_size=per_size,changes=changes,extended=extra))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
