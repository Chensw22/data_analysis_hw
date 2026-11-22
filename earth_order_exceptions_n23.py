"""Learn to separate cross-booking outcomes that contradict the usual spending order."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.special import logit
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS
from class_analogue_experts_n23 import design
from earth_bill_order_n23 import BILLS, dominated_by
from exaone_validation_data import compact_features
from relational_error_corrector import write


def matches(rows, y, x, missing, group, rng):
    pairs = []
    for mask in np.unique(missing[rows]):
        part = rows[missing[rows] == mask]
        a = rng.permutation(part[y[part]])
        b = rng.permutation(part[~y[part]])
        if a.size and b.size:
            edges = dominated_by(x[a], x[b]) & (group[a, None] != group[b])
            m = maximum_bipartite_matching(csr_matrix(edges), perm_type="column")
            pairs.extend(zip(a[m >= 0], b[m[m >= 0]]))
    return np.asarray(pairs, dtype=int)


def rank(x, pairs, q):
    a, b = pairs.T
    dif = x[a] - x[b]
    reg = LogisticRegression(C=.025, fit_intercept=False, max_iter=250)
    reg.fit(np.vstack([dif, -dif]), np.repeat([1, 0], len(a)))
    return reg.decision_function(x[q])


def order_accuracy(score, pairs, bills):
    a, b = pairs.T
    tied = (bills[a] == bills[b]).all(1)
    diff = score[a]-score[b]
    return {key:dict(rows=int(m.sum()), accuracy=float(np.mean((diff[m]>0)+.5*(diff[m]==0))))
            for key,m in [("all",np.ones(len(a),bool)),("same_bills",tied),("ordered_bills",~tied)]}


def run():
    with np.load(ART/f"nov23_group_kernel/{SEEDS[0]}.npz") as z:
        row,y = z["row"],z["y"]
    full = pd.read_csv(ROOT/"data/raw/train.csv", usecols=lambda c: c not in {"Name","Transported"})
    group = full.PassengerId.str.split("_").str[0]
    amounts = np.log1p(full[BILLS])
    peers = amounts.groupby(group).transform("mean").fillna(0).to_numpy()[row]/6
    size = group.map(group.value_counts()).to_numpy()[row]
    group = group.to_numpy()[row]
    raw = full.iloc[row].reset_index(drop=True)
    frame = compact_features(raw)
    earth = raw.HomePlanet.eq("Earth").to_numpy()
    bills = raw[BILLS].fillna(0).to_numpy()*[1,1,-1,-1,-1]
    missing = raw[BILLS].isna().to_numpy() @ (2**np.arange(5))
    data = {}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["y"],y)
            np.testing.assert_array_equal(z["row"],row)
            data[seed] = z["base"],z["fold"]
    repeat = np.logical_and.reduce([(base>=.5)!=y for base,_ in data.values()]) & earth
    out = ART/"nov23_earth_order_exceptions"
    out.mkdir(exist_ok=True)
    report = {}
    for seed,(base,fold) in data.items():
        names = ["base","conflicts","same_people","random_pairs"]
        probs = {k:base.copy() for k in names}
        fold_reports = []
        for f in np.unique(fold):
            t = np.flatnonzero(earth & (fold!=f))
            q = np.flatnonzero(earth & (fold==f))
            assert not np.intersect1d(group[t],group[q]).size
            _, features = design(frame,t)
            inputs = np.column_stack([features,peers,np.log1p(size)])
            nonlinear = RBFSampler(gamma=.15,n_components=256,random_state=591).fit(inputs[t])
            x = np.column_stack([inputs,4*nonlinear.transform(inputs)])
            rng = np.random.default_rng(53)
            conflict = np.vstack([matches(t,y,bills,missing,group,rng) for _ in range(3)])
            shuffled = conflict.copy()
            for mask in np.unique(missing[conflict[:,0]]):
                ix = np.flatnonzero(missing[conflict[:,0]] == mask)
                shuffled[ix,1] = rng.permutation(conflict[ix,1])
            random = np.column_stack([rng.choice(t[y[t]],len(conflict)),rng.choice(t[~y[t]],len(conflict))])
            raw_scores = {"base":logit(base.clip(.001,.999))}
            for name,pairs in [("conflicts",conflict),("same_people",shuffled),("random_pairs",random)]:
                raw_scores[name] = np.full(len(y),np.nan)
                raw_scores[name][q] = rank(x,pairs,q)
            held_pairs = matches(q,y,bills,missing,group,rng)
            # On retrospectively selected strictly ordered contradictions, simply
            # reversing the ordering is perfect by construction.
            rank_diagnostics = {**raw_scores, "reverse_bills": -bills.sum(axis=1)}
            fold_reports.append(dict(train_pairs=len(conflict),distinct_training_people=len(np.unique(conflict)),
                         held={k:order_accuracy(s,held_pairs,bills) for k,s in rank_diagnostics.items()}))
            for a,b in GroupKFold(3,shuffle=True,random_state=338).split(q,groups=group[q]):
                assert not np.intersect1d(group[q[a]],group[q[b]]).size
                for name,s in raw_scores.items():
                    v = np.column_stack([raw_scores["base"]]) if name=="base" else np.column_stack([raw_scores["base"],s])
                    clf = LogisticRegression(C=.3).fit(v[q[a]],y[q[a]])
                    probs[name][q[b]] = clf.predict_proba(v[q[b]])[:,1]
        before = (base>=.5)==y
        scores = {}
        for name,p in probs.items():
            ok = (p>=.5)==y
            scores[name] = dict(accuracy=float(ok.mean()),fixed=int((ok&~before).sum()),harmed=int((~ok&before).sum()),
                                repeat=int((ok&repeat).sum()),fixed_misses=int((ok&~before&y).sum()),
                                fixed_false_alarms=int((ok&~before&~y).sum()))
        report[seed] = dict(folds=fold_reports,scores=scores)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,original=base,**probs)
    write(ROOT/"reports/nov23_earth_order_exceptions/result.json",report)


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
