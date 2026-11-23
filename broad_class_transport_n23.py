"""Classify passengers by averaging normalized class-conditional densities over all spanning trees."""
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel
from broad_error_scope_balance_n23 import archived_votes
from broad_bill_exception_n23 import strata
from broad_geographic_transfer_n23 import BILLS, load
from released_correction_audit_n23 import development
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize

REPORT = ROOT / "reports/nov23_class_transport"


def encode(frame):
    cuts = {k: [0, 50, 250, 500, 750, 1000, 1500, 3000] for k in BILLS}
    cuts.update(Age=[5, 12, 18, 25, 35, 50, 65], BookingSize=[1, 2, 3, 5],
                CabinNumber=list(range(100, 2000, 150)))
    cols = []
    for k, s in frame.items():
        if k in cuts:
            vals = np.searchsorted(cuts[k], s.to_numpy(), side="left")
            vals[s.isna()] = len(cuts[k])+1
        else:
            vals = pd.factorize(s, sort=True)[0]
        assert (vals >= 0).all()
        cols.append(vals)
    return np.column_stack(cols)


def counts(x, rows, widths):
    # Uniform cell prior has the same one-field marginal for every pair.
    single = [(np.bincount(x[rows, i], minlength=n)+1/n)/(len(rows)+1)
              for i, n in enumerate(widths)]
    pairs = {}
    for i, j in zip(*np.triu_indices(len(widths), 1)):
        ni, nj = widths[i], widths[j]
        cell = x[rows, i]*nj + x[rows, j]
        table = np.bincount(cell, minlength=ni*nj).reshape(ni, nj)
        pairs[i, j] = (table+1/(ni*nj))/(len(rows)+1)
    return single, pairs


def log_tree_sum(w):
    lap = -w.copy()
    diag = np.arange(w.shape[-1])
    lap[..., diag, diag] = w.sum(-1)
    sign, val = np.linalg.slogdet(lap[..., :-1, :-1])
    assert np.all(sign == 1)
    return val


def scores(x, y, t, q):
    widths = x.max(0)+1
    singles, pairs = counts(x, t, widths)
    d = x.shape[1]
    weights = np.zeros((d,d))
    for (i,j), joint in pairs.items():
        mi = (joint*np.log(joint/np.outer(singles[i], singles[j]))).sum()
        weights[i,j] = weights[j,i] = np.exp(8*mi)
    tree = minimum_spanning_tree(-weights).toarray() != 0
    tree = tree | tree.T
    assert tree.sum() == 2*(d-1)
    normalizer = log_tree_sum(weights)
    logs = []
    for label in (False, True):
        train = t[y[t] == label]
        own, joint = counts(x, train, widths)
        fraction = (len(train)+1)/(len(train)+31)
        smooth = [fraction*p + (1-fraction)*common for p,common in zip(own,singles)]
        independent = sum(np.log(p[x[q,i]]) for i,p in enumerate(smooth))
        single_tree = independent.copy()
        edges = np.zeros((len(q),d,d))
        for (i,j), table in joint.items():
            table = fraction*table + (1-fraction)*pairs[i,j]
            assert np.allclose(table.sum(0), smooth[j]) and np.allclose(table.sum(1),smooth[i])
            ratio = (table/np.outer(smooth[i],smooth[j]))[x[q,i],x[q,j]]
            if tree[i,j]:
                single_tree += np.log(ratio)
            edges[:,i,j] = edges[:,j,i] = weights[i,j]*ratio
        mixture = independent + log_tree_sum(edges) - normalizer
        logs.append(np.stack([independent,single_tree,mixture],1)+np.log(len(train)/len(t)))
    return logs[1]-logs[0]


def main():
    row,y,frame,groups,_ = load()
    x = encode(frame)
    ids, targets, masks = historical()
    np.testing.assert_array_equal(ids,row)
    np.testing.assert_array_equal(targets,y)
    all_hit, base_hit = {}, []
    for s in SEEDS:
        with np.load(ART / f"nov23_group_kernel/{s}.npz") as z:
            np.testing.assert_array_equal(z["row"], row)
            np.testing.assert_array_equal(z["y"], y)
            fold, base = z["fold"], z["base"]
        z = np.full((len(y),3), np.nan)
        for f in np.unique(fold):
            t,q = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            assert not np.intersect1d(groups[t],groups[q]).size
            z[q] = scores(x,y,t,q)
        assert np.isfinite(z).all()
        methods = {}
        for i, name in enumerate(("independent","single_tree","tree_mixture")):
            methods[name] = expit(z[:,i])
            pred = np.full(len(y),np.nan)
            features = np.column_stack([logit(base.clip(.001,.999)), z[:,i]])
            for f in np.unique(fold):
                rows = np.flatnonzero(fold == f)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(rows,groups=groups[rows]):
                    t,q = rows[a],rows[b]
                    assert not np.intersect1d(groups[t],groups[q]).size
                    pred[q] = LogisticRegression(C=.5).fit(features[t],y[t]).predict_proba(features[q])[:,1]
            methods["combined_"+name] = pred
        out = ART / "nov23_class_transport"
        out.mkdir(exist_ok=True)
        np.savez_compressed(out/f"{s}.npz", row=row,y=y,fold=fold,base=base,**methods)
        base_hit.append((base>=.5)==y)
        for k,p in methods.items():
            assert np.isfinite(p).all()
            all_hit.setdefault(k,[]).append((p>=.5)==y)
    stats = {k:summarize(np.stack(v),np.stack(base_hit),masks) for k,v in all_hit.items()}
    write(REPORT/"results.json",stats)
    archive_rows, labels, bases, archive, *_ = panel()
    np.testing.assert_array_equal(archive_rows,row)
    np.testing.assert_array_equal(labels,y)
    votes = archived_votes(row,y,bases,archive)
    prior = (votes == y[None,:,None])
    labels, releases, _ = development()
    np.testing.assert_array_equal(labels,y)
    missed = np.logical_and.reduce([v != y for v in releases.values()])
    state = strata(frame)
    core = missed & (((state==2)&y) | ((state==0)&~y))
    cohorts = {**masks, "repeat_outside":masks["repeat"] & ~core}
    stable = {k:np.logical_and.reduce(v) for k,v in all_hit.items()}
    stable["any_native"] = stable["tree_mixture"] | stable["single_tree"] | stable["independent"]
    rescue = {k:{name:dict(stable=int((m&h).sum()),
                           beyond_67=int((m&h&~prior.all(0).any(1)).sum()),
                           previously_never_right=int((m&h&~prior.any((0,2))).sum()))
                  for name,h in stable.items()} for k,m in cohorts.items()}
    write(REPORT/"coverage.json",rescue)
    print(rescue["repeat"],rescue["repeat_outside"])


if __name__ == "__main__":
    with threadpool_limits(2):
        main()
