"""Can another booking identify when a population-wide analog beats the incumbent?"""
import numpy as np
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load, strata
from broad_error_vote_tables_n23 import bank
from broad_partition_evidence_n23 import SEEDS
from broad_passenger_transport_n23 import manifest_coordinates, kernel
from broad_release_transfer_n23 import REPORT, write
from released_correction_audit_n23 import development


def score_options(x, base_features, rewards, train, test):
    constant = np.full(len(test), rewards[train].mean())
    def linear(features):
        model = LogisticRegression(C=.25, max_iter=500)
        return model.fit(features[train], rewards[train]).predict_proba(features[test])[:, 1]
    model = CatBoostClassifier(iterations=140, depth=4, learning_rate=.04, verbose=False,
                               thread_count=2, random_seed=341, allow_writing_files=False)
    tree = model.fit(x[train], rewards[train]).predict_proba(x[test])[:, 1]
    return {"constant": constant, "confidence": linear(base_features), "linear": linear(x), "tree": tree}


def run():
    row, y, frame, groups, _ = load()
    labels, deployed, masks = development()
    np.testing.assert_array_equal(labels, y)
    missed = np.logical_and.reduce([p != y for p in deployed.values()])
    masks = {**masks, "release_missed": missed, "paid_transported": missed & y & (strata(frame)==2),
             "zero_not_transported": missed & ~y & (strata(frame)==0)}
    assert [masks[k].sum() for k in ("release_missed","paid_transported","zero_not_transported")] == [941,415,281]
    roster = manifest_coordinates(frame)
    records, saved = {}, {}
    for panel in [*map(str, SEEDS), "v131"]:
        fold, base, ref, _, _ = bank(panel, row, y)
        outputs = {}
        for outer in np.unique(fold):
            ix = np.flatnonzero(fold==outer)
            support = np.flatnonzero((fold!=outer) & ~np.isin(groups, groups[ix]))
            assert not np.intersect1d(groups[support],groups[ix]).size
            weight = kernel(roster[ix], roster[support])
            analog = weight @ y[support].astype(float)
            pick = analog >= .5
            conflict = pick != ref[ix]
            base_logit = logit(base[ix].clip(.001,.999))
            confidence = np.column_stack([abs(base_logit), abs(analog-.5), ref[ix]])
            # A squared-weight sum captures how many independent passenger rows contribute.
            concentration = np.asarray(weight.multiply(weight).sum(axis=1)).ravel()
            x = np.column_stack([confidence, concentration, roster[ix], base_logit, analog])
            rewards = pick == y[ix]
            policies = {"analog":pick.copy()}
            for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=groups[ix]):
                train,test = a[conflict[a]], b[conflict[b]]
                assert not np.intersect1d(groups[ix[train]], groups[ix[test]]).size
                for key, prob in score_options(x,confidence,rewards,train,test).items():
                    candidate = policies.setdefault(key,ref[ix].copy())
                    candidate[test] = np.where(prob>.5,pick[test],ref[ix[test]])
            for key,v in policies.items():
                outputs.setdefault(key,ref.copy())[ix] = v
        records[panel] = {}
        for key,v in outputs.items():
            hit = v==y
            fixed, harmed = hit & (ref!=y), ~hit & (ref==y)
            records[panel][key] = dict(accuracy=float(hit.mean()),fixed=int(fixed.sum()),harmed=int(harmed.sum()),
                    cohort_correct={s:int((hit&m).sum()) for s,m in masks.items()},
                    net_by_fold=[int(fixed[fold==i].sum()-harmed[fold==i].sum()) for i in np.unique(fold)])
            if panel != "v131":
                saved.setdefault(key,[]).append(hit)
        print(panel,[(k,r["fixed"],r["harmed"]) for k,r in records[panel].items()],flush=True)
    stable = {k:{s:int((np.logical_and.reduce(h)&m).sum()) for s,m in masks.items()} for k,h in saved.items()}
    write(REPORT / "neighbor_arbitration.json",dict(plans=records,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
