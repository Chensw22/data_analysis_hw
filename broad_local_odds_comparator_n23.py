"""Estimate absolute outcome odds by comparing passengers with similarly sized service bills."""
import numpy as np
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load, strata
from broad_geographic_transfer_n23 import BILLS
from broad_error_vote_tables_n23 import bank
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, evaluate, write
from broad_passenger_transport_n23 import manifest_coordinates
from released_correction_audit_n23 import development

REPORT = ROOT / "reports/nov23_local_odds_comparator"


def neighbors(bills, groups, query, pool, k):
    # Randomize tied bills without looking at labels, especially all-zero baskets.
    pool = np.random.default_rng(879).permutation(pool)
    blocks = []
    for q in np.array_split(query, max(1, int(np.ceil(len(query)/256)))):
        dist = cdist(bills[q], bills[pool], "sqeuclidean")
        dist[groups[q,None] == groups[pool]] = np.inf
        order = np.argsort(dist, axis=1, kind="stable")[:, :k]
        assert np.isfinite(np.take_along_axis(dist, order, axis=1)).all()
        blocks.append(pool[order])
    return np.concatenate(blocks)


def model():
    return CatBoostClassifier(iterations=180, depth=5, learning_rate=.075, thread_count=2,
                              verbose=False, random_seed=951, allow_writing_files=False)


def contrasts(x, left, right):
    return np.column_stack([x[left] - x[right], (x[left] + x[right]) * .5])


def absolute_logodds(differences, labels):
    # If the query's log odds are u, its j-th comparator has log odds u-d_j.
    # Match the observed number of positive reference outcomes, with one
    # positive and one negative pseudo-observation at zero difference.
    count = labels.sum(1) + 1
    lo, hi = np.full(len(count), -20.), np.full(len(count), 20.)
    for _ in range(28):
        mid = .5 * (lo + hi)
        below = expit(mid[:,None] - differences).sum(1) + 2*expit(mid) < count
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return .5 * (lo + hi)


def compare(x, y, training_left, training_right, query, refs):
    classifier = model()
    both = np.r_[training_left, training_right]
    reverse = np.r_[training_right, training_left]
    classifier.fit(contrasts(x, both, reverse), y[both])
    odds = []
    for q, block in zip(np.array_split(query, 12), np.array_split(refs, 12)):
        left, right = np.repeat(q, block.shape[1]), block.ravel()
        def raw(a,b):
            return classifier.predict(contrasts(x,a,b), prediction_type="RawFormulaVal", thread_count=2)
        # Averaging with the reversed comparison makes d(i,j) equal -d(j,i).
        margin = .5 * (raw(left,right)-raw(right,left)).reshape(block.shape)
        odds.append(absolute_logodds(margin, y[block]))
    return np.concatenate(odds)


def run():
    row,y,frame,group,_ = load()
    labels,deployed,masks = development()
    np.testing.assert_array_equal(y,labels)
    miss = np.logical_and.reduce([p!=y for p in deployed.values()])
    state = strata(frame)
    masks = {**masks,"release_missed":miss,"paid_transported":miss & y & (state==2),
             "zero_stayed":miss & ~y & (state==0)}
    assert [masks[k].sum() for k in ("release_missed","paid_transported","zero_stayed")]==[941,415,281]
    amounts = frame[BILLS].to_numpy()
    bills = np.c_[np.log1p(np.nan_to_num(amounts))/3, amounts>0, np.isnan(amounts)]
    register = manifest_coordinates(frame.drop(columns=BILLS))
    full = np.column_stack([bills,register])
    records,history = {},{}
    for name in [*map(str,SEEDS),"v131"]:
        fold,base,ref,_,_ = bank(name,row,y)
        scores = {}
        for f in np.unique(fold):
            q = np.flatnonzero(fold==f)
            train = np.flatnonzero((fold!=f) & ~np.isin(group,group[q]))
            # Hold out reference outcomes from training the comparison model.
            fit,reference = next(GroupKFold(3,shuffle=True,random_state=591).split(train,groups=group[train]))
            fit,reference = train[fit],train[reference]
            assert not np.intersect1d(group[fit],group[reference]).size
            assert not np.intersect1d(group[train],group[q]).size
            near = neighbors(bills,group,fit,fit,28)
            a,b = np.repeat(fit,near.shape[1]),near.ravel()
            discord = y[a]!=y[b]
            refs = neighbors(bills,group,q,reference,64)
            native = {"donors": logit((1+y[refs].sum(1))/(2+refs.shape[1]))}
            for key,x in (("bill_comparator",bills),("full_comparator",full)):
                native[key] = compare(x,y,a[discord],b[discord],q,refs)
            native["direct"] = model().fit(full[train],y[train]).predict(full[q],prediction_type="RawFormulaVal")
            for key,z in native.items():
                scores.setdefault(key,np.full(len(y),np.nan))[q] = expit(z)
            o = logit(base[q].clip(.002,.998))
            # Evaluate additional evidence on different bookings within this
            # untouched outer fold. All candidate scores are label-free here.
            for ia,ib in GroupKFold(3,shuffle=True,random_state=925).split(q,groups=group[q]):
                assert not np.intersect1d(group[q[ia]],group[q[ib]]).size
                modes = {"base":o[:,None], "direct":np.c_[o,native["direct"]],
                         "matched":np.c_[o,native["donors"],native["bill_comparator"],native["full_comparator"]]}
                for key,x in modes.items():
                    p = LogisticRegression(C=.5,max_iter=500).fit(x[ia],y[q[ia]]).predict_proba(x[ib])[:,1]
                    scores.setdefault("update_"+key,np.full(len(y),np.nan))[q[ib]] = p
        records[name],hits = evaluate(y,base,ref,scores,fold,name=="v131",masks)
        if name!="v131":
            for key,(hit,_) in hits.items():
                history.setdefault(key,[]).append(hit)
        dest = ART/"nov23_local_odds_comparator"
        dest.mkdir(exist_ok=True)
        np.savez_compressed(dest/f"{name}.npz",row=row,y=y,fold=fold,ref=ref,base=base,**scores)
        print(name,[(k,r["fixed"],r["harmed"]) for k,r in records[name].items()],flush=True)
    stable = {k:{s:int((np.logical_and.reduce(h)&m).sum()) for s,m in masks.items()} for k,h in history.items()}
    write(REPORT/"results.json",dict(plans=records,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
