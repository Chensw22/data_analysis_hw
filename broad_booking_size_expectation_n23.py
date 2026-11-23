"""Learn the booking sizes suggested by individual records, including solo travellers."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_booking_canonical_n23 import REPORT, load_roster
from broad_boosting_path_n23 import calibrated
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize
from class_analogue_experts_n23 import design
from conditional_view_density_n23 import unrelated_order
from exaone_validation_data import CATEGORICAL
from released_correction_audit_n23 import development


def predict_size(frame, groups, size):
    target = np.minimum(size, 4) - 1
    result = np.full((len(frame), 4), np.nan)
    control = np.full_like(result, np.nan)
    for t, q in GroupKFold(5, shuffle=True, random_state=182).split(frame, groups=groups):
        assert not np.intersect1d(groups[t], groups[q]).size
        model = CatBoostClassifier(iterations=340, depth=5, learning_rate=.075, thread_count=2,
                                   random_seed=148, verbose=False, allow_writing_files=False)
        model.fit(frame.iloc[t], target[t], cat_features=CATEGORICAL)
        np.testing.assert_array_equal(model.classes_, np.arange(4))
        result[q] = model.predict_proba(frame.iloc[q])
        control[q] = np.bincount(target[t], minlength=4) / len(t)
    np.testing.assert_allclose(result.sum(1), 1.)
    checks = dict(logloss=float(log_loss(target, result)), frequency_logloss=float(log_loss(target, control)),
                  auc={str(k): float(roc_auc_score(size >= k, result[:, k-1:].sum(1))) for k in [2,3,4]})
    cumulative = np.column_stack([result[:, k-1:].sum(1) for k in [2,3,4]])
    features = np.column_stack([logit(cumulative.clip(.001,.999)), -np.log(result[np.arange(len(size)),target])])
    return features, checks


def reshuffle(ids, booking, size, state, base, cuts, rng):
    pred = base[ids]
    # Exact observed size, payment status, baseline decision and risk band are held fixed.
    keys = pd.DataFrame(dict(size=size[ids], state=state[ids], positive=pred>=.5, band=np.digitize(pred,cuts)))
    order = ids.copy()
    for cell in keys.groupby(list(keys)).indices.values():
        g = booking[ids[cell]]
        if len(cell) > 1 and np.unique(g, return_counts=True)[1].max() * 2 <= len(cell):
            order[cell] = ids[cell[unrelated_order(g, rng)]]
    return order


def main():
    frame, group, size, _ = load_roster()
    row, y, masks = historical()
    labels, releases, _ = development()
    np.testing.assert_array_equal(labels, y)
    aux, pretext = predict_size(frame, group, size)
    tickets = pd.concat([pd.read_csv(ROOT/f'data/raw/{s}.csv',usecols=['PassengerId'])
                         for s in ('train','test')],ignore_index=True).PassengerId
    slot = tickets.str.split('_').str[1].astype(int).to_numpy()[row]
    assert (slot <= size[row]).all()
    g, size = group[row], size[row]
    extra = aux[row]
    state = strata(frame.iloc[row])
    masks = {**masks, 'solo_repeat':masks['repeat'] & (size == 1),
             'paid_transported':masks['repeat'] & (state == 2) & y,
             'zero_stayed':masks['repeat'] & (state == 0) & ~y}
    x = frame.iloc[row].reset_index(drop=True)
    out = ART / 'nov23_booking_size_expectation'
    out.mkdir(exist_ok=True)
    history, refs, plans = {}, [], {}
    for seed in [*map(str, SEEDS), 'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as cache:
            np.testing.assert_array_equal(cache['row'], row)
            np.testing.assert_array_equal(cache['y'], y)
            ref, base, fold = cache['ref'], cache['base'], cache['fold']
        np.testing.assert_array_equal(ref, releases[131] if seed == 'v131' else base >= .5)
        modes = ['observed','expected','shuffled']
        predictions = {k:ref.copy() for k in modes}
        fraction = []
        for f in np.unique(fold):
            ix = np.flatnonzero((fold == f) & (ref == (base >= .5)))
            for a, b in GroupKFold(3, shuffle=True, random_state=925).split(ix, groups=g[ix]):
                t, q = ix[a], ix[b]
                assert not np.intersect1d(g[t],g[q]).size
                _, own = design(x, t)
                core = np.c_[logit(base.clip(.002,.998)), own, np.log(size), np.log(slot), slot / size]
                signed = np.where(base >= .5, 1., -1.)
                blocks = np.c_[extra, extra * signed[:,None], extra * (size==1)[:,None]]
                cuts = np.quantile(base[t], np.arange(1,5)/5)
                rng = np.random.default_rng(581)
                shuffled_train, shuffled_query = [reshuffle(v,g,size,state,base,cuts,rng) for v in (t,q)]
                fraction.append(float((shuffled_query != q).mean()))
                for mode in modes:
                    train, query = core[t], core[q]
                    if mode != 'observed':
                        ti, qi = (t, q) if mode == 'expected' else (shuffled_train, shuffled_query)
                        train, query = np.c_[train,blocks[ti]], np.c_[query,blocks[qi]]
                    predictions[mode][q] = calibrated(train,query,y,t) >= .5
        anchor = ref == y
        if seed != 'v131':
            refs.append(anchor)
        metrics = {}
        for mode, vote in predictions.items():
            hit = vote == y
            metrics[mode] = dict(accuracy=float(hit.mean()), fixed=int((hit & ~anchor).sum()), harmed=int((anchor & ~hit).sum()))
            if seed != 'v131':
                history.setdefault(mode, []).append(hit)
        plans[seed] = dict(models=metrics, fraction_shuffled=float(np.mean(fraction)))
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,base=base,ref=ref,fold=fold,aux=extra,**predictions)
        print(seed,metrics,flush=True)
    write(REPORT/'size_expectation.json',dict(pretext=pretext,plans=plans,grouped={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in history.items()}))


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
