"""Test whether paid solo passengers transported against unanimous predictions can be identified."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from audit_stable_rescue_placebo_n23 import family_scores
from broad_geographic_transfer_n23 import BILLS, load
from broad_partition_evidence_n23 import SEEDS, write
from broad_tree_field_dividends_n23 import hard_cohorts
from exaone_validation_data import CATEGORICAL
from solo_dissent_neighbors_n23 import REPORT


def run():
    row, y, frame, groups, _ = load()
    _, _, hard, _ = hard_cohorts(row, y)
    fees = frame[BILLS]
    amounts = np.log1p(fees.fillna(0).to_numpy())
    own = np.column_stack([amounts, fees.gt(0), fees.isna(), frame.Age.fillna(-1), frame.CabinNumber.fillna(-1),
                           pd.get_dummies(frame[CATEGORICAL], dtype=float)])
    paid_solo = (frame.BookingSize.to_numpy() == 1) & fees.gt(0).any(axis=1).to_numpy()
    history, populations, plans = [], [], {}
    for seed in [*map(str, SEEDS), 'v131']:
        fold, base, ref, families = family_scores(seed, row, y, frame)
        scores = np.column_stack(list(families.values()))
        scope = paid_solo & ~ref & (scores < .5).all(1)
        ids = np.flatnonzero(scope)
        assert len(np.unique(groups[ids])) == len(ids)
        features = np.column_stack([logit(base.clip(.002, .998)), own])
        prob = np.full(len(y), np.nan)
        top = np.zeros(len(y), bool)
        for f in np.unique(fold):
            bucket = ids[fold[ids] == f]
            for a, b in GroupKFold(3, shuffle=True, random_state=782).split(bucket, groups=groups[bucket]):
                train, query = bucket[a], bucket[b]
                assert not np.intersect1d(groups[train], groups[query]).size
                model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=300))
                model.fit(features[train], y[train])
                prob[query] = model.predict_proba(features[query])[:, 1]
                # Predefined diagnostic budget, always report both corrections and damage.
                n = max(1, len(query)//10)
                top[query[np.argsort(-prob[query], kind='stable')[:n]]] = True
        np.testing.assert_array_equal(np.isfinite(prob), scope)
        flip = prob >= .5
        plans[seed] = dict(people=len(ids), old_errors=int(y[ids].sum()), old_correct=int((~y[ids]).sum()),
                           risk_auc=float(roc_auc_score(y[ids], base[ids])),
                           fitted_auc=float(roc_auc_score(y[ids], prob[ids])),
                           fixed=int((flip & y).sum()), harmed=int((flip & ~y).sum()),
                           highest_tenth_fixed=int((top & y).sum()), highest_tenth_harmed=int((top & ~y).sum()))
        if seed != 'v131':
            populations.append(scope)
            history.append(flip)
    shared = np.logical_and.reduce(populations)
    target = hard & shared & y
    assert target.sum() == 86
    hits = np.stack(history)
    out = dict(plans=plans, stable_scope=int(shared.sum()), stable_scope_transported=int((shared & y).sum()),
               stable_scope_not_transported=int((shared & ~y).sum()), hard_target=int(target.sum()),
               target_ever_fixed=int((hits.any(0) & target).sum()), target_stably_fixed=int((hits.all(0) & target).sum()))
    write(REPORT/'unanimous_payers.json', out)
    print(out)


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
