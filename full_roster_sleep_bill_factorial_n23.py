"""Cross availability of sleep and service bills and test the remaining shared errors."""
import json

import numpy as np
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from broad_geographic_transfer_n23 import BILLS
from broad_booking_policy_replay_n23 import interval
from broad_tree_field_dividends_n23 import hard_cohorts
from exaone_validation_data import CATEGORICAL, compact_features
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, data
from full_roster_consensus_ranking_n23 import half_bank
from full_roster_metric_adjudication_n23 import load_scores
from full_roster_neighborhood_disputes_n23 import DEST, NAMES


VIEWS = {'without_bills': BILLS, 'without_sleep': ['CryoSleep'], 'without_both': ['CryoSleep', *BILLS]}


def fit_views(frame, y, groups, fold):
    predictions = {}
    for name, exclude in VIEWS.items():
        x = frame.drop(columns=exclude)
        p = np.empty(len(y))
        for f in np.unique(fold):
            t, q = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            assert not np.isin(groups[t], groups[q]).any()
            model = CatBoostClassifier(iterations=350, depth=6, learning_rate=.065, random_seed=1923,
                                       verbose=False, allow_writing_files=False, thread_count=3)
            model.fit(x.iloc[t], y[t].astype(int), cat_features=[k for k in CATEGORICAL if k in x])
            p[q] = model.predict_proba(x.iloc[q])[:, 1]
        predictions[name] = p
        print('completed', name, len(np.unique(fold)), 'fits', flush=True)
    return np.column_stack(list(predictions.values()))


def run():
    raw, y, groups, fold = data()
    full = np.column_stack([load_scores(raw, y, fold, OUT/f'{k}.npz',['p'])[:, 0] for k in (*FAMILIES, 'mitra96')])
    half = half_bank(raw, y, fold)
    geography = load_scores(raw,y,fold,DEST/'full.npz',NAMES)
    base = full[:, -1] >= .5
    unanimous = ((full >= .5) == base[:, None]).all(1)
    same = unanimous & ((geography[:, [0,2]] >= .5) == base[:, None]).all(1)
    remained = same & (base != y)
    assert remained.sum() == 819
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    hard = np.zeros(len(y), bool)
    hard[dev] = hard_cohorts(dev, y[dev])[2]
    frame = compact_features(raw)
    fv, hv = [fit_views(frame, y, groups, split) for split in (fold, fold % 2)]
    root = ROOT/'artifacts/nov23_sleep_bill_factorial'
    root.mkdir(exist_ok=True)
    np.savez_compressed(root/'oof.npz', passenger=raw.PassengerId.to_numpy(str), y=y, fold=fold, full=fv, half=hv)
    decisions = {k: fv[:,i] >= .5 for i,k in enumerate(VIEWS)}
    decisions['two_of_three'] = (fv >= .5).sum(1) >= 2
    decisions['all_three_contrary'] = np.where(((fv >= .5) != base[:,None]).all(1), ~base, base)
    for name in ['original_scores', 'joint', 'joint_on_agreement']:
        predictions = base.copy()
        for f in np.unique(fold):
            t = (fold%2 == f%2) & (fold != f)
            q = fold == f
            if name == 'joint_on_agreement':
                # Restrict by the five observable votes. Error labels never define an operational group.
                t &= ((half >= .5) == (half[:, -1:] >= .5)).all(1)
                q &= unanimous
            tx, qx = half[t], full[q]
            if name != 'original_scores':
                tx, qx = np.column_stack([tx, hv[t]]), np.column_stack([qx, fv[q]])
            assert not np.isin(groups[q], groups[t]).any()
            model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500))
            model.fit(logit(tx.clip(.001,.999)), y[t])
            predictions[q] = model.predict_proba(logit(qx.clip(.001,.999)))[:,1] >= .5
        decisions[name] = predictions
    report = {}
    for name, pred in decisions.items():
        fixed, harm = (pred == y) & (base != y), (pred != y) & (base == y)
        delta = fixed.astype(int)-harm
        report[name] = dict(accuracy=float((pred == y).mean()), fixed=int(fixed.sum()), harmed=int(harm.sum()),
            residual_fixed=int((fixed & remained).sum()), matched_correct_harmed=int((harm & same).sum()),
            common_fixed=int((fixed & unanimous).sum()), historical_fixed=int((fixed & hard).sum()),
            additional_net=int(delta.sum()-delta[dev].sum()), fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)],
            ci_pp=interval(delta,groups))
    any_hit = ((fv >= .5) == y[:,None]).any(1)
    report['coverage'] = dict(errors=int(remained.sum()), comparable_correct=int((same & ~remained).sum()),
        can_recover=int((any_hit & remained).sum()), still_missed=int((~any_hit & remained).sum()),
        correct_challenged_by_any=int((same & (base == y) & ((fv >= .5) != y[:,None]).any(1)).sum()))
    path = ROOT/'reports/nov23_matched_learning_curve/sleep_bill_factorial.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    run()
