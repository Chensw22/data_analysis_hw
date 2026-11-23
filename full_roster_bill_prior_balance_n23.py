"""Balance the outcome conditional on a cross-fitted spending and sleep prior, then restore odds."""
import argparse
import json

import numpy as np
from catboost import CatBoostClassifier, Pool
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS
from broad_tree_field_dividends_n23 import hard_cohorts
from exaone_validation_data import CATEGORICAL, compact_features
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, SEED, data
from full_roster_metric_adjudication_n23 import load_scores
from full_roster_neighborhood_disputes_n23 import DEST, NAMES


def model():
    return CatBoostClassifier(iterations=350, depth=6, learning_rate=.065, random_seed=1923,
                              thread_count=3, verbose=False, allow_writing_files=False)


def nested_prior(x, y, t, q, g):
    prior = np.empty(len(t))
    predictions = []
    for a, b in GroupKFold(3, shuffle=True, random_state=SEED).split(t, groups=g[t]):
        assert not np.isin(g[t[a]], g[t[b]]).any()
        fit = model().fit(x.iloc[t[a]], y[t[a]].astype(int), cat_features=['CryoSleep'])
        prior[b] = fit.predict_proba(x.iloc[t[b]])[:,1]
        predictions.append(fit.predict_proba(x.iloc[q])[:,1])
    return prior.clip(.08,.92), np.mean(predictions, axis=0).clip(.08,.92)


def save_scores(raw, y, split, probabilities, panel):
    folder = ROOT/'artifacts/nov23_bill_prior_balance'
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{panel}.npz', passenger=raw.PassengerId.to_numpy(str), y=y,
                        fold=split, **probabilities)


def predict_priors(x, cue, context, y, groups, t, q):
    p = {}
    assert not np.isin(groups[t], groups[q]).any()
    prior, query_prior = nested_prior(x[cue], y, t, q, groups)
    p['cue'] = query_prior
    w = .5/np.where(y[t], prior, 1-prior)
    for design, label in [(x,'full'), (context,'context')]:
        cat = [k for k in CATEGORICAL if k in design]
        clf = model().fit(design.iloc[t], y[t].astype(int), cat_features=cat, sample_weight=w/w.mean())
        z = clf.predict(design.iloc[q],prediction_type='RawFormulaVal')
        p['restored_'+label] = expit(z+logit(query_prior))
        if label == 'full':
            p['balanced_full'] = expit(z)
    cats = [k for k in CATEGORICAL if k in context]
    train = Pool(context.iloc[t], label=y[t].astype(int), baseline=logit(prior), cat_features=cats)
    query = Pool(context.iloc[q], baseline=logit(query_prior), cat_features=cats)
    p['offset_context'] = model().fit(train).predict_proba(query)[:,1]
    return p


def fit_priors(x, cue, context, y, groups, fold):
    methods = ['cue', 'balanced_full', 'restored_full', 'restored_context', 'offset_context']
    p = {name: np.empty(len(y)) for name in methods}
    for f in np.unique(fold):
        t, q = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        for name, score in predict_priors(x, cue, context, y, groups, t, q).items():
            p[name][q] = score
        print('completed fold', f, flush=True)
    return p


def run():
    raw, y, groups, fold = data()
    x = compact_features(raw)
    cue = ['CryoSleep', *BILLS]
    context = x.drop(columns=cue)
    columns = [*FAMILIES, 'mitra96']
    scores = np.column_stack([load_scores(raw,y,fold,OUT/f'{k}.npz',['p'])[:,0] for k in columns])
    base = scores[:,-1] >= .5
    near = load_scores(raw,y,fold,DEST/'full.npz',NAMES)
    seven = ((scores >= .5) == base[:,None]).all(1) & ((near[:,[0,2]] >= .5) == base[:,None]).all(1)
    views = load_scores(raw,y,fold,ROOT/'artifacts/nov23_sleep_bill_factorial/oof.npz',['full'])
    # load_scores returns a matrix here since the single bank key holds three columns.
    assert views.shape == (len(y),3)
    missed_all = seven & (base != y) & ((views >= .5) != y[:,None]).all(1)
    assert (seven & (base != y)).sum() == 819 and missed_all.sum() == 532
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    hard = np.zeros(len(y), bool)
    hard[dev] = hard_cohorts(dev, y[dev])[2]
    p = fit_priors(x, cue, context, y, groups, fold)
    save_scores(raw, y, fold, p, 'full')
    def compare(prob, ref):
        pred = prob >= .5
        fixed, harmed = (pred == y) & (ref != y), (pred != y) & (ref == y)
        delta = fixed.astype(int)-harmed
        return dict(accuracy=float((pred == y).mean()), fixed=int(fixed.sum()), harmed=int(harmed.sum()),
            of_819=int((fixed & seven & (base != y)).sum()), seven_correct_harmed=int((harmed & seven & (base == y)).sum()),
            of_532=int((fixed & missed_all).sum()), historical_fixed=int((fixed & hard).sum()),
            additional_net=int(delta.sum()-delta[dev].sum()), fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)],
            ci_pp=interval(delta,groups))
    control = (scores[:,columns.index('cat')] + scores[:,-1]) >= 1
    result = {}
    for k, prob in p.items():
        assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
        blend = (prob+scores[:,-1])/2
        result[k] = dict(standalone=compare(prob,base), blend_vs_mitra=compare(blend,base),
                         blend_vs_unweighted_blend=compare(blend,control))
    report = ROOT/'reports/nov23_matched_learning_curve/bill_prior_balance.json'
    report.write_text(json.dumps(result, indent=2)+'\n')
    print([(k, v['standalone']['accuracy'], v['blend_vs_mitra']['accuracy']) for k,v in result.items()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--half', action='store_true')
    if parser.parse_args().half:
        raw, y, groups, fold = data()
        frame = compact_features(raw)
        cue = ['CryoSleep', *BILLS]
        split = fold % 2
        probabilities = fit_priors(frame, cue, frame.drop(columns=cue), y, groups, split)
        save_scores(raw, y, split, probabilities, 'half')
    else:
        run()
