"""Predict transport through outcome subclasses defined by independent model failures."""
import json

import numpy as np
from catboost import CatBoostClassifier

from broad_booking_policy_replay_n23 import interval
from broad_tree_field_dividends_n23 import hard_cohorts
from exaone_validation_data import CATEGORICAL, compact_features
from full_roster_booking_validation_n23 import ROOT, SEED, data
from full_roster_conditional_pool_n23 import banks, latest_reference


def subclass(scores, target):
    vote = scores >= .5
    correct = vote == target[:,None]
    core = correct[:,:5].sum(1)
    # The marginal event Transported has four possible routes in the training labels.
    kind = np.select([core >= 3, core > 0, correct.any(1)], [0,1,2], default=3)
    return target.astype(int)*4 + kind


def run():
    raw,y,g,fold = data()
    names,h,p = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,p)
    raw_fields = compact_features(raw)
    _, ids, counts = np.unique(g,return_inverse=True,return_counts=True)
    raw_fields['BookingSize'] = counts[ids]
    hframe = raw_fields.assign(**dict(zip(names,h.T,strict=True)))
    pframe = raw_fields.assign(**dict(zip(names,p.T,strict=True)))
    labels = subclass(h,y)
    allowed = np.column_stack([subclass(p,np.full(len(y),value)) for value in (False,True)])
    original = ((p[:,:5] >= .5) != y[:,None]).all(1)
    hard819 = original & ((p[:,[names.index('without_near'),names.index('only_near')]] >= .5) != y[:,None]).all(1)
    all_wrong = ((p >= .5) != y[:,None]).all(1)
    assert (hard819.sum(), all_wrong.sum()) == (819,257)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    hard = np.zeros(len(y),bool)
    hard[dev] = hard_cohorts(dev,y[dev])[2]
    outputs = {}
    for key, a, b in [('raw',raw_fields,raw_fields),('raw_and_scores',hframe,pframe)]:
        pred = {k:np.empty(len(y)) for k in ['binary','shuffled_marginal','marginal','compatible']}
        for f in np.unique(fold):
            t = np.flatnonzero((fold % 2 == f % 2) & (fold != f))
            q = np.flatnonzero(fold == f)
            assert not np.isin(g[q],g[t]).any()
            shuffled = labels[t].copy()
            rng = np.random.default_rng(SEED+int(f))
            for value in [False, True]:
                mask = y[t] == value
                shuffled[mask] = rng.permutation(shuffled[mask])
            for mode, target in [('binary',y[t].astype(int)),('shuffled',shuffled),('subclass',labels[t])]:
                model = CatBoostClassifier(iterations=280,depth=4,learning_rate=.055,random_seed=SEED,
                    verbose=False,thread_count=3,allow_writing_files=False,cat_features=CATEGORICAL)
                model.fit(a.iloc[t],target)
                z = model.predict_proba(b.iloc[q])
                if mode == 'binary':
                    pred['binary'][q] = z[:,1]
                    continue
                np.testing.assert_array_equal(model.classes_,np.arange(8))
                marginal = z[:,4:].sum(1)
                if mode == 'shuffled':
                    pred['shuffled_marginal'][q] = marginal
                else:
                    pred['marginal'][q] = marginal
                    values = np.take_along_axis(z,allowed[q],axis=1)
                    pred['compatible'][q] = values[:,1]/values.sum(1)
        for method, prob in pred.items():
            assert np.isfinite(prob).all() and ((0 <= prob) & (prob <= 1)).all()
            pick = prob >= .5
            delta = (pick == y).astype(int)-(ref == y)
            outputs[key+'_'+method] = dict(accuracy=float((pick == y).mean()),
                fixed_vs_124=int((delta == 1).sum()), harmed_vs_124=int((delta == -1).sum()),
                original_common_correct=int(((pick == y) & original).sum()),of_819=int(((pick == y) & hard819).sum()),
                of_257=int(((pick == y) & all_wrong).sum()), all_experts_right_harmed=int(((pick != y) & ((p >= .5) == y[:,None]).all(1)).sum()),
                historical_correct=int(((pick == y) & hard).sum()),additional_net=int(delta.sum()-delta[dev].sum()),
                by_fold=[int(delta[fold == f].sum()) for f in np.unique(fold)],ci_pp=interval(delta,g))
        print('completed',key,flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/failure_subclasses.json'
    path.write_text(json.dumps(outputs,indent=2)+'\n')
    print({k: (v['accuracy'],v['of_257']) for k,v in outputs.items()})


if __name__ == '__main__':
    run()
