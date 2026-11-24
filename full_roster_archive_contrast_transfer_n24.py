"""Transfer historical confidence contrasts across related scorers to unfamiliar orders."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats


def logits(p):
    return logit(np.clip(p, .005, .995))


def design(core, predictions, sources):
    base = logits(core)
    scores = logits(predictions)
    family_evidence = []
    contrasts = []
    for family in np.unique(sources):
        members = scores[:, sources == family]
        mean = members.mean(axis=1, keepdims=True)
        family_evidence.append(mean - base[:,[0]])
        if members.shape[1] > 1:
            # The last centered member is determined by all the other members.
            contrasts.append((members-mean)[:,:-1])
    anchor = np.column_stack([base[:,0], base[:,1:] - base[:,[0]]])
    means = np.column_stack([anchor,*family_evidence])
    return [anchor,means,np.column_stack([means,*contrasts])]


def adjust(source, query, baseline, query_baseline, y):
    center = source.mean(axis=0)
    scale = np.maximum(source.std(axis=0), .15)
    x = np.column_stack([np.ones(len(source)),(source-center)/scale])
    z = np.column_stack([np.ones(len(query)),(query-center)/scale])
    offset = logits(baseline)
    # A common penalty for each standardized predictor measured in log-odds.
    def objective(w):
        score = offset + x @ w
        loss = (np.logaddexp(0,score)-y*score).sum() + 50*(w@w)
        return loss, x.T @ (expit(score)-y) + 100*w
    fit = minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':200})
    assert fit.success, fit.message
    return expit(logits(query_baseline)+z @ fit.x)


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,oldfold,_,champ,_,_,_,_,old,_ = aligned()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    current = full[:,[names.index(k) for k in ('mitra96','v44_without_name','tabicl')]]
    folder = ROOT/'artifacts/nov24_remaining_cases'
    with np.load(folder/'archive_scores.npz') as archive:
        for k,val in [('row',rows),('old_fold',oldfold),('fold',fold),('passenger',raw.PassengerId),('y',y)]:
            np.testing.assert_array_equal(archive[k],val)
        sources = np.array([k.split('/')[0] for k in archive['names']])
        assert len(sources)==33 and len(np.unique(sources))==10
        source = design(old,archive['historical'],sources)
        dest = design(current,archive['full'],sources)
        model_hits = (archive['full'] >= .5) == y[:,None]
        oracle = model_hits.any(axis=1)
        disagree = (~model_hits).any(axis=1)
    with np.load(folder/'after_archive.npz') as archive:
        np.testing.assert_array_equal(archive['passenger'],raw.PassengerId)
        shared = archive['missed']
        assert shared.sum()==199 and not oracle[shared].any()
    unseen = ~np.isin(g,og)
    assert unseen.sum()==1044
    report = dict(samples=len(y),unseen_orders_passengers=int(unseen.sum()),
        base_errors=int((ref!=y).sum()), base_errors_with_any_correct_archived_prediction=int((oracle & (ref!=y)).sum()),
        base_correct_with_any_wrong_archived_prediction=int((disagree & (ref==y)).sum()))
    for mode,train_x,test_x in zip(('anchor','family_means','within_family_contrasts'),source,dest):
        p = np.full(len(y),np.nan)
        for f in np.unique(fold):
            q = fold==f
            t = ~np.isin(og,g[q])
            p[q] = adjust(train_x[t],test_x[q],old[t,0],current[q,0],oy[t].astype(float))
        assert np.isfinite(p).all()
        pred = np.where((current[:,0]>=.5)==ref,p>=.5,ref)
        fixed = shared & (pred==y)
        report[mode] = dict(full=stats(pred,ref,y,g,fold),
            previously_unseen=stats(pred[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
            all_199_fixed=int(fixed.sum()), unseen_shared=int((shared & unseen).sum()),
            unseen_shared_fixed=int((fixed & unseen).sum()), previously_unseen_fixed_by_fold=[
                int((fixed & unseen & (fold==f)).sum()) for f in np.unique(fold)])
        print(mode,report[mode]['full']['net'],report[mode]['previously_unseen']['net'],fixed.sum(),flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/archive_contrast_transfer.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
