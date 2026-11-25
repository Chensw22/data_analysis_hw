"""Orient local contrasts with one-to-one pairs of similar manifests from different bookings."""
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from frozen_queue_shared_errors_n26 import paired
from manifest_sleep_contrast_n26 import design
from price_implied_deck_n26 import FEES
from target_mix_expert_routing_n26 import ROOT, ticker


def coordinates(x, train):
    med = np.nanmedian(x[train], axis=0)
    med = np.nan_to_num(med)
    scale = np.maximum(np.nan_to_num(np.nanstd(x[train], axis=0)), .5)
    values = np.nan_to_num((x - med) / scale, nan=0).clip(-5, 5)
    return np.column_stack([values, np.isnan(x)])


def comparisons(z, strata, group, y, source, query, seed):
    rng = np.random.default_rng(seed)
    assert not np.isin(group[source], group[query]).any()
    cases, controls = [], []
    for key in np.unique(strata[source]):
        c = source[(strata[source] == key) & ~y[source]]
        h = source[(strata[source] == key) & y[source]]
        if not min(len(c),len(h)):
            continue
        d = cdist(z[c], z[h], 'sqeuclidean')
        blocked = group[c,None] == group[h]
        d[blocked] = d.max() * (min(d.shape)+1) + 1
        i, j = linear_sum_assignment(d)
        ok = ~blocked[i,j]
        cases.extend(c[i[ok]])
        controls.extend(h[j[ok]])
    cases, controls = np.array(cases), np.array(controls)
    assert len(np.unique(np.r_[cases, controls])) == 2 * len(cases)
    assert np.all(group[cases] != group[controls])
    c, h = z[cases], z[controls]
    orient = rng.choice([-1,1], size=len(cases))
    # For each query, look up nearby pair midpoints without using query labels.
    mid = (c + h)/2
    dist = cdist(z[query], mid, 'sqeuclidean')
    dist += 12 * (strata[query,None] != strata[cases])
    order = np.argsort(dist,axis=1,kind='stable')[:,:32]
    delta = c - h
    length = np.sqrt(np.square(delta).sum(axis=1)).clip(.25)
    normal = delta / length[:,None]
    signed = np.einsum('qkd,qkd->qk', z[query,None,:] - mid[order], normal[order])
    weights = 1/(1+np.take_along_axis(dist,order,axis=1))
    def summarize(v):
        return np.column_stack([np.average(v[:,:k], axis=1, weights=weights[:,:k]) for k in (4,12,32)])
    return summarize(signed), summarize(signed * orient[order])


def predict(x, strata, group, y, t, q, seed):
    scores = np.empty((3,len(q),3))
    for run, (a,b) in enumerate(GroupKFold(3,shuffle=True,random_state=seed).split(t,groups=group[t])):
        bank, calibrate = t[a], t[b]
        z = coordinates(x, bank)
        real, placebo = comparisons(z,strata,group,y,bank,np.r_[calibrate,q],seed+run)
        assert not np.isin(group[calibrate],group[q]).any()
        n = len(calibrate)
        # Fit class prevalence on the naturally sampled calibration bookings.
        for j, contrast in enumerate((None, real, placebo)):
            zz = z[np.r_[calibrate,q]]
            if contrast is not None:
                zz = np.column_stack([zz,contrast])
            scale = StandardScaler().fit(zz[:n])
            xx = scale.transform(zz).clip(-6,6)
            m = LogisticRegression(C=.1, max_iter=800).fit(xx[:n],~y[calibrate])
            scores[run,:,j] = m.predict_proba(xx[n:])[:,1]
    return scores.mean(axis=0)


def run():
    raw, y, group, fold, known, versions = paired()
    eligible = raw.CryoSleep.eq(True).to_numpy() & ~raw[FEES].gt(0).any(axis=1).to_numpy()
    x = design(raw,group)
    strata = (raw.HomePlanet.fillna('?') + '/' + raw.Cabin.str.split('/').str[0].fillna('?')).to_numpy()
    scores = np.full((len(y),3),np.nan)
    for k in np.unique(fold):
        t,q = [np.flatnonzero(v & eligible) for v in (fold != k,fold == k)]
        scores[q] = predict(x,strata,group,y,t,q,2061+k)
    assert np.isfinite(scores[eligible]).all()
    core = eligible & known & versions.all(axis=1)
    assert (core & ~y).sum()==388
    direct, real, shuffled = (scores>.5).T
    report = {}
    for name, action in dict(direct=direct, paired=real, randomly_oriented=shuffled,
                            new_vs_direct=real & ~direct, new_vs_random=real & ~shuffled).items():
        report[name] = ticker(core,~y,action)
        report[name]['fold_net'] = [int(np.sum(np.where(y,-1,1)*core*action*(fold==k))) for k in np.unique(fold)]
    print(report,flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/localized_opposite_order_pairs.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
