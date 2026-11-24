"""Estimate when independent nearby booking panels can overrule the incumbent."""
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, BILLS, banks, data, latest_reference


def evidence(xy, lane, group, labels, fold, panel):
    """Give each donor booking one vote independently in each of two disjoint panels."""
    features = np.zeros((len(group), 2, 5))
    for f in np.unique(fold):
        for side in np.unique(lane):
            query = np.flatnonzero((fold == f) & (lane == side) & np.isfinite(xy[:, 1]))
            if not len(query):
                continue
            for split in (0, 1):
                donors = np.flatnonzero((fold != f) & (panel == split) & (lane == side) & np.isfinite(xy[:, 1]))
                assert not np.isin(group[donors], group[query]).any()
                if len(donors) < 4:
                    continue
                # One coordinate and one mean outcome for each other order on this side.
                frame = pd.DataFrame(dict(order=group[donors], x=xy[donors, 0], z=xy[donors, 1],
                                          outcome=labels[donors])).groupby('order').mean()
                if len(frame) < 4:
                    continue
                distance, near = cKDTree(frame[['x','z']]).query(xy[query], k=min(len(frame), 32))
                outcomes = frame.outcome.to_numpy()[near]
                for j, k in enumerate((4, 12, 32)):
                    features[query, split, j] = outcomes[:, :k].mean(1) - .5
                features[query, split, 3] = np.log1p(distance[:, :12][:, -1])
                features[query, split, 4] = 1
    return features


def decide(p, incumbent, cues, context, y, g, fold):
    # Apply the same eligibility and reference model to all four comparisons.
    direct = cues[:, :, 1].mean(axis=1) >= 0
    both = (cues[:, :, 1] >= 0) == direct[:, None]
    eligible = (direct != incumbent) & (incumbent == (p >= .5)) & (cues[:, :, 4] == 1).all(1)
    sign = np.where(direct, 1, -1)
    means = cues[:, :, :3].mean(axis=1)*sign[:, None]
    gaps = np.abs(cues[:, 0, :3]-cues[:, 1, :3])
    base = np.column_stack([abs(p-.5), direct, context])
    detail = np.column_stack([base, means, gaps, cues[:, :, 3].mean(1), both.all(1),
                              means*(.5-abs(p-.5))[:, None]])
    votes = dict(direct=eligible, panel_agreement=eligible & both.all(1))
    for name, x in [('profile', base), ('with_neighbours', detail)]:
        accept = np.zeros(len(y), bool)
        for f in np.unique(fold):
            rows = np.flatnonzero((fold == f) & eligible)
            for a,b in GroupKFold(4, shuffle=True, random_state=502).split(rows, groups=g[rows]):
                fit, ask = rows[a], rows[b]
                assert not np.isin(g[fit], g[ask]).any()
                fit_y = (direct == y)[fit]
                model = make_pipeline(StandardScaler(), LogisticRegression(C=.12, max_iter=300))
                model.fit(x[fit], fit_y)
                accept[ask] = model.predict_proba(x[ask])[:,1] > .5
        votes[name] = accept
    return votes


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    incumbent = latest_reference(y, group, fold, names, half, full)
    _, _, old_groups, *_ = aligned()
    p = full[:, names.index('mitra96')]
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as cache:
        for k,v in [('y', y), ('passenger', raw.PassengerId), ('fold', fold)]:
            np.testing.assert_array_equal(cache[k],v)
        hard = cache['remaining']
    assert hard.sum() == 147 and (incumbent[hard] != y[hard]).all()
    cabin = raw.Cabin.str.extract(r'^([^/]+)/(\d+)/([PS])$')
    lane = (cabin[0].fillna('?')+cabin[2].fillna('?')).to_numpy(str)
    xy = np.column_stack([group.astype(float)/5, pd.to_numeric(cabin[1])])
    solo = raw.groupby(group).PassengerId.transform('size').to_numpy() == 1
    context = np.column_stack([solo, raw.CryoSleep.eq(True), raw.CryoSleep.isna(), raw.Age.fillna(30)/40,
                               raw[BILLS].gt(0).sum(axis=1), np.log1p(raw[BILLS].fillna(0)),
                               raw.HomePlanet.eq('Earth'), raw.HomePlanet.eq('Europa')])
    orders, indices = np.unique(group, return_inverse=True)
    panel = np.random.default_rng(520).integers(0, 2, len(orders))[indices]
    shuffled = xy.copy()
    rng = np.random.default_rng(810)
    for side in np.unique(lane):
        members = np.flatnonzero(lane == side)
        shuffled[members] = xy[rng.permutation(members)]
    sections = dict(all=np.ones(len(y), bool), solo=solo, grouped=~solo, new_orders=~np.isin(group,old_groups))
    report = {}
    for mode, coords in [('real', xy), ('permuted', shuffled)]:
        cues = evidence(coords, lane, group, y, fold, panel)
        report[mode] = {}
        for method, vote in decide(p, incumbent, cues, context, y, group, fold).items():
            result = {}
            for key, mask in sections.items():
                fix = vote & mask & (incumbent != y)
                harm = vote & mask & (incumbent == y)
                result[key] = dict(n=int(mask.sum()), fixed=int(fix.sum()), harmed=int(harm.sum()),
                                   hard_fixed=int((fix & hard).sum()), hard_total=int((mask & hard).sum()),
                                   fold_net=[int(fix[fold==f].sum()-harm[fold==f].sum()) for f in np.unique(fold)])
            report[mode][method] = result
            r = result['all']
            print(mode, method, r['fixed'], r['harmed'], r['hard_fixed'], flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/cabin_donor_reliability.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
