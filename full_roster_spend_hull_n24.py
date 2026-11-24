"""Compare paid passengers to mixtures of transport-specific spending profiles."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from full_roster_conditional_pool_n23 import BILLS, ROOT, banks, data, latest_reference
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_same_model_calibration_n23 import learn


def distances(x, vertices):
    """Exact squared distance to the nearest vertex or any point of their triangle."""
    a, b, c = vertices
    near = np.min(((x[:, None, :] - vertices[None, :, :])**2).sum(2), axis=1)
    best = near.copy()
    for start, end in [(a, b), (b, c), (c, a)]:
        line = end - start
        coeff = np.clip(((x-start) @ line) / max(line @ line, 1e-12), 0, 1)
        best = np.minimum(best, ((x-start-coeff[:, None]*line)**2).sum(1))
    basis = np.stack([b-a,c-a])
    xy = (x-a) @ np.linalg.pinv(basis)
    inside = (xy >= 0).all(1) & (xy.sum(1) <= 1)
    best[inside] = np.minimum(best[inside], ((x[inside]-a-xy[inside]@basis)**2).sum(1))
    assert np.all(best <= near + 1e-12)
    return near, best


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    ref = latest_reference(y, group, fold, names, half, full)
    gate, _ = gate_probabilities(names, half, full, y, group, fold)
    np.testing.assert_array_equal(gate >= .5, ref)
    bills = raw[BILLS].fillna(0).to_numpy()
    total = bills.sum(1)
    count = (bills>0).sum(1)
    # Missing amounts are unknown, so the experiment applies to complete bills only.
    eligible = (total>0) & raw[BILLS].notna().all(axis=1).to_numpy()
    x = np.sqrt(bills/np.maximum(total[:,None],1))
    shared = ((full >= .5) != y[:,None]).all(1)
    common_right = ((full >= .5) == y[:,None]).all(1)
    cohort = (raw.HomePlanet.eq('Earth') & raw.CryoSleep.eq(False)).to_numpy() & (count>=2) & (count<=4)
    controls = pd.DataFrame(dict(planet=raw.HomePlanet.fillna('missing'),
                                 sleep=raw.CryoSleep.fillna('missing'), count=count))
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as archived:
        additional = np.ones(len(y), bool)
        additional[archived['development']] = False
    report = dict(paid_complete=int(eligible.sum()), shared_eligible=int((eligible&shared).sum()),
                  cohort=dict(all=int(cohort.sum()), eligible=int((cohort&eligible).sum()),
                              common_wrong=int((cohort&shared).sum()),
                              common_transported=int((cohort&shared&y).sum()),
                              eligible_common_transported=int((cohort&shared&eligible&y).sum())),
                  methods={})
    for label in ['real','stratified_shuffle']:
        delta = {k:np.zeros(len(y)) for k in ['vertex','hull']}
        for f in np.unique(fold):
            t = np.flatnonzero((fold!=f)&eligible)
            q = np.flatnonzero((fold==f)&eligible)
            assert not np.isin(group[t],group[q]).any()
            target = y[t].copy()
            if label=='stratified_shuffle':
                rng = np.random.default_rng(714+int(f))
                for ids in controls.iloc[t].groupby(list(controls)).indices.values():
                    target[ids] = rng.permutation(target[ids])
            per_class = []
            for outcome in [False,True]:
                centers = KMeans(n_clusters=3,n_init=10,random_state=596).fit(x[t[target==outcome]]).cluster_centers_
                per_class.append(distances(x[q],centers))
            for i,k in enumerate(delta):
                delta[k][q] = per_class[0][i]-per_class[1][i]
        for geometry, gap in delta.items():
            # Positive gap favors transported. Calibrate one slope on independent bookings.
            # Direct nearest-class classification is evaluated without seeing any query label.
            direct = ref.copy()
            direct[eligible] = gap[eligible]>0
            p = gate.copy()
            for f in np.unique(fold):
                outer = np.flatnonzero((fold==f)&eligible)
                for a,b in GroupKFold(4,shuffle=True,random_state=462).split(outer,groups=group[outer]):
                    t,q = outer[a],outer[b]
                    scale = max(np.std(gap[t]),1e-8)
                    w = learn(gap[t,None]/scale, y[t], logit(gate[t].clip(.001,.999)), np.zeros(1))
                    p[q] = expit(logit(gate[q].clip(.001,.999))+gap[q]*w[0]/scale)
            for policy,pick in [('direct',direct),('combined',p>=.5)]:
                change = (pick==y).astype(int)-(ref==y)
                report['methods']['_'.join([label,geometry,policy])] = dict(
                    accuracy=float((pick==y).mean()),fixed=int((change==1).sum()),harmed=int((change==-1).sum()),
                    common_fixed=int((change[shared]==1).sum()),unanimous_harmed=int((change[common_right]==-1).sum()),
                    cohort_fixed=int((change[cohort]==1).sum()),cohort_harmed=int((change[cohort]==-1).sum()),
                    cohort_common_fixed=int((change[cohort&shared]==1).sum()),additional_net=int(change[additional].sum()),
                    fold_net=[int(change[fold==f].sum()) for f in np.unique(fold)],ci_pp=interval(change,group))
    path = ROOT/'reports/nov23_matched_learning_curve/spend_hull.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
