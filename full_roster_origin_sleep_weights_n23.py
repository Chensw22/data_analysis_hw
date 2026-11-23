"""Partially pool expert weights across all observed origin and sleep categories."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from broad_booking_policy_replay_n23 import interval
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    reference = latest_reference(y, group, fold, names, half, full)
    origin = raw.HomePlanet.fillna('unknown')
    sleep = raw.CryoSleep.astype('string').fillna('unknown')
    strata = (origin + '/' + sleep).to_numpy(str)
    code = OneHotEncoder(sparse_output=False).fit_transform(strata[:, None])
    profile = {}
    for k in np.unique(strata):
        profile[k] = {}
        for direction in (True, False):
            m = (strata == k) & (reference == direction)
            profile[k][str(direction)] = dict(wrong=int(np.sum(m & (y != reference))),
                                               right=int(np.sum(m & (y == reference))))
    assert sum(v['wrong'] for k in profile.values() for v in k.values()) == 1575
    policies = {k: reference.copy() for k in ['global','intercepts','shared_and_specific','local']}
    for f in np.unique(fold):
        t, q = (fold % 2 == f % 2) & (fold != f), fold == f
        assert not np.isin(group[t], group[q]).any()
        scaler = StandardScaler()
        a = scaler.fit_transform(logit(half[t].clip(.001, .999)))
        b = scaler.transform(logit(full[q].clip(.001, .999)))
        for key in policies:
            def matrix(x, one_hot):
                if key == 'global':
                    return x
                if key == 'intercepts':
                    return np.column_stack([x, one_hot])
                slopes = (x[:, :, None] * one_hot[:, None, :]).reshape(len(x), -1)
                if key == 'local':
                    return np.column_stack([one_hot, slopes])
                return np.column_stack([x, one_hot, slopes])
            model = LogisticRegression(C=.1, max_iter=500)
            model.fit(matrix(a, code[t]), y[t])
            policies[key][q] = model.predict(matrix(b, code[q]))
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    # The destination cohort is recorded for interpretation. Training covers every origin.
    earth_sleep = strata == 'Earth/True'
    assert earth_sleep.sum() == 1382
    result = {}
    for key, pred in policies.items():
        delta = (pred == y).astype(int) - (reference == y)
        result[key] = dict(accuracy=float(np.mean(pred == y)), fixed=int((delta == 1).sum()),
                           harmed=int((delta == -1).sum()), earth_sleep_fixed=int((delta[earth_sleep] == 1).sum()),
                           earth_sleep_harmed=int((delta[earth_sleep] == -1).sum()),
                           additional_net=int(delta.sum()-delta[dev].sum()),
                           fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)],
                           ci_pp=interval(delta, group))
    output = dict(profile=profile, experiments=result)
    (ROOT / 'reports/nov23_matched_learning_curve/origin_sleep_weights.json').write_text(
        json.dumps(output, indent=2) + '\n')
    print(result)


if __name__ == '__main__':
    run()
