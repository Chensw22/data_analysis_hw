"""Learn smooth probability corrections on independent bookings using landmark kernels."""
import json

import numpy as np
from scipy.spatial.distance import pdist
from scipy.special import expit, logit
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from deployment_booking_mix_n23 import passenger_vectors
from full_roster_adaptive_bill_prior_n23 import EVIDENCE, proposal_design
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_same_model_calibration_n23 import learn


def half_anchor(half, y, fold, donors, conflict, design):
    result = half[:, 4].copy()
    for h in np.unique(fold[donors]):
        t = donors & (fold != h) & conflict
        q = donors & (fold == h) & conflict
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500))
        direction = half[:, 4] >= .5
        model.fit(design[t], (y[t] != direction[t]))
        flip = model.predict_proba(design[q])[:, 1]
        result[q] = np.where(direction[q], 1-flip, flip)
    return result[donors]


def kernel(train, query):
    sample = np.random.default_rng(714).choice(len(train), min(len(train), 512), replace=False)
    distance = np.median(pdist(train[sample], metric='sqeuclidean'))
    assert distance > 0
    model = Nystroem(gamma=1/distance, n_components=144, random_state=714)
    return model.fit_transform(train), model.transform(query)


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    base = latest_reference(y, group, fold, names, half, full)
    gate, _ = gate_probabilities(names, half, full, y, group, fold)
    np.testing.assert_array_equal(gate >= .5, base)
    cols = [names.index(k) for k in EVIDENCE]
    conflict, design = proposal_design(half[:, :5], half[:, cols])
    features = passenger_vectors(raw)
    shared = ((full >= .5) != y[:, None]).all(1)
    unanimous_correct = ((full >= .5) == y[:, None]).all(1)
    _, ids, sizes = np.unique(group, return_inverse=True, return_counts=True)
    solo = sizes[ids] == 1
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as archive:
        additional = np.ones(len(y), bool)
        additional[archive['development']] = False
    fit_keys = ('transfer', 'same_model', 'transfer_then_same')
    outputs = {(view, key): np.full(len(y), np.nan)
               for view in ('calibration', 'attributes', 'attributes_experts') for key in fit_keys}
    for f in np.unique(fold):
        donors = (fold % 2 == f % 2) & (fold != f)
        q = np.flatnonzero(fold == f)
        assert not np.isin(group[donors], group[q]).any()
        ht = logit(half_anchor(half,y,fold,donors,conflict,design).clip(.001,.999))
        ft = logit(gate[q].clip(.001,.999))
        control = np.column_stack([np.ones(len(ht)), ht])
        test_control = np.column_stack([np.ones(len(ft)), ft])
        for view in ('calibration', 'attributes', 'attributes_experts'):
            x, v = control, test_control
            if view != 'calibration':
                a, b = features[donors], features[q]
                if view == 'attributes_experts':
                    a = np.column_stack([a, logit(half[donors, :5].clip(.02,.98))/2])
                    b = np.column_stack([b, logit(full[q, :5].clip(.02,.98))/2])
                ka, kb = kernel(a,b)
                x, v = np.column_stack([x, ka]), np.column_stack([v, kb])
            zero = np.zeros(x.shape[1])
            transfer = learn(x, y[donors], ht, zero)
            outputs[view, 'transfer'][q] = expit(ft + v @ transfer)
            for a,b in GroupKFold(4,shuffle=True,random_state=462).split(q, groups=group[q]):
                assert not np.isin(group[q[a]], group[q[b]]).any()
                for key, center in [('same_model',zero), ('transfer_then_same',transfer)]:
                    w = learn(v[a], y[q[a]], ft[a], center)
                    outputs[view,key][q[b]] = expit(ft[b] + v[b] @ w)
    report = {}
    for (view, key), p in outputs.items():
        assert np.isfinite(p).all()
        correct = (p >= .5) == y
        d = correct.astype(int) - (base == y)
        peer = ((outputs['calibration',key] >= .5) == y)
        report[view+'_'+key] = dict(accuracy=float(correct.mean()),fixed=int((d==1).sum()),harmed=int((d==-1).sum()),
                common_fixed=int((d[shared]==1).sum()),unanimous_harmed=int((d[unanimous_correct]==-1).sum()),
                solo_net=int(d[solo].sum()),multi_net=int(d[~solo].sum()),additional_net=int(d[additional].sum()),
                vs_calibration=int(correct.sum()-peer.sum()),fold_net=[int(d[fold==f].sum()) for f in np.unique(fold)],
                ci_pp=interval(d,group))
        print(view,key,report[view+'_'+key],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/kernel_bias.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
