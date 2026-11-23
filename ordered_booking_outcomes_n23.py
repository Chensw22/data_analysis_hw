"""Learn directed transitions between successive party members and marginalize unknown outcomes."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partial_booking_likelihood_n23 import contexts
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from exaone_validation_data import compact_features


def directions(ids, group, position, prior, sleep, age, cabin, shuffled):
    parties = pd.Series(ids).groupby(group[ids]).agg(list)
    width = max(map(len, parties))
    rng = np.random.default_rng(2882)
    rows = [[], []]
    for members in parties:
        ordered = np.asarray(members)[np.argsort(position[members])]
        if shuffled:
            ordered = rng.permutation(ordered)
        for traversal, seq in zip(rows, [ordered, ordered[::-1]]):
            traversal.append(np.pad(seq, (0, width-len(seq)), constant_values=-1))
    parts = []
    for traversal in rows:
        index = np.asarray(traversal)
        valid = index >= 0
        z = np.where(valid, prior[index], 0.).clip(-6, 6)
        s = expit(z)
        n = valid.sum(1)
        mean = (valid*s).sum(1)/n
        previous = np.roll(index, 1, axis=1)
        previous[:, 0] = index[:, 0]
        has_previous = valid & (np.arange(width)[None, :] > 0)
        prev_p = np.where(has_previous, expit(prior[previous]), .5)
        same_cabin = (cabin[index] != '') & (cabin[index] == cabin[previous])
        # A pair of unrecorded cryosleep statuses also has the same recorded category.
        same_sleep = sleep[index] == sleep[previous]
        close_age = np.abs(age[index]-age[previous]) < 10
        attrs = [np.ones_like(z), 2*prev_p-1, 2*s-1, same_cabin, same_sleep, close_age]
        edge = np.stack(attrs, axis=2)*has_previous[:, :, None]
        node = np.stack([np.ones_like(z), z, np.broadcast_to(mean[:, None]-.5, z.shape),
                         np.broadcast_to(np.log(n)[:, None], z.shape), *np.moveaxis(edge, 2, 0)], axis=2)
        parts.append((index, z, prev_p, node, edge))
    return parts


def forward(part, w, labels=None, gradient=False):
    index, offset, prev_p, node, edge = part
    n_node = node.shape[-1]
    base = node @ w[:n_node] + offset
    linked = len(w) > n_node
    delta = edge @ w[n_node:] if linked else np.zeros_like(base)
    p0, p1 = expit(base - prev_p*delta), expit(base + (1-prev_p)*delta)
    prev = np.full(len(index), .5)
    derivative = np.zeros((len(index), len(w)))
    grad = np.zeros(len(w))
    loglik = np.zeros(len(index))
    pred = np.zeros_like(base)
    for j in range(index.shape[1]):
        low, high = p0[:, j], p1[:, j]
        pred[:, j] = prob = (1-prev)*low + prev*high
        if gradient:
            a, b = (1-prev)*low*(1-low), prev*high*(1-high)
            derivative *= (high-low)[:, None]
            derivative[:, :n_node] += node[:, j]*(a+b)[:, None]
            if linked:
                derivative[:, n_node:] += edge[:, j]*(b - prev_p[:, j]*(a+b))[:, None]
        prev = prob.copy()
        if labels is not None:
            outcome = labels[index[:, j]]
            known = (index[:, j] >= 0) & (outcome >= 0)
            p = prob[known].clip(1e-12, 1-1e-12)
            y = outcome[known]
            loglik[known] += np.where(y > 0, np.log(p), np.log1p(-p))
            prev[known] = y
            if gradient:
                grad += ((p-y)/(p*(1-p))) @ derivative[known]
                derivative[known] = 0
    return pred, loglik, grad


def fit(parts, labels, linked):
    node, edge = parts[0][-2:]
    width = node.shape[-1] + (edge.shape[-1] if linked else 0)
    def objective(w):
        loss, derivative = 6*w@w, 12*w
        for part in parts:
            _, ll, grad = forward(part, w, labels, gradient=True)
            loss -= .5*ll.sum()
            derivative += .5*grad
        return loss, derivative
    model = minimize(objective, np.zeros(width), jac=True, method='L-BFGS-B',
                     options={'ftol': 1e-11, 'maxiter': 150})
    if not model.success:
        raise RuntimeError(model.message)
    return model.x


def infer(parts, w, labels):
    probabilities, lls = [], []
    for part in parts:
        p = np.full(len(labels), np.nan)
        index = part[0]
        mask = index >= 0
        # All marginals are computed before revealing any outcome.
        p[index[mask]] = forward(part, w)[0][mask]
        probabilities.append(p)
        lls.append(forward(part, w, labels)[1])
    return .5*(probabilities[0]+probabilities[1]), np.logaddexp(*lls)-np.log(2.)


def run():
    row, y, _, group, _ = load()
    raw = pd.read_csv(ROOT/'data/raw/train.csv', usecols=lambda c: c not in ['Name', 'Transported'])
    gid, pos = raw.PassengerId.str.split('_', expand=True).astype(int).to_numpy().T
    np.testing.assert_array_equal(group, gid[row])
    frame = compact_features(raw)
    frame['BookingSize'] = pd.Series(gid).map(pd.Series(gid).value_counts()).to_numpy()
    attrs = (frame.CryoSleep.to_numpy(), frame.Age.to_numpy(), raw.Cabin.fillna('').to_numpy())
    records, histories, anchors, allwrong = {}, {}, [], []
    out = ART/'nov23_ordered_booking_outcomes'
    out.mkdir(exist_ok=True)
    for seed in map(str, SEEDS):
        fold, ref, scores, _ = read_opinions(seed, row, y)
        anchors.append(ref == y)
        allwrong.append(((scores >= .5) != y[:, None]).all(1))
        bank, losses = {}, {}
        for f in np.unique(fold):
            t, q = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            assert not np.intersect1d(group[t], group[q]).size
            obs, it, zt, iq, zq = contexts(frame, gid, row, y, t, q)
            assert np.all(obs[iq] == -1)
            hold = np.full(len(raw), -1, int)
            hold[row[q]] = y[q]
            # The three teachers were each trained on the same fraction of the
            # outer training groups as the teacher behind an OOF training score.
            for shuffled, prefix in [(False, 'ticket'), (True, 'random')]:
                train = directions(it, gid, pos, zt, *attrs, shuffled)
                query = [directions(iq, gid, pos, score, *attrs, shuffled) for score in zq]
                for linked in [False, True]:
                    key = prefix + ('_chain' if linked else '_independent')
                    w = fit(train, obs, linked)
                    predictions, likelihoods = zip(*(infer(parts, w, hold) for parts in query))
                    bank.setdefault(key, np.full(len(y), np.nan))[q] = np.mean([p[row[q]] for p in predictions], 0)
                    # Mixture over three independently fitted teacher models and both directions.
                    values = np.logaddexp.reduce(np.stack(likelihoods), axis=0)-np.log(len(query))
                    losses.setdefault(key, []).append(float(-values.sum()))
        for k in list(bank):
            assert np.isfinite(bank[k]).all()
            bank['blend_'+k] = expit((logit(scores[:, 0].clip(.002,.998)) + logit(bank[k].clip(.002,.998)))/2)
        np.savez_compressed(out/f'{seed}.npz', row=row, y=y, fold=fold, ref=ref, **bank)
        records[seed] = {}
        for k, p in bank.items():
            hit = (p >= .5) == y
            compare = (bank[k.replace('_chain', '_independent')] >= .5) == y
            histories.setdefault(k, []).append(hit)
            records[seed][k] = dict(vs_ref=paired(hit, ref == y), vs_independent=paired(hit, compare))
            if k in losses:
                records[seed][k]['joint_nll_per_label'] = sum(losses[k])/len(y)
        print(seed, records[seed]['ticket_chain'], flush=True)
    ref = np.stack(anchors)
    shared = np.stack(allwrong).all(0)
    assert int(shared.sum()) == 332
    scopes = {'all': np.ones(len(y), bool), 'stable_errors': ~ref.any(0), 'unanimous_wrong': shared,
              'unstable': ref.any(0) & ~ref.all(0), 'always_correct': ref.all(0),
              'solo': frame.BookingSize.to_numpy()[row] == 1, 'multi': frame.BookingSize.to_numpy()[row] > 1}
    result = {}
    for k, values in histories.items():
        h = np.stack(values)
        compare = np.stack(histories[k.replace('_chain','_independent')])
        result[k] = dict(vs_independent_ci_pp=interval((h.astype(float)-compare).mean(0), group),
                         scopes={s: dict(n=int(m.sum()), correct_all=int(h[:,m].all(0).sum()),
                                   fixed_vs_independent=(h[:,m] & ~compare[:,m]).sum(1).tolist(),
                                   harmed_vs_independent=(~h[:,m] & compare[:,m]).sum(1).tolist()) for s,m in scopes.items()})
    write(ROOT/'reports/nov23_ordered_booking_outcomes/result.json', dict(plans=records, grouped=result))


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
