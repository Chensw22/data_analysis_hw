"""Fit additive passenger profiles and select interactions between locally linear basis functions."""
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, CATEGORICAL, load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write


def basis(frame, train):
    """Partition each observed numeric range with overlapping triangular functions."""
    columns, names, parents = [], [], []
    for field in frame.columns:
        v = frame[field]
        block = {}
        if field in CATEGORICAL:
            for key in sorted(v.iloc[train].dropna().unique()):
                block[str(key)] = v.eq(key).to_numpy()
        else:
            x = v.to_numpy(dtype=float, copy=True)
            observed = np.isfinite(x)
            if field in BILLS:
                block['zero'] = x == 0
                observed &= x > 0
                x = np.log1p(x)
            block['missing'] = ~np.isfinite(x)
            known = x[train][observed[train]]
            if len(known):
                knots = np.unique(np.quantile(known, [0, .2, .5, .8, 1]))
                for j in range(len(knots)):
                    values = np.zeros(len(x))
                    values[observed] = np.interp(x[observed], knots, np.eye(len(knots))[j])
                    block[f'k{j}'] = values
        for key, values in block.items():
            if np.std(values[train]) > .01:
                columns.append(values)
                names.append(f'{field}={key}')
                parents.append(field)
    return np.asfortranarray(np.column_stack(columns)), names, parents


def fit(design, train, query, y):
    x = np.ascontiguousarray(design[train])
    def loss(coef):
        margin = coef[0] + x @ coef[1:]
        grad = expit(margin) - y[train]
        return (np.logaddexp(0., margin).sum() - margin @ y[train] + .9 * (coef[1:] @ coef[1:]),
                np.r_[grad.sum(), x.T @ grad + 1.8 * coef[1:]])
    res = minimize(loss, np.zeros(x.shape[1]+1), jac=True, method='L-BFGS-B',
                   options={'maxiter': 700, 'ftol': 1e-10})
    if not res.success:
        raise ValueError(res.message)
    return expit(res.x[0] + design[query] @ res.x[1:]), expit(res.x[0] + x @ res.x[1:])


def select_products(anchors, terms, groups, train, y, prob, modes, keep):
    """Screen sparse interactions by a score statistic using complementary training bookings."""
    halves = np.random.default_rng(147).permutation(np.unique(groups[train]))
    first = np.isin(groups[train], halves[::2])
    scores = {name: [] for name in modes}
    for start in range(0, len(terms), 128):
        batch = terms[start:start+128]
        x = np.prod([anchors[train[:, None], batch[:, j]] for j in range(batch.shape[1])], axis=0)
        a, b = [((y[train][mask] - prob[mask]) @ x[mask]) /
                np.sqrt(2. + (prob[mask]*(1-prob[mask])) @ (x[mask]**2)) for mask in (first, ~first)]
        for mode in modes:
            evidence = abs(a+b) if mode == 'pooled' else np.where(a*b>0, np.minimum(abs(a),abs(b)), 0)
            scores[mode].extend(evidence.tolist())
    return {key: terms[np.argsort(-np.asarray(values), kind='stable')[:keep]] for key,values in scores.items()}


def extend(anchors, terms):
    return np.prod([anchors[:, terms[:, j]] for j in range(terms.shape[1])], axis=0)


def run_fold(frame, groups, y, train, query):
    h, names, parents = basis(frame, train)
    additive, fitted = fit(h, train, query, y)
    pred = {'additive': additive}
    pairs = np.asarray([(i,j) for i,j in combinations(range(len(parents)),2) if parents[i] != parents[j]])
    choices = select_products(h, pairs, groups, train, y, fitted, ['pooled','replicated'], 48)
    examples = {}
    for mode, chosen in choices.items():
        design = np.column_stack([h, extend(h, chosen)])
        doubles, p = fit(design, train, query, y)
        pred[f'{mode}_pair'] = doubles
        triples = sorted({tuple(sorted((a,b,c))) for a,b in chosen for c in range(len(parents))
                          if parents[c] not in (parents[a], parents[b])})
        top = select_products(h, np.asarray(triples), groups, train, y, p, [mode], 48)[mode]
        pred[f'{mode}_triple'] = fit(np.column_stack([design, extend(h, top)]), train, query, y)[0]
        examples[mode] = [[names[index] for index in term] for term in chosen[:6]]
    return pred, examples


def main():
    row,y,frame,g,_ = load()
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        shared_wrong = (z['votes'] != y[None,:,None]).all((0,2))
        shared_right = (z['votes'] == y[None,:,None]).all((0,2))
    out = ART/'nov23_adaptive_manifest_splines'
    out.mkdir(exist_ok=True)
    plans, trials = {}, {}
    for seed in [*map(str,SEEDS), 'v131']:
        fold, ref, op, _ = read_opinions(seed,row,y)
        locked = ref != (op[:,0] >= .5)
        pred, examples = {}, []
        for f in np.unique(fold):
            q = np.flatnonzero(fold==f)
            t = np.flatnonzero((fold!=f) & ~np.isin(g, g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            scores, ex = run_fold(frame,g,y,t,q)
            examples.append(ex)
            for mode, values in scores.items():
                pred.setdefault(mode,np.full(len(y),np.nan))[q] = values
        hits = {'reference':ref==y}
        for mode, values in pred.items():
            assert np.isfinite(values).all()
            hits[mode] = (values>=.5)==y
            blend = .75*op[:,0] + .25*values
            hits['quarter_'+mode] = np.where(locked,ref,blend>=.5)==y
        plans[seed] = {'models': {k:paired(v,ref==y) for k,v in hits.items()}, 'first_pairs':examples}
        if seed != 'v131':
            for k,v in hits.items():
                trials.setdefault(k,[]).append(v)
        np.savez_compressed(out/f'{seed}.npz', row=row,y=y,fold=fold,ref=ref,base=op[:,0],**pred)
        print(seed,{k:d['fixed']-d['harmed'] for k,d in plans[seed]['models'].items()},flush=True)
    ref = np.stack(trials['reference'])
    summary = {}
    for k, runs in trials.items():
        a = np.stack(runs)
        summary[k] = dict(shared_wrong_stable=int((a.all(0)&shared_wrong).sum()),
                          shared_right_missed_stable=int((~a.any(0)&shared_right).sum()),
                          mean_gain95_pp=interval((a.astype(float)-ref).mean(0),g))
    write(ROOT/'reports/nov23_adaptive_manifest_splines/result.json',dict(plans=plans,stable=summary))


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
