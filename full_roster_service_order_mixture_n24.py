"""Classify by integrating conditional service probabilities over all 120 observation orders."""
import json

import numpy as np
import torch
from scipy.special import expit, logit
from torch import nn
from torch.nn import functional as F

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_zero_truncated_bills_n24 import inputs

SERVICES = 5
CLASSES = 12  # missing, zero, ten positive-amount bands
BITS = (1 << np.arange(SERVICES))
MASKS = (np.arange(1 << SERVICES)[:,None] & BITS) != 0


def amount_codes(bills, t):
    codes = np.zeros_like(bills, dtype=int)
    for service, v in enumerate(bills.T):
        values = v[t]
        cuts = np.unique(np.quantile(values[values > 0], np.arange(1,10)/10))
        codes[:,service] = np.where(np.isnan(v), 0, np.where(v == 0, 1, 2+np.searchsorted(cuts, v)))
    assert codes.min() >= 0 and codes.max() < CLASSES
    return codes


class ServiceDecoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(width, 40), nn.SiLU())
        self.prior = nn.Linear(40, 1)
        self.encoder = nn.Embedding(SERVICES*(CLASSES+1), 6)
        self.predictor = nn.Sequential(nn.Linear(40 + 6*SERVICES + 1, 80), nn.SiLU(),
                                       nn.Linear(80, SERVICES*CLASSES))
        self.register_buffer('offset', torch.arange(SERVICES)*(CLASSES+1))

    def decode(self, context, code, mask, label):
        visible = torch.where(mask, code+1, 0)
        embedding = self.encoder(visible+self.offset).flatten(1)
        return self.predictor(torch.cat([context, embedding, label[:,None]], 1)).reshape(-1,SERVICES,CLASSES)


def integrate(log_prob):
    # log_prob[subset, passenger, next service]; each permutation has mass 1/120.
    total = np.zeros((1 << SERVICES, log_prob.shape[1]))
    for state in range(1, len(total)):
        active = np.flatnonzero(MASKS[state])
        paths = [total[state ^ BITS[j]] + log_prob[state ^ BITS[j], :, j] for j in active]
        total[state] = np.logaddexp.reduce(paths, axis=0)-np.log(len(active))
    return total[-1]


def evaluate(model, x, codes, q, independent):
    with torch.inference_mode():
        context = model.context(x[q])
        prior = model.prior(context).flatten().numpy()
        bill = codes[q]
        likelihoods = []
        for side in (0, 1):
            observed = []
            states = MASKS[:1] if independent else MASKS[:-1]
            label = torch.full((len(q),), float(side))
            for visible in states:
                logits = model.decode(context, bill, torch.from_numpy(visible), label)
                logs = torch.log_softmax(logits, dim=-1).gather(2, bill.unsqueeze(-1)).squeeze(-1)
                observed.append(logs.numpy().astype(float))
            product = observed[0].sum(1)
            joint = product if independent else integrate(np.stack(observed))
            likelihoods.append([product, joint])
        log_likelihood = np.array(likelihoods).transpose(2, 1, 0)
        posterior = expit(prior[:,None] + log_likelihood[:,:,1]-log_likelihood[:,:,0])
        return posterior, log_likelihood


def fit(context, bills, y, t, q):
    x = torch.tensor(((context-context[t].mean(0))/context[t].std(0).clip(.25)).clip(-10,10), dtype=torch.float32)
    codes = torch.tensor(amount_codes(bills, t))
    target = torch.tensor(y, dtype=torch.float32)
    score, likelihood = [], []
    for independent in (True, False):
        torch.manual_seed(731)
        rng = np.random.default_rng(731)
        model = ServiceDecoder(x.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.02)
        for epoch in range(45):
            for ix in np.array_split(rng.permutation(t), int(np.ceil(len(t)/384))):
                h = model.context(x[ix])
                # Uniform prefix length, uniform subset of that length.
                count = rng.integers(0, SERVICES, len(ix))[:,None]
                rank = np.argsort(rng.random((len(ix), SERVICES)), axis=1)
                mask = torch.from_numpy((rank < count) & (not independent))
                logits = model.decode(h, codes[ix], mask, target[ix])
                ce = F.cross_entropy(logits.transpose(1,2), codes[ix], reduction='none')
                conditional = ((ce * ~mask).sum(1)/(~mask).sum(1)).mean()
                loss = conditional + F.binary_cross_entropy_with_logits(model.prior(h).flatten(), target[ix])
                opt.zero_grad()
                loss.backward()
                opt.step()
        p, ll = evaluate(model, x, codes, q, independent)
        start = 1 if independent else 0
        score.append(p[:,start:])
        likelihood.append(ll[:,start:,:])
    return np.column_stack(score), np.concatenate(likelihood, axis=1)


def protocol(x, bills, y, groups, rows, fold, tag):
    out = ROOT/'artifacts/nov24_service_order_mixture'
    out.mkdir(exist_ok=True)
    scores = np.empty((len(rows), 3))
    ll = np.empty((len(rows), 3, 2))
    for f in np.unique(fold):
        held = fold == f
        q = rows[held]
        t = rows[~held & ~np.isin(groups[rows], groups[q])]
        assert not np.isin(groups[t],groups[q]).any()
        path = out/f'{tag}_{f}.npz'
        if not path.exists():
            p, density = fit(x, bills, y, t, q)
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, train=t, query=q, truth=y[t], p=p, density=density)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for k,v in [('train',t),('query',q),('truth',y[t])]:
                np.testing.assert_array_equal(z[k],v)
            scores[held], ll[held] = z['p'], z['density']
        print(tag,int(f),flush=True)
    assert np.isfinite(scores).all() and np.isfinite(ll).all()
    return scores, ll


def decisions(scores, base, protected):
    results = {}
    for j,name in enumerate(['independent', 'empty_prefix', 'all_orders']):
        results[name] = scores[:,j] >= .5
        results[name+'_quarter'] = np.where((base >= .5) != protected, protected,
                                            .75*base+.25*scores[:,j] >= .5)
    logits = logit(scores.clip(.00001,.99999))
    change = logits[:,2]-logits[:,1]
    results['dependence_only'] = np.where((base >= .5) != protected, protected,
                                          logit(base.clip(.00001,.99999))+.25*change >= 0)
    return results


def run():
    _, oy, og, rows, ofold, _, champion, _, _, _, _, old, _ = aligned()
    raw,y,g,fold = data()
    names, half, full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    gate, _ = gate_probabilities(names,half,full,y,g,fold)
    np.testing.assert_array_equal(gate >= .5,ref)
    x,bills,_ = inputs(raw)
    a, _ = protocol(x,bills,y,g,rows,ofold,'historical')
    b, ll = protocol(x,bills,y,g,np.arange(len(y)),fold,'full')
    previous, current = decisions(a,old[:,0],champion), decisions(b,gate,ref)
    unseen = ~np.isin(g,og)
    common = ((full >= .5) != y[:,None]).all(1)
    hard = common[rows] & (champion != oy)
    _, inv, counts = np.unique(g,return_inverse=True,return_counts=True)
    solo = counts[inv] == 1
    report = dict(conditional_bill_nll=(-ll[np.arange(len(y)),:,y.astype(int)]).mean(0).tolist(), results={})
    for name, p in current.items():
        result = dict(full=stats(p,ref,y,g,fold), historical=stats(previous[name],champion,oy,og,ofold),
                      unseen=stats(p[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                      common_fixed=int((common & (p == y)).sum()), solo_missed_recovered=int((common & y & solo & p).sum()),
                      persistent_both=int((hard & (previous[name] == oy) & (p[rows] == oy)).sum()))
        report['results'][name] = result
        print(name,[result[k]['net'] for k in ('historical','full','unseen')],result['common_fixed'],
              result['solo_missed_recovered'],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/service_order_mixture.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    torch.set_num_threads(1)
    run()
