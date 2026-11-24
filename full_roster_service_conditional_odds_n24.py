"""Evaluate incremental bill likelihoods when the incumbent transport probability is known."""
import json

import numpy as np
import torch
from scipy.special import logit
from torch.nn import functional as F

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_service_order_mixture_n24 import SERVICES, amount_codes, evaluate, ServiceDecoder
from full_roster_zero_truncated_bills_n24 import inputs

MODES = ('registration_only', 'conditional_likelihood', 'conditional_hybrid')


def path_likelihood(model, h, codes, ranks):
    """Both label hypotheses use precisely the same sampled permutation per passenger."""
    n = len(codes)
    rank = torch.as_tensor(ranks)
    steps = torch.arange(SERVICES)[:,None,None]
    prefix = (rank[None,:,:] < steps).expand(SERVICES,n,SERVICES)
    chosen = rank.argsort(dim=1).T
    sides = torch.arange(2, dtype=h.dtype)[:,None,None].expand(2,SERVICES,n)
    masks = prefix.unsqueeze(0).expand(2,-1,-1,-1).reshape(-1,SERVICES)
    bill = codes.repeat(2*SERVICES,1)
    logits = model.decode(h.repeat(2*SERVICES,1), bill, masks, sides.reshape(-1))
    log_probs = torch.log_softmax(logits, dim=-1).gather(2,bill.unsqueeze(-1)).squeeze(-1)
    by_service = log_probs.reshape(2,SERVICES,n,SERVICES)
    next_logp = by_service.gather(3,chosen[None,:,:,None].expand(2,-1,-1,-1)).squeeze(-1)
    return next_logp.sum(1).T


def fit(context, bills, y, t, q, prior, mode, seed=860):
    odds = logit(prior.clip(.0001,.9999))
    if mode != 'registration_only':
        context = np.column_stack([context, odds])
    x = torch.tensor(((context-context[t].mean(0))/context[t].std(0).clip(.25)).clip(-10,10),dtype=torch.float32)
    classes = torch.as_tensor(amount_codes(bills,t))
    truth = torch.tensor(y, dtype=torch.long)
    initial = torch.tensor(odds,dtype=torch.float32)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ServiceDecoder(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.025)
    for epoch in range(30):
        for ids in np.array_split(rng.permutation(t), int(np.ceil(len(t)/256))):
            ranks = np.argsort(rng.random((len(ids),SERVICES)), axis=1)
            ll = path_likelihood(model, model.context(x[ids]), classes[ids], ranks)
            nll = -ll.gather(1,truth[ids,None]).mean()/SERVICES
            if mode == 'conditional_hybrid':
                score = initial[ids]+ll[:,1]-ll[:,0]
                nll = nll+F.binary_cross_entropy_with_logits(score,truth[ids].float())
            opt.zero_grad()
            nll.backward()
            opt.step()
    # evaluate() also returns a standalone posterior. Only conditional bill likelihoods enter this update.
    _, ll = evaluate(model, x, classes, q, False)
    return ll[:,1,:]


def run():
    _, oy, og, rows, _, _, champion, _, _, _, _, _, _ = aligned()
    raw,y,g,fold = data()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    k = names.index('mitra96')
    base = full[:,k]
    common = ((full >= .5) != y[:,None]).all(1)
    hard = common[rows] & (champion != oy)
    _, mapping, counts = np.unique(g, return_inverse=True, return_counts=True)
    solo = counts[mapping] == 1
    unseen = ~np.isin(g,og)
    x, bills, _ = inputs(raw)
    dest = ROOT/'artifacts/nov24_service_conditional_odds'
    dest.mkdir(exist_ok=True)
    report = {}
    for mode in MODES:
        density = np.full((len(y),2), np.nan)
        for f in np.unique(fold):
            t = np.flatnonzero((fold%2 == f%2) & (fold != f))
            q = np.flatnonzero(fold == f)
            assert not np.isin(g[t],g[q]).any()
            prior = half[:,k].copy()
            prior[q] = base[q]
            path = dest/f'{mode}_{f}.npz'
            if not path.exists():
                result = fit(x,bills,y,t,q,prior,mode)
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream,train=t,query=q,truth=y[t],prior=prior[np.r_[t,q]],density=result)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for key,val in [('train',t),('query',q),('truth',y[t]),('prior',prior[np.r_[t,q]])]:
                    np.testing.assert_array_equal(z[key],val)
                density[q] = z['density']
        assert np.isfinite(density).all()
        delta = density[:,1]-density[:,0]
        scores = {}
        for weight in [1.,.25]:
            proposal = logit(base.clip(.0001,.9999))+weight*delta >= 0
            vote = np.where((base >= .5) == ref,proposal,ref)
            scores[str(weight)] = dict(full=stats(vote,ref,y,g,fold),
                        unseen=stats(vote[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                        common_correct=int((common & (vote == y)).sum()),
                        solo_misses_correct=int((common & solo & y & vote).sum()),
                        correct_on_errors_also_missed_by_champion=int((hard & (vote[rows] == oy)).sum()))
        loss = -density[np.arange(len(y)),y.astype(int)].mean()
        report[mode] = dict(bill_nll=float(loss),weights=scores)
        print(mode,loss,[(v['full']['net'],v['common_correct'],v['solo_misses_correct']) for v in scores.values()],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/service_conditional_odds.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    torch.set_num_threads(1)
    run()
