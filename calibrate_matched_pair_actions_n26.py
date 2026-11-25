"""Calibrate selection of a pair's more likely exception on a separate set of matched bookings."""
import json
import numpy as np
from scipy.special import expit,logit,softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from evaluate_nested_release_n26 import OUT, protocols, stages
from frozen_queue_shared_errors_n26 import paired
from held_expert_portfolios_n26 import DECK,panel_predictions,sources
from labelled_matched_order_reference_n26 import fit_system,conditional_system
from localized_opposite_order_pairs_n26 import coordinates
from manifest_sleep_contrast_n26 import design
from matched_case_ordering_n26 import forecasts
from matched_pair_exception_count_n26 import make_edges
from price_implied_deck_n26 import FEES
from target_mix_expert_routing_n26 import ROOT,ticker


def pair_signal(ids,allowed,usual,geom,group,p,scores,model):
    a,b,gaps = make_edges(ids,group,geom,3917)
    keep = allowed[a] & allowed[b]
    a,b,gaps = a[keep],b[keep],gaps[keep]
    base,features,dx = conditional_system(p,scores,a,b,gaps)
    base[:,1:] += features@model[0]
    mass = softmax(base,axis=1)
    rank = expit(logit(p[a])-logit(p[b])+dx@model[1])
    if usual:
        mass = mass[:,::-1]
        rank = 1-rank
    member = np.where(rank>=.5,a,b)
    probability = mass[:,2] + mass[:,1]*np.maximum(rank,1-rank)
    return member,np.column_stack([probability,1-mass[:,0]])


def choose(score,correct):
    # The no-change option has zero gain. shrink precision toward the observed rate
    # for proposed members, without using a query label.
    if not len(score):
        return np.inf
    best,threshold = 0.,np.inf
    for value in np.unique(np.r_[.5, np.quantile(score,[.60,.75,.90,.95])]):
        mask = score >= value
        n = int(mask.sum())
        gain = n * (2 * (correct[mask].sum()+5*correct.mean())/(n+5) - 1)
        if gain > best:
            best,threshold = gain,value
    return threshold


def run():
    raw,y,g,f,known,versions = paired()
    scores = forecasts(raw,y,g,f)
    p = expit(np.median(scores[:,:3],axis=1)).clip(.01,.99)
    match = np.column_stack([design(raw,g),np.log1p(raw[FEES].to_numpy(float))])
    panels = sources()[-1]
    bank = panel_predictions(panels['full'],'full')
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled = z['eligible']
    report = {}
    for usual in (True,False):
        paid = raw[FEES].gt(0).any(axis=1).to_numpy()
        cohort = raw.CryoSleep.eq(usual).to_numpy() & (paid!=usual)
        eligible = known & cohort & (versions==usual).all(axis=1)
        counts = np.zeros((len(y),2),int)
        for key,src,_,t,q in protocols(panels,g,include_full=True):
            if not key.startswith('full_'):
                continue
            with np.load(OUT/f'{key}.npz') as z:
                np.testing.assert_array_equal(z['ids'],t)
                np.testing.assert_array_equal(z['outer'],q)
                anchor,*later = stages(src,bank,t,enabled[t],z['risk'])
            np.testing.assert_array_equal(anchor,versions[t,6])
            cal_allowed = np.zeros(len(y),bool)
            cal_allowed[t] = (versions[t,:7]==usual).all(1) & (np.column_stack(later)==usual).all(1)
            t,q = t[cohort[t]],q[cohort[q]]
            for a,b in GroupKFold(2,shuffle=True,random_state=4097).split(t,groups=g[t]):
                fit,cal = t[a],t[b]
                assert not np.isin(g[fit],g[cal]).any() and not np.isin(g[q],g[t]).any()
                geom = coordinates(match,fit)
                edges = tuple(np.concatenate(z) for z in zip(*[make_edges(fit,g,geom,s) for s in (1403,2159,3217)],strict=True))
                model = fit_system(p,scores,fit,y,edges)
                cm,cs = pair_signal(cal,cal_allowed,usual,geom,g,p,scores,model)
                qm,qs = pair_signal(q,eligible,usual,geom,g,p,scores,model)
                for j in range(2):
                    cutoff = choose(cs[:,j],y[cm]!=usual)
                    counts[qm[qs[:,j]>=cutoff],j] += 1
        result = {}
        new = ~np.isin(g,g[panels['historical']['ids']])
        for label,active in [('member_any',counts[:,0]>0),('member_both',counts[:,0]==2),
                             ('pair_any',counts[:,1]>0),('pair_both',counts[:,1]==2),
                             ('both_signals',np.all(counts>0,axis=1))]:
            result[label] = dict(full=ticker(eligible,y!=usual,active),new_orders=ticker(eligible & new,y!=usual,active),
                     net_by_fold=[int(np.sum(active*np.where(y!=usual,1,-1)*(f==k))) for k in np.unique(f)])
        report[str(usual)] = result
        print(usual,[(n,r['full']['fixed'],r['full']['harmed']) for n,r in result.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/calibrate_matched_pair_actions.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
