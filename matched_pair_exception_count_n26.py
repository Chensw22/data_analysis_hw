"""Estimate the number of exceptions in a label-blind pair and separately evaluate ordering given an exception."""
import json
import numpy as np
from scipy.special import expit, logit, softmax
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from frozen_queue_shared_errors_n26 import paired
from held_expert_portfolios_n26 import sources
from evaluate_nested_release_n26 import protocols
from labelled_matched_order_reference_n26 import fit_system,conditional_system
from localized_opposite_order_pairs_n26 import coordinates
from matched_case_ordering_n26 import forecasts,pair
from manifest_sleep_contrast_n26 import design
from price_implied_deck_n26 import FEES
from target_mix_expert_routing_n26 import ROOT,ticker


def make_edges(ids,g,geometry,seed):
    a,b = pair(ids,g,geometry,seed)
    return a,b,np.square(geometry[a]-geometry[b]).sum(1)


def run():
    raw,y,g,f,known,w = paired()
    panel = sources()[-1]
    signals = forecasts(raw,y,g,f)
    p = expit(np.median(signals[:,:3],axis=1)).clip(.01,.99)
    geom = np.column_stack([design(raw,g),np.log1p(raw[FEES].to_numpy(float))])
    report = {}
    paid = raw[FEES].gt(0).any(axis=1).to_numpy()
    for usual, cohort in [(True,raw.CryoSleep.eq(True).to_numpy() & ~paid),
                          (False,raw.CryoSleep.eq(False).to_numpy() & paid)]:
        covered = known & cohort & (w==usual).all(1)
        records = []
        flip = np.zeros(len(y),bool)
        at_most_one = flip.copy()
        seen = flip.copy()
        for key, _, _, t, q in protocols(panel,g,include_full=True):
            if not key.startswith('full_'):
                continue
            t,q = t[cohort[t]],q[cohort[q]]
            assert not np.isin(g[t],g[q]).any() and np.unique(f[np.r_[t,q]]).size == 1
            x = coordinates(geom,t)
            edges = tuple(np.concatenate(z) for z in zip(*[make_edges(t,g,x,s) for s in (1403,2159,3217)],strict=True))
            count_model, ordering = fit_system(p,signals,t,y,edges)
            a,b,gap = make_edges(q,g,x,3917)
            prior,features,dx = conditional_system(p,signals,a,b,gap)
            prior[:,1:] += features @ count_model
            mass = softmax(prior,axis=1)
            rank = expit(logit(p[a])-logit(p[b])+dx@ordering)
            if usual:
                mass = mass[:,::-1]
                rank = 1-rank
            ra = mass[:,2]+mass[:,1]*rank
            rb = mass[:,2]+mass[:,1]*(1-rank)
            keep = covered[a] & covered[b]
            ai,bi = a[keep],b[keep]
            assert not np.any(seen[np.r_[ai,bi]])
            seen[np.r_[ai,bi]] = True
            flip[ai],flip[bi] = ra[keep]>.5,rb[keep]>.5
            at_most_one[ai] = (ra[keep]>.5) & (ra[keep]>=rb[keep])
            at_most_one[bi] = (rb[keep]>.5) & (rb[keep]>ra[keep])
            for i,j,risk,choice in zip(ai,bi,1-mass[keep,0],rank[keep],strict=True):
                records.append((int(y[i]!=usual)+int(y[j]!=usual),risk,float(choice),bool(y[i]!=usual)))
        counts,risks,choices,label_a = np.array(records).T
        actual_any = counts>0
        only_one = counts==1
        select_a = choices>=.5
        right_when_one = select_a==label_a
        report[str(usual)] = dict(pairs=len(records),exceptions_per_pair=np.bincount(counts.astype(int),minlength=3).tolist(),
                auc_for_any_exception=float(roc_auc_score(actual_any,risks)),
                top_quarter_pairs=int(np.ceil(len(records)/4)), top_quarter_with_exception=int(
                    actual_any[np.argsort(-risks,kind='stable')[:int(np.ceil(len(records)/4))]].sum()),
                correctly_ordered_if_exactly_one_known=int((only_one & right_when_one).sum()),
                correctly_ordered_fraction_if_exactly_one=float(right_when_one[only_one].mean()),
                marginal_actions=ticker(seen,y!=usual,flip),at_most_one=ticker(seen,y!=usual,at_most_one))
        print(usual,report[str(usual)],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/matched_pair_exception_count.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
