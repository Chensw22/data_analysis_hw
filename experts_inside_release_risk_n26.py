"""Test fixed expert actions and honest source-order selection within the independently identified risk tail."""
import json
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import fit
from evaluate_nested_release_n26 import OUT, protocols, stages
from frozen_queue_shared_errors_n26 import paired
from held_expert_portfolios_n26 import DECK,NEW,panel_predictions,sources
from target_mix_expert_routing_n26 import ROOT,batches,ticker


def best_action(actions,wrong,selection):
    # The first column always retains the release. Ties prefer fewer edits.
    vals = actions[selection].T @ np.where(wrong[selection],1.,-1.)
    return int(np.lexsort((np.arange(len(vals)),actions[selection].sum(0),-vals))[0])


def run():
    raw,y,g,fold,known,versions = paired()
    parent = versions[:,-1]
    common = (versions==parent[:,None]).all(1)
    with np.load(ROOT/'artifacts/nov26_independently_calibrated_release_risk/held_risk.npz') as z:
        np.testing.assert_array_equal(np.sort(z['ids']),np.flatnonzero(known))
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[z['ids']])
        high = np.zeros(len(y),bool)
        high[z['ids']] = z['top20']
    target = common & high & known
    panels = sources()[-1]
    full = panels['full']
    names = ['keep',*full['names'],*NEW]
    assert len(names)==100
    bank = panel_predictions(full,'full')
    full_splits = (s for s in protocols(panels,g,include_full=True) if s[0].startswith('full_'))
    held_batches = (b for b in batches(True) if b.tag=='full')
    all_actions = np.zeros((len(y),len(names)),bool)
    out = {}
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
    for b,(key,src,_,t,q) in zip(held_batches,full_splits,strict=True):
        np.testing.assert_array_equal(q,b.ids)
        np.testing.assert_array_equal(t,b.train_ids)
        with np.load(OUT/f'{key}.npz') as z:
            np.testing.assert_array_equal(z['ids'],t)
            np.testing.assert_array_equal(z['outer'],q)
            _,review,fare,final = stages(src,bank,t,enabled[t],z['risk'])
        np.testing.assert_array_equal(final,b.train_cell%2!=0)
        ag = (versions[t,:7]==final[:,None]).all(1) & (review==final) & (fare==final)
        source_risk = np.empty(len(t))
        for tr,hold in GroupKFold(3,shuffle=True,random_state=433).split(t,groups=g[t]):
            assert not np.isin(g[t[tr]],g[t[hold]]).any()
            source_risk[hold] = fit(b.train_fields[tr],b.train_error[tr],b.train_fields[hold],np.ones(len(tr)))
        source = ag & (source_risk>=np.quantile(source_risk,.8))
        at,aq = b.train_actions[:,:len(names)], b.query_actions[:,:len(names)]
        assert not at[:,0].any() and not aq[:,0].any()
        all_actions[q] = aq
        # Compare one winner over the whole tail against choosing a separate winner per current class.
        for mode in ('shared','per_direction'):
            pooled = np.zeros(len(q),bool)
            confirmed = pooled.copy()
            for c in ([None] if mode=='shared' else [False,True]):
                mt = source if c is None else source & (final==c)
                mq = target[q] if c is None else target[q] & (parent[q]==c)
                winner = best_action(at,b.train_error,mt)
                pooled |= mq & aq[:,winner]
                approved = []
                for a,h in GroupKFold(2,shuffle=True,random_state=9621).split(t,groups=g[t]):
                    win = best_action(at[a],b.train_error[a],mt[a])
                    validation = at[h,win] & mt[h]
                    net = (np.where(b.train_error[h],1,-1)*validation).sum()
                    approved.append(aq[:,win] & (net>0))
                confirmed |= mq & approved[0] & approved[1]
            for name,rule in [('pooled',pooled),('confirmed',confirmed)]:
                out.setdefault(mode+'_'+name,np.zeros(len(y),bool))[q] = rule
    wrong = parent!=y
    assert target.sum()==1268 and (target & wrong).sum()==525
    results = []
    for j,n in enumerate(names):
        a = all_actions[:,j]
        counts = ticker(target,wrong,a)
        counts['expert']=n
        counts['net_by_fold']=[int((np.where(wrong,1,-1)*a*target*(fold==k)).sum()) for k in range(8)]
        results.append(counts)
    results.sort(key=lambda r:(r['harmed']-r['fixed'],r['harmed']))
    policies = {n:ticker(target,wrong,a) for n,a in out.items()}
    report = dict(fixed_experts=results,selectors=policies,
                  any_challenger=ticker(target,wrong,all_actions.any(1)))
    print('fixed',results[:3],'selected',policies,'any',report['any_challenger'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/experts_inside_release_risk.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
