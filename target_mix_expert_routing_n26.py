"""Select an expert on historical bookings by estimated net correct decisions in the new roster."""
import argparse
import json
from types import SimpleNamespace
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_public_transfer_balance_n23 import balanced
from evaluate_nested_release_n26 import DECK, OUT, REVIEW, protocols, sources, panel_predictions, outcome_scores, replay, stages
from full_roster_booking_validation_n23 import ROOT
from held_expert_portfolios_n26 import NEW
from price_implied_deck_n26 import FEES
from release_field_contributions_n26 import evidence
from solo_archive_recovery_n26 import ticker


def bank(v,p,s,idx,risk,enabled,paid):
    anchor,review,fare,parent = stages(v,p,idx,enabled,risk)
    _,_,summary = evidence(v,s,p,idx,risk,enabled)
    actions = (s[idx]>=.5)!=parent[:,None]
    assert NEW[-2:]==['register_paid','sequence_paid']
    actions[~paid,-2:] = False
    unanimous = ~actions.any(1)
    # Include the possibility that every applicable expert agrees on the wrong direction.
    actions = np.column_stack([np.zeros(len(idx),bool),actions,anchor!=parent,review!=parent,fare!=parent,
                               np.ones(len(idx),bool)])
    cell = parent.astype(int)+2*paid.astype(int)+4*unanimous.astype(int)
    return parent,summary,actions,cell


def risk_model(x,y,groups,z):
    def fit(train, target):
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(C=.12,max_iter=450).fit(scaler.transform(x[train]),y[train])
        return model.predict_proba(scaler.transform(target))[:,1]
    oof = np.full(len(x),np.nan)
    for t,q in GroupKFold(3,shuffle=True,random_state=2913).split(x,groups=groups):
        assert not np.isin(groups[t],groups[q]).any()
        oof[q] = fit(t,x[q])
    assert np.isfinite(oof).all()
    return oof, fit(np.arange(len(x)),z)


def gains(t,q,ct,cq,mt,mq,truth,pt,pq,wt,wq):
    action,query = mt[t],mq[q]
    weight = balanced(wt[t],wq[q])
    observed = 2*truth[t].astype(float)-1
    forecast = (2*pq[q]-1)@query/len(q)
    correction = (2*weight*(truth[t]-pt[t]))@action
    assert (ct[t]==cq[q[0]]).all() and (cq[q]==cq[q[0]]).all()
    est = np.stack([observed@action/len(t), (weight*observed)@action, forecast, forecast+correction])
    est[:,~(action.any(0)&query.any(0))] = -np.inf
    est[:,0] = 0.
    return est


def choose(mt,mq,ct,cq,wrong,rt,rq,wt,wq,groups):
    names = ['source','balanced_source','forecast','augmented','both_order_subsets']
    results = np.zeros((len(mq),len(names)),bool)
    split = next(GroupKFold(2,shuffle=True,random_state=3612).split(mt,groups=groups))
    for cell in np.unique(cq):
        t,q = np.flatnonzero(ct==cell),np.flatnonzero(cq==cell)
        if len(t)<8:
            continue
        args = (q,ct,cq,mt,mq,wrong,rt,rq,wt,wq)
        estimates = gains(t,*args)
        best = np.argmax(estimates,axis=1)
        results[q,:4] = mq[q][:,best]
        halves = [np.intersect1d(t,rows) for rows in split]
        if min(map(len,halves))<8:
            continue
        half_est = np.stack([gains(rows,*args)[-1] for rows in halves])
        support = []
        for i in range(2):
            chosen = int(half_est[i].argmax())
            support.append(mq[q,chosen] if half_est[1-i,chosen]>0 else np.zeros(len(q),bool))
        results[q,4] = support[0]&support[1]
    return dict(zip(names,results.T,strict=True))


def batches(include_full):
    raw,y,g,panels = sources()
    p = {name:panel_predictions(v,name) for name,v in panels.items()}
    score = {name:outcome_scores(v,p[name]) for name,v in panels.items()}
    paid = raw[FEES].gt(0).any(axis=1).to_numpy()
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled = z['eligible']
    raw_fields = np.column_stack([raw.booking_size.eq(1),raw.booking_size.ge(4),raw.Deck.eq('G'),
                                 raw.Side.eq('S'),raw.HomePlanet.eq('Earth'),raw.Age.lt(13),raw.Age.isna(),
                                 raw.CryoSleep.eq('True'),raw[FEES].isna().any(axis=1),np.log1p(raw[FEES].sum(axis=1))/8])
    for key,src,dst,t,q in protocols(panels,g,include_full=include_full):
        tag = 'full' if key.startswith('full_') else ('bookingmix' if key.startswith('bookingmix') else key)
        origin = 'full' if tag=='full' else 'historical'
        a,b = src['ids'][t],dst['ids'][q]
        assert not np.isin(g[a],g[b]).any()
        with np.load(OUT/f'{key}.npz') as z:
            np.testing.assert_array_equal(z['ids'],a)
            np.testing.assert_array_equal(z['outer'],b)
            r = z['risk']
        old,xt,at,ct = bank(src,p[origin],score[origin],t,r,enabled[a],paid[a])
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],b)
            r = np.column_stack([z['identities_error'],z['identities_outcome']])
        parent,xq,aq,cq = bank(dst,p['full'],score['full'],q,r,enabled[b],paid[b])
        np.testing.assert_array_equal(parent,replay(key,q,dst,p['full'],enabled))
        wrong = old!=y[a]
        wt,wq = np.column_stack([xt,raw_fields[a]]),np.column_stack([xq,raw_fields[b]])
        strict = ((score['full'][q]>=.5)==parent[:,None]).all(1)
        yield SimpleNamespace(tag=tag,ids=b,query_error=parent!=y[b],train_error=wrong,groups=g[a],
                              train_actions=at,query_actions=aq,train_cell=ct,query_cell=cq,
                              train_fields=wt,query_fields=wq,strict=strict,multiple=raw.booking_size.iloc[b].to_numpy()>1)


def run(include_full):
    modes = {}
    for batch in batches(include_full):
        rt,rq = risk_model(batch.train_fields,batch.train_error,batch.groups,batch.query_fields)
        actions = choose(batch.train_actions,batch.query_actions,batch.train_cell,batch.query_cell,
                         batch.train_error,rt,rq,batch.train_fields,batch.query_fields,batch.groups)
        modes.setdefault(batch.tag,[]).append((batch.ids,batch.query_error,batch.query_cell,batch.strict,batch.multiple,actions))
    report = {}
    for tag,items in modes.items():
        ids,wrong,cells,strict,multiple = [np.concatenate([item[k] for item in items]) for k in range(5)]
        assert len(np.unique(ids))==len(ids)
        masks = {'all':np.ones(len(ids),bool),'original_unanimous':strict,
                 'added_unanimous':(cells>=4)&~strict,'contested':cells<4,'multi_person':multiple}
        r = {}
        for method in items[0][-1]:
            action = np.concatenate([item[-1][method] for item in items])
            r[method] = {name:ticker(mask,wrong,action) for name,mask in masks.items()}
        report[tag] = r
        print(tag,{k:(v['all']['fixed'],v['all']['harmed']) for k,v in r.items()},flush=True)
    if include_full:
        assert [report[tag]['source']['all']['rows'] for tag in ['full','bookingmix','historical_to_new']]==[8532,6055,1033]
        (ROOT/'reports/nov23_matched_learning_curve/target_mix_expert_routing.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--include-full',action='store_true')
    with threadpool_limits(1):
        run(**vars(parser.parse_args()))
