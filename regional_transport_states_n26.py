"""Test for persistent local shifts in transport risk along booking and berth coordinates."""
import itertools
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from audit_fare_routes_n26 import DECK, held_fare
from export_fare_deck_n26 import combine
from full_roster_booking_validation_n23 import ROOT, data
from solo_archive_recovery_n26 import ticker
from tabicl_consensus_review_n26 import OUT as REVIEW, sources, splits

STATES = np.array([-1.4, 0., 1.4])
PRIOR = np.array([.20, .60, .20])
PATH = ROOT/'reports/nov23_matched_learning_curve/regional_transport_states.json'


def transition(gap, scale):
    persistence = np.exp(-gap / scale)
    return persistence * np.eye(len(PRIOR)) + (1-persistence) * PRIOR


def smooth(positions, likelihood, scale):
    persistence = np.exp(-np.diff(positions)/scale)
    forward = np.empty_like(likelihood)
    msg = PRIOR.copy()
    for i, ev in enumerate(likelihood):
        msg = msg * ev
        forward[i] = msg / msg.sum()
        if i < len(persistence):
            msg = persistence[i]*forward[i] + (1-persistence[i])*PRIOR
    msg = np.ones(len(PRIOR))
    for i in range(len(positions)-1, -1, -1):
        forward[i] *= msg
        forward[i] /= forward[i].sum()
        e = likelihood[i]*msg
        if i:
            msg = persistence[i-1]*e + (1-persistence[i-1])*(PRIOR@e)
            msg /= msg.max()
    return forward


def check_smoother():
    positions = np.array([1.,2.,12.,18.])
    ev = np.array([[.6,1.,1.2],[1.,1.,1.],[1.,.7,.3],[.3,.2,1.]])
    expected = np.zeros_like(ev)
    for seq in itertools.product(range(3), repeat=4):
        w = PRIOR[seq[0]]*ev[0,seq[0]]
        for i in range(1,4):
            w *= transition(positions[i]-positions[i-1],8)[seq[i-1],seq[i]]*ev[i,seq[i]]
        expected[np.arange(4),seq] += w
    np.testing.assert_allclose(smooth(positions,ev,8), expected/expected.sum(1,keepdims=True), atol=1e-12)
    np.testing.assert_allclose(smooth(positions, np.ones_like(ev),8), np.tile(PRIOR,(4,1)), atol=1e-12)


def probabilities(p):
    return expit(logit(np.clip(p,.001,.999))[:,None]+STATES)


def effects(train, query, group, lane, position, scale, pt, pq, truth):
    assert not np.isin(group[train],group[query]).any()
    chance = probabilities(pt)
    logs = np.log(np.where(truth[:,None],chance,1-chance))
    logs -= logs[:,1:2]
    obs = pd.DataFrame(logs, columns=['lower','center','upper'])
    obs['group'], obs['lane'], obs['position'] = group[train], lane[train], position[train]
    # Multiple passengers in a booking give a single average log likelihood on each lane.
    evidence = obs.groupby(['lane','group'],as_index=False).agg(dict(
        position='median',lower='mean',center='mean',upper='mean'))
    delta = {k:np.zeros(len(query)) for k in ('real','shuffled')}
    for label, sub in evidence.groupby('lane'):
        select = np.flatnonzero((lane[query]==label)&np.isfinite(position[query]))
        sub = sub.dropna(subset=['position']).sort_values(['position','group'])
        if not len(select) or len(sub)<6:
            continue
        coords = sub.position.to_numpy()
        query_pos = position[query[select]]
        grid = np.unique(np.r_[coords,query_pos])
        additions = sub[['lower','center','upper']].to_numpy()
        permutation = np.arange(len(sub))
        rng = np.random.default_rng(9322)
        # Preserve coarse location while disrupting the order of the individual booking residuals.
        for block in np.array_split(permutation,max(1,len(permutation)//48)):
            permutation[block] = rng.permutation(block)
        for mode, vals in [('real',additions),('shuffled',additions[permutation])]:
            log_emit = np.zeros((len(grid),len(PRIOR)))
            np.add.at(log_emit,np.searchsorted(grid,coords),vals)
            emit = np.exp(log_emit-log_emit.max(1,keepdims=True))
            weights = smooth(grid,emit,scale)[np.searchsorted(grid,query_pos)]
            cp = probabilities(pq[select])
            shifted = (cp*weights).sum(1)
            stationary = cp@PRIOR
            delta[mode][select] = logit(shifted)-logit(stationary)
    return delta


def feature_matrix(v, idx, bank):
    occupied = np.clip(bank['neural_deck_fine'][idx],.005,.995)
    tree = np.clip(bank['tree_deck_fine'][idx],.005,.995)
    native = v['off'][idx]*np.where(v['anchor'][idx],-1,1)
    return np.column_stack([native,logit(occupied),logit(tree),v['anchor'][idx],
                            v['common'][idx],native*v['common'][idx]])


def run():
    check_smoother()
    raw,y,g,_=data()
    panels=sources()[-1]
    old=held_fare(panels['historical'],'historical',with_routes=False)
    cur=held_fare(panels['full'],'full',with_routes=False)
    cabin=raw.Cabin.str.split('/',expand=True)
    track=(cabin[0]+cabin[2]).fillna('?').to_numpy()
    berth=pd.to_numeric(cabin[1]).to_numpy()
    booking=g.astype(float)
    with np.load(DECK/'deck_estimates.npz') as z:
        eligible=z['eligible'][:len(y)]
        agrees=(z['label'][:len(y)]>=0)&(z['fine'][:len(y)].argmax(1)==z['label'][:len(y)])
    bags={}
    for key,src,dst,t,q in splits(panels,g):
        if not (key.startswith('bookingmix') or key=='historical_to_new'):
            continue
        a,b=src['ids'][t],dst['ids'][q]
        xt,xq=feature_matrix(src,t,old),feature_matrix(dst,q,cur)
        est=LogisticRegression(C=1,max_iter=300).fit(xt,y[a])
        pt,pq=est.predict_proba(xt)[:,1],est.predict_proba(xq)[:,1]
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],b)
            review=dst['anchor'][q]^((z['identities_error']>.5)&(z['identities_outcome']>.5))
        parent=combine(review,cur['neural_deck_raw'][q],cur['neural_deck_fine'][q],eligible[b])>=.5
        shifts={}
        for name, lanes, pos, scale in [('booking_order',np.repeat('all',len(raw)),booking,40),
                                        ('berth',track,berth,8),('opposite_sides_pooled',cabin[0].fillna('?').to_numpy(),berth,8)]:
            result=effects(a,b,g,lanes,pos,scale,pt,pq,y[a])
            for n,delta in result.items():
                shifts[name+'_'+n]=delta
        for mode in ('real','shuffled'):
            shifts['average_'+mode]=np.mean([shifts[k+'_'+mode] for k in ('booking_order','berth','opposite_sides_pooled')],axis=0)
        tag='bookingmix' if key.startswith('bookingmix') else key
        bags.setdefault(tag,[]).append((b,parent,pq,dst['common'][q],shifts))
    report={}
    for tag,items in bags.items():
        ids,parent,p,common=[np.concatenate([item[j] for item in items]) for j in range(4)]
        assert len(np.unique(ids))==len(ids)
        wrong=parent!=y[ids]
        groups={'all':np.ones(len(ids),bool),'unanimous':common,
                'paid_agree_unanimous':common&eligible[ids]&agrees[ids]}
        logprob=lambda pr:np.log(np.where(y[ids],pr,1-pr))
        report[tag]={}
        for name in items[0][4]:
            offset=np.concatenate([item[4][name] for item in items])
            final=expit(logit(p)+offset)
            changed=((p>=.5)==parent)&((final>=.5)!=parent)
            gains=logprob(final)-logprob(p)
            report[tag][name]={k:{**ticker(m,wrong,changed),'mean_log_probability_gain':float(gains[m].mean())}
                               for k,m in groups.items()}
        print(tag,report[tag]['average_real'],flush=True)
    PATH.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
