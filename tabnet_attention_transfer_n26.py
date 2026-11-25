"""Check two fixed countervote policies when the training bookings and reviewers change."""
import json

import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import fit
from attentive_tabnet_n26 import OUT
from evaluate_nested_release_n26 import DECK, REVIEW, panel_predictions, protocols, sources, stages
from frozen_queue_shared_errors_n26 import paired
from tabnet_attention_reliability_n26 import network_outputs, signals
from target_mix_expert_routing_n26 import ROOT, batches, ticker

RESULT = ROOT/'reports/nov23_matched_learning_curve/tabnet_attention_transfer.json'


def historical_outputs(raw,y,g,v):
    outputs = None
    for k in np.unique(v['fold']):
        with np.load(OUT/'history'/f'fold{k}.npz') as z:
            q = v['ids'][v['fold']==k]
            pool = v['ids'][v['fold']!=k]
            t = pool[~np.isin(g[pool],g[q])]
            for key, expected in [('q',q),('t',t),('train_y',y[t]),('passenger',raw.PassengerId.iloc[q])]:
                np.testing.assert_array_equal(z[key],expected)
            packet = network_outputs(z)
            if outputs is None:
                outputs = [np.full((len(y),x.shape[1]),np.nan) for x in packet]
            for dst,src in zip(outputs,packet,strict=True):
                dst[q] = src
    return outputs


def design(ordinary, probability, parent):
    votes = logit(probability.clip(.002,.998))*np.where(parent,-1,1)[:,None]
    return np.column_stack([ordinary,votes,votes.std(1),votes.min(1)])


def decide(train,query,old,now,outcome,p1,p2,a1,a2):
    assert all(np.isfinite(x).all() for x in (train,query,p1,p2,a1,a2))
    xt,xq = design(train,p1,old),design(query,p2,now)
    weights = np.where((p1.mean(1)>=.5)!=old,1.,.2)
    challenge = (p2.mean(1)>=.5)!=now
    probability = fit(xt,outcome,xq,weights)>.5
    attention = fit(np.column_stack([xt,a1]),outcome,np.column_stack([xq,a2]),weights)>.5
    return {'attention':challenge&attention,'actual_and_scores':challenge&attention&probability}


def collect():
    raw,y,g,fold,known,versions = paired()
    _,_,_,panels = sources()
    current = panel_predictions(panels['full'],'full')
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled = z['eligible']
    full = signals(raw,y,g,fold)
    hist = historical_outputs(raw,y,g,panels['historical'])
    accumulated = {}
    for details,b in zip(protocols(panels,g,True),batches(True),strict=True):
        key,src,dst,it,iq = details
        t,q = b.train_ids,b.ids
        np.testing.assert_array_equal(t,src['ids'][it])
        np.testing.assert_array_equal(q,dst['ids'][iq])
        assert known[q].all() and not np.isin(g[t],g[q]).any()
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            risk = np.column_stack([z['identities_error'],z['identities_outcome']])
        anchor,review,fare,now = stages(dst,current,iq,enabled[q],risk)
        np.testing.assert_array_equal(anchor,versions[q,6])
        np.testing.assert_array_equal(now,b.query_cell%2)
        common = (versions[q,:7]==now[:,None]).all(1)&(review==now)&(fare==now)
        old = b.train_cell%2!=0
        p,a = full if b.tag=='full' else hist
        actions = decide(b.train_fields,b.query_fields,old,now,b.train_error,p[t],full[0][q],a[t],full[1][q])
        accumulated.setdefault(b.tag,[]).append((q,common,now,actions))
    return accumulated,y,g,fold,versions


def run():
    accumulated,y,_,fold,_ = collect()
    report = {}
    for tag,items in accumulated.items():
        q,common,parent = (np.concatenate([i[j] for i in items]) for j in range(3))
        assert np.unique(q).size==len(q)
        wrong = y[q]!=parent
        report[tag] = {}
        for method in items[0][3]:
            act = np.concatenate([i[3][method] for i in items])&common
            delta = act*np.where(wrong,1,-1)
            result = ticker(common,wrong,act)
            result.update(total_rows=len(q),parent_accuracy=float((~wrong).mean()),
                          accuracy=float((~(wrong^act)).mean()),gain_pp=float(100*delta.mean()),
                          net_by_fold=[int(delta[fold[q]==k].sum()) for k in range(8)])
            report[tag][method] = result
            print(tag,method,result,flush=True)
    old = json.loads((RESULT.parent/'tabnet_attention_reliability.json').read_text())
    for name,res in report['full'].items():
        for metric in ['fixed','harmed','errors','rows']:
            assert res[metric]==old[name]['common'][metric]
    assert [report[tag]['attention']['total_rows'] for tag in ('full','bookingmix','historical_to_new')]==[8532,6055,1033]
    RESULT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
