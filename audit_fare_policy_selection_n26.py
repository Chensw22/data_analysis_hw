"""Audit label boundaries and leave-order-fold-out policy selection for the Nov 26 fare change."""
import json
from collections import Counter

import numpy as np
from scipy.special import logit

from nested_release_review_n26 import OUT as NESTED
from price_implied_deck_n26 import OUT as FARE
from tabicl_consensus_review_n26 import OUT as REVIEW, ROOT, sources, splits

RESULT = ROOT/'reports/nov23_matched_learning_curve/fare_policy_selection_audit.json'
NETWORK = ROOT/'artifacts/nov26_tabnet'


def counts(values):
    return dict(fixed=int((values>0).sum()), harmed=int((values<0).sum()), net=int(values.sum()))


def load_base(groups, labels, panels):
    scores, train = {}, {}
    for tag, panel in panels.items():
        scores[tag] = {}
        for k in np.unique(panel['fold']):
            mask = panel['fold']==k
            query = panel['ids'][mask]
            for arch in ['tree','neural','tabnet']:
                path = FARE/f'{arch}_{tag}_{k}.npz'
                if arch=='tabnet':
                    path = (NETWORK/'history' if tag=='historical' else NETWORK)/f'fold{k}.npz'
                with np.load(path) as packet:
                    t = packet['t']
                    np.testing.assert_array_equal(packet['q'],query)
                    np.testing.assert_array_equal(packet['train_y' if arch=='tabnet' else 'y'],labels[t])
                    assert not np.isin(groups[t],groups[query]).any()
                    train[tag,arch,int(k)] = t
                    if arch!='tabnet':
                        for variant in ['raw','fine','coarse','displaced']:
                            p = scores[tag].setdefault((arch,variant),np.zeros(len(panel['ids'])))
                            p[mask] = packet[variant]
    return scores, train


def policies(scores, q, parent, anchor, offset, enabled):
    moves = {}
    for name in ['tree','neural']:
        raw = scores[name,'raw'][q]>=.5
        for variant in ['coarse','fine','displaced']:
            opposes = (scores[name,variant][q]>=.5)!=parent
            moves[name+'_'+variant] = opposes
            moves[name+'_'+variant+'_new'] = opposes&(raw==parent)
        step = logit(scores[name,'fine'][q].clip(.002,.998))-logit(scores[name,'coarse'][q].clip(.002,.998))
        odds = offset*np.where(anchor,-1,1)
        moves[name+'_increment'] = ((odds>=0)==parent)&((odds+step>=0)!=parent)
    moves['both_new'] = moves['tree_fine_new']&moves['neural_fine_new']
    moves['one_new_both_confirm'] = (moves['tree_fine_new']|moves['neural_fine_new'])&moves['tree_fine']&moves['neural_fine']
    return list(moves), np.column_stack(list(moves.values())) & enabled[:,None]


def score_policy(matrix, names, fold, groups, trusted):
    fixed = names.index('neural_fine_new')
    chosen = int(matrix.sum(0).argmax())
    cross = np.zeros(len(fold),int)
    choices = {}
    for k in np.unique(fold):
        t, q = fold!=k, fold==k
        # Historical folds can split a booking; exclude all its other members before selection.
        t &= ~np.isin(groups,groups[q])
        winner = int(matrix[t].sum(0).argmax())
        cross[q] = matrix[q,winner]
        choices[str(k)] = dict(choice=names[winner], held=counts(cross[q]), rows=int(q.sum()))
    return dict(rows=len(fold),bookings=len(np.unique(groups)), audited_policies=len(names), frozen_fare=counts(matrix[:,fixed]),
                global_winner=dict(choice=names[chosen],**counts(matrix[:,chosen])), grouped_outer_selection=dict(**counts(cross),folds=choices),
                historical_choice=dict(choice=names[trusted],**counts(matrix[:,trusted])),
                all_policies={n:counts(col) for n,col in zip(names,matrix.T,strict=True)})


def run():
    _,y,g,panels=sources()
    scores,base_train=load_base(g,y,panels)
    with np.load(FARE/'deck_estimates.npz') as z:
        enabled=z['eligible']
    packets,coverage={},{}
    for key,src,dst,it,iq in splits(panels,g):
        src_name='full' if key.startswith('full_') else 'historical'
        dst_name='historical' if key.startswith('historical_') and key!='historical_to_new' else 'full'
        protocol='historical_to_new' if key=='historical_to_new' else key.split('_')[0]
        t,q=src['ids'][it],dst['ids'][iq]
        item=coverage.setdefault(protocol,Counter())
        item['query_rows']+=len(q)
        item['meta_fits']+=1
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            np.testing.assert_array_equal(z['t'],t)
            np.testing.assert_array_equal(z['train_y'],y[t])
            review=dst['anchor'][iq]^((z['identities_error']>.5)&(z['identities_outcome']>.5))
        item['query_rows_in_direct_meta_train']+=np.isin(q,t).sum()
        item['query_rows_with_booking_in_direct_meta_train']+=np.isin(g[q],g[t]).sum()
        for arch in ('neural','tabnet'):
            hits=np.zeros(len(q),bool)
            for f in np.unique(src['fold'][it]):
                T=base_train[src_name,arch,int(f)]
                hits|=np.isin(g[q],g[T])
            item[f'query_rows_with_booking_in_{arch}_model_generating_meta_train_features']+=hits.sum()
            hits=np.zeros(len(q),bool)
            for f in np.unique(dst['fold'][iq]):
                cells=dst['fold'][iq]==f
                hits[cells]=np.isin(g[q[cells]],g[base_train[dst_name,arch,int(f)]])
            item[f'query_rows_with_booking_in_{arch}_model_generating_query_features']+=hits.sum()
        if protocol!='historical':
            with np.load(NESTED/f'{key}.npz') as inner:
                np.testing.assert_array_equal(inner['ids'],t)
                np.testing.assert_array_equal(inner['outer'],q)
                for h in np.unique(inner['fold']):
                    assert not np.isin(g[t[inner['fold']==h]],g[t[inner['fold']!=h]]).any()
        names,action=policies(scores[dst_name],iq,review,dst['anchor'][iq],dst['off'][iq],enabled[q])
        deltas=action*np.where(review!=y[q],1,-1)[:,None]
        packets.setdefault(protocol,[]).append((q,dst['fold'][iq],deltas))
    archive={tag:[np.concatenate([a[i] for a in chunks]) for i in range(3)] for tag,chunks in packets.items()}
    # Preselect on labeled historical orders alone, apply unchanged on the 1,033 new order passengers.
    hist=archive['historical'];trusted=int(hist[2].sum(0).argmax())
    fresh=archive['historical_to_new'][0]
    assert not np.isin(g[fresh],g[panels['historical']['ids']]).any()
    scores={tag:score_policy(delta,names,fold,g[ids],trusted) for tag,(ids,fold,delta) in archive.items()}
    expected=json.loads((RESULT.parent/'price_implied_deck.json').read_text())['full']['review86']
    for name in names:
        for field in ['fixed','harmed']:
            assert scores['bookingmix']['all_policies'][name][field]==expected[name]['all'][field]
    report=dict(label_lineage={k:{n:int(v) for n,v in row.items()} for k,row in coverage.items()},
                fare_policy_results=scores,scope='Audited fare and TabNet base model training indices; observed labels available to the review/meta calibrator checked against its stored inputs. The 16-rule exercise diagnoses selection reuse, conditional on the existing candidate menu and upstream frozen parent. Only historical_to_new is completely disjoint from historical discovery orders. It is a retrospective check, not a prospective untouched test.')
    RESULT.write_text(json.dumps(report,indent=2)+'\n')
    print({k:(v['frozen_fare'],v['global_winner'],v['grouped_outer_selection']['net'],v['historical_choice']) for k,v in scores.items()})


if __name__=='__main__':
    run()
