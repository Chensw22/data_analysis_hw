"""Bound the gains from one hard expert per observable group and replay its actual order-held selection."""
import json
from math import ceil

import numpy as np
import pandas as pd

from awake_paid_soft_unanimity_n26 import original_scores
from evaluate_nested_release_n26 import DECK, OUT, panel_predictions, protocols, sources, stages
from export_consensus_review_n26 import dig
from export_expert_vote_panel_n26 import OUT as PANEL, REPORT as PANEL_REPORT
from frozen_queue_shared_errors_n26 import RELEASES, paired
from price_implied_deck_n26 import FEES
from six_way_against_scored_nine_n26 import test_decisions
from target_mix_expert_routing_n26 import ROOT

REPORT=ROOT/'reports/nov23_matched_learning_curve/grouped_expert_choice_ceiling.json'


def fields(raw,base):
    total=raw[FEES].sum(axis=1).to_numpy()
    positive=raw[FEES].gt(0).any(axis=1).to_numpy()
    deck=raw.Cabin.str.split('/').str[0]
    deck=np.where(deck.eq('G'),'G',np.where(deck.eq('F'),'F','other_or_missing'))
    amt=np.where(~positive,'no_positive',np.where(total<500,'1_499',np.where(total<1500,'500_1499','1500_plus')))
    base=np.where(base,'positive','negative')
    categories={
        'one':[np.repeat('all',len(raw))],
        'parent':[base],
        'parent_paid':[base,np.where(positive,'positive_spend','no_positive')],
        'parent_spend':[base,amt],
        'parent_spend_deck':[base,amt,deck],
    }
    return {k:np.array(['|'.join(vals) for vals in zip(*v,strict=True)]) for k,v in categories.items()}


def bank(p,base,raw):
    edits=(p>=.5)!=base[:,None]
    eligible=raw[FEES].gt(0).any(axis=1).to_numpy()
    edits[~eligible,-2:]=False
    return np.column_stack([np.zeros(len(base),bool),edits])


def best(group_edits,wrong):
    gains=group_edits.T @ np.where(wrong,1,-1)
    winner=int(np.lexsort((np.arange(len(gains)),group_edits.sum(0),-gains))[0])
    return winner


def metrics(edits,wrong,fold,correct,n):
    fixed=edits&wrong
    harmed=edits&~wrong
    net=fixed.astype(int)-harmed
    return dict(fixed=int(fixed.sum()),harmed=int(harmed.sum()),net=int(net.sum()),
                overall_correct=correct+int(net.sum()),overall_accuracy=float((correct+net.sum())/n),
                net_by_fold=[int(net[fold==k].sum()) for k in np.unique(fold)])


def run():
    raw,y,g,f,known,versions=paired()
    parent=versions[:,-1]
    common=known&(versions==parent[:,None]).all(1)
    wrong=y!=parent
    score,names=original_scores(y,g,f)
    assert names[-2:]==['register_paid','sequence_paid']
    edits=bank(score,parent,raw)
    names=['keep',*names]
    cells=fields(raw,versions[:,0])
    test=pd.read_csv(ROOT/'data/raw/test.csv')
    tc={k:test_decisions(v,test.PassengerId)[0] for k,v in RELEASES.items()}
    tcommon=(np.column_stack(list(tc.values()))==tc['six'][:,None]).all(1)
    tfields=fields(test,tc['nine'])
    assert dig(PANEL/'scores.npz')==json.loads(PANEL_REPORT.read_text())['sha256']
    with np.load(PANEL/'scores.npz') as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        np.testing.assert_array_equal(z['prediction'],tc['six'])
        np.testing.assert_array_equal(z['names'],names[1:])
        tedits=bank(z['scores'],tc['six'],test)
    correct=int((known&~wrong).sum())
    n=int(known.sum())
    target=ceil(.85*n)
    assert n==8532 and common.sum()==7891 and tcommon.sum()==3929 and int((common&wrong).sum())==1181
    report=dict(rows=n,baseline_correct=correct,target_correct=target,net_needed=target-correct,
                baseline_accuracy=correct/n,common_rows=int(common.sum()),test_rows=len(test),
                common_test_rows=int(tcommon.sum()),strata={},reference='The six-stage parent is kept outside the ten-way unanimous rows.',
                coverage_order='Only outcome experts are eligible; keep parent on ties if no change wins; two paid-only experts abstain on no-positive-spend rows.',
                inference='The hindsight upper bound applies only to the stated fixed strata and one of 99 hard expert decisions or keep per stratum. Finer partitions, soft combinations, new models and inter-expert rules are different classes. Membership covers all test rows but 8532 known training rows.',
                test_interpretation='The test target is unavailable: population counts and changes quantify reach only; no public leaderboard transfer is assumed.')
    _,_,_,panels=sources()
    standard=panel_predictions(panels['full'],'full')
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
    bags=[]
    seen=np.zeros(len(y),bool)
    for key,src,_,t,q in protocols(panels,g,include_full=True):
        if not key.startswith('full_'):
            continue
        assert np.unique(f[np.r_[t,q]]).size==1 and not np.isin(g[t],g[q]).any() and not seen[q].any()
        with np.load(OUT/f'{key}.npz') as z:
            np.testing.assert_array_equal(z['ids'],t)
            np.testing.assert_array_equal(z['outer'],q)
            for k in np.unique(z['fold']):
                assert not np.isin(g[t[z['fold']==k]],g[t[z['fold']!=k]]).any()
            anchor,review,fare,final=stages(src,standard,t,enabled[t],z['risk'])
        np.testing.assert_array_equal(anchor,versions[t,6])
        ok=(versions[t,:7]==final[:,None]).all(1)&(review==final)&(fare==final)
        train=t[ok]
        # Every training row kept by nested replay agrees with the scored nine. The
        # outer-final six-stage value need not agree; reconstruct edits versus nine.
        np.testing.assert_array_equal(versions[train,0],final[ok])
        bags.append((train,q[common[q]]))
        seen[q]=True
    np.testing.assert_array_equal(seen,known)
    train_edits=bank(score,versions[:,0],raw)
    train_wrong=y!=versions[:,0]
    for scheme,labels in cells.items():
        test_labels=tfields[scheme]
        observed=np.union1d(labels[common],test_labels[tcommon])
        oracle=np.zeros(len(y),bool)
        held=np.zeros(len(y),bool)
        entries={}
        for name in observed:
            subset=common&(labels==name)
            ts=tcommon&(test_labels==name)
            win=best(edits[subset],wrong[subset])
            oracle[subset]=edits[subset,win]
            for train,query in bags:
                tr=train[labels[train]==name]
                q=query[labels[query]==name]
                choice=best(train_edits[tr],train_wrong[tr])
                held[q]=edits[q,choice]
            entries[name]=dict(local_rows=int(subset.sum()),test_rows=int(ts.sum()),winner=names[win],
                               local_fixed=int((oracle&subset&wrong).sum()),local_harmed=int((oracle&subset&~wrong).sum()),
                               edits_on_test_rows_using_hindsight_winner=int(tedits[ts,win].sum()))
        o=metrics(oracle,wrong,f,correct,n)
        h=metrics(held,wrong,f,correct,n)
        tv=sum(abs(v['local_rows']/common.sum()-v['test_rows']/tcommon.sum()) for v in entries.values())/2
        report['strata'][scheme]=dict(cells=entries,oracle=o,order_held=h,
             optimistic_gap_to_85=target-o['overall_correct'],held_gap_to_85=target-h['overall_correct'],
             test_edits_with_hindsight_winners=sum(v['edits_on_test_rows_using_hindsight_winner'] for v in entries.values()),
             local_test_population_total_variation=float(tv))
    reach=common&wrong&edits.any(1)
    report['separate_passenger_oracle']=dict(repairable=int(reach.sum()),unreachable_errors=int((common&wrong&~edits.any(1)).sum()),
        overall_accuracy_if_every_available_correction_were_known=(correct+int(reach.sum()))/n,
        test_rows_with_any_challenger=int((tcommon&tedits.any(1)).sum()))
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print('baseline',report['baseline_correct'],'need',report['net_needed'])
    for s,r in report['strata'].items():
        print(s,r['oracle']['net'],r['order_held']['net'],r['test_edits_with_hindsight_winners'],flush=True)


if __name__=='__main__':
    run()
