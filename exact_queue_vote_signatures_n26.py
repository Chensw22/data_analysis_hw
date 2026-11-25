"""Bound and cross-fit fixed decisions per hard vote signature on the ten-release agreement region."""
import json
from math import ceil

import numpy as np
import pandas as pd

from evaluate_nested_release_n26 import DECK, OUT, panel_predictions, protocols, sources, stages
from export_consensus_review_n26 import dig
from export_expert_vote_panel_n26 import OUT as PANEL, REPORT as PANEL_REPORT
from frozen_queue_shared_errors_n26 import RELEASES, paired
from grouped_expert_choice_ceiling_n26 import bank, metrics, original_scores
from price_implied_deck_n26 import FEES
from six_way_against_scored_nine_n26 import test_decisions
from target_mix_expert_routing_n26 import ROOT

REPORT=ROOT/'reports/nov23_matched_learning_curve/exact_queue_vote_signatures.json'


def joint_codes(before, votes, paid, tbase, tvotes, tpaid, width, mark_abstention):
    def packed(base,decisions,funded):
        cols=[base,decisions[:,1:width+1]]
        if mark_abstention:
            # Paid-only experts abstain on zero spend. A separate bit distinguishes
            # that state from active agreement. The two have the same eligibility.
            cols.append(funded)
        return np.packbits(np.column_stack(cols),axis=1)
    rows=packed(before,votes,paid)
    tests=packed(tbase,tvotes,tpaid)
    _,index=np.unique(np.vstack([rows,tests]),axis=0,return_inverse=True)
    return index[:len(rows)],index[len(rows):]


def frequency(ids,groups,code,n):
    counts=np.bincount(code[ids],minlength=n)
    pairs=np.unique(np.column_stack([code[ids],groups[ids]]),axis=0)
    books=np.bincount(pairs[:,0],minlength=n)
    return counts,books


def report_support(query,code,counts,books):
    result={}
    for scale,choices in [('passengers',counts),('bookings',books)]:
        weight=choices[code[query]]
        result[scale]={k:int(m.sum()) for k,m in [('unseen',weight==0),('one',weight==1),('two_to_four',(weight>=2)&(weight<5)),('five_plus',weight>=5)]}
        assert sum(result[scale].values())==len(query)
    return result


def nested(y,g,f,versions,known,common):
    panels=sources()[-1]
    bank=panel_predictions(panels['full'],'full')
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
    sets=[]
    seen=np.zeros(len(y),bool)
    for key,src,_,t,q in protocols(panels,g,include_full=True):
        if not key.startswith('full_'):
            continue
        assert not np.isin(g[t],g[q]).any() and np.unique(f[np.r_[t,q]]).size==1 and not seen[q].any()
        with np.load(OUT/f'{key}.npz') as z:
            np.testing.assert_array_equal(z['ids'],t)
            np.testing.assert_array_equal(z['outer'],q)
            for k in np.unique(z['fold']):
                assert not np.isin(g[t[z['fold']==k]],g[t[z['fold']!=k]]).any()
            anchor,review,fare,final=stages(src,bank,t,enabled[t],z['risk'])
        np.testing.assert_array_equal(anchor,versions[t,6])
        m=(versions[t,:7]==final[:,None]).all(1)&(review==final)&(fare==final)
        np.testing.assert_array_equal(versions[t[m],0],final[m])
        sets.append((t[m],q[common[q]]))
        seen[q]=True
    np.testing.assert_array_equal(seen,known)
    return sets


def summarize(codes,testcodes,known,common,wrong,nine_wrong,group,fold,tests,nested_sets,can_flip):
    n=int(max(codes.max(),testcodes.max()))+1
    ids=np.flatnonzero(common)
    counts,books=frequency(ids,group,codes,n)
    positives=np.bincount(codes[ids],weights=wrong[ids],minlength=n).astype(int)
    signed=2*positives-counts
    mistake_given_same_signature=np.minimum(positives,counts-positives)
    gains=signed.clip(0)
    group_count=counts>0
    correct=int((known&~wrong).sum())
    total=int(known.sum())
    threshold=ceil(total*.85)
    edits=common & (signed[codes]>0)
    oracle=metrics(edits,wrong,fold,correct,total)
    assert int(gains.sum())==oracle['net'] and int(mistake_given_same_signature.sum())==int(common[wrong].sum())-oracle['net']
    cold={}
    supported_by=np.zeros(len(known),int)
    for needed in (1,2):
        take=np.zeros(len(known),bool)
        for t,q in nested_sets:
            cnt,ordcnt=frequency(t,group,codes,n)
            err=np.bincount(codes[t],weights=nine_wrong[t],minlength=n)
            take[q]=(ordcnt[codes[q]]>=needed)&(2*err[codes[q]]>cnt[codes[q]])
            supported_by[q]=ordcnt[codes[q]]
        cold['at_least_'+str(needed)+'_booking']=metrics(take,wrong,fold,correct,total)
    rare={}
    for key,m in [('one_passenger',counts==1),('one_booking',books==1),('under_five_bookings',books<5),('at_least_five_bookings',books>=5)]:
        mask=group_count&m
        rare[key]=dict(signatures=int(mask.sum()),rows=int(counts[mask].sum()),oracle_net=int(gains[mask].sum()),
                       best_total_accuracy_if_only_this_support_can_flip=float((correct+gains[mask].sum())/total))
    return dict(distinct_signatures=int(group_count.sum()),conflicting_signatures=int(((positives>0)&(positives<counts)).sum()),
                passengers_in_conflicting_signatures=int(counts[(positives>0)&(positives<counts)].sum()),
                unavoidable_common_errors=int(mistake_given_same_signature.sum()),oracle=oracle,
                oracle_net_if_dissent_required=int((edits & can_flip & wrong).sum()-(edits & can_flip & ~wrong).sum()),
                upper_net_needed_for_target=threshold-correct,oracle_net_by_support=rare,
                test_support_in_local_common=report_support(tests,testcodes,counts,books),
                order_held=cold,held_query_in_unseen_signatures=int((common&(supported_by==0)).sum()),
                held_query_support_of_one_order=int((common&(supported_by==1)).sum()),
                test_passengers_changed_by_hindsight_choice=int((signed[testcodes[tests]]>0).sum()))


def run():
    raw,y,g,f,known,versions=paired()
    nine,parent=versions[:,0],versions[:,-1]
    common=known&(versions==parent[:,None]).all(1)
    wrong=y!=parent
    score,names=original_scores(y,g,f)
    assert names[-2:]==['register_paid','sequence_paid']
    actions=bank(score,nine,raw)
    paid=raw[FEES].gt(0).any(axis=1).to_numpy()
    test=pd.read_csv(ROOT/'data/raw/test.csv')
    test_votes=np.column_stack([test_decisions(v,test.PassengerId)[0] for v in RELEASES.values()])
    tbase=test_votes[:,0]
    same=(test_votes==tbase[:,None]).all(1)
    assert int(same.sum())==3929 and int(common.sum())==7891
    assert dig(PANEL/'scores.npz')==json.loads(PANEL_REPORT.read_text())['sha256']
    with np.load(PANEL/'scores.npz') as z:
        np.testing.assert_array_equal(z['names'],names)
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        np.testing.assert_array_equal(z['prediction'],test_votes[:,-1])
        ta=bank(z['scores'],tbase,test)
    tp=test[FEES].gt(0).any(axis=1).to_numpy()
    _,grp=np.unique(g,return_inverse=True)
    sets=nested(y,g,f,versions,known,common)
    report=dict(local_total=int(known.sum()),local_common=int(common.sum()),local_common_errors=int((wrong&common).sum()),
                test_total=len(test),test_common=int(same.sum()),
                key='Incumbent decision plus identity of all experts opposing it. The 99-column primary key also records paid-only expert applicability, distinguishing an abstention and an active agreement.',
                policies='Hindsight chooses empirical majority for each signature. Held lookup chooses the training passenger majority after nested order-excluded release replay; ties and unseen signatures retain the parent. Each test passenger contributes only to unlabeled support counts.',
                families={})
    for label,width,mark in [('18_family',18,False),('99_applicable',99,True),('99_abstain_as_agreement',99,False)]:
        codes,tcodes=joint_codes(nine,actions,paid,tbase,ta,tp,width,mark)
        r=summarize(codes,tcodes,known,common,wrong,nine!=y,grp,f,np.flatnonzero(same),sets,actions[:,1:width+1].any(1))
        report['families'][label]=r
        print(label,r['oracle']['net'],r['order_held']['at_least_1_booking']['net'],r['test_support_in_local_common'],flush=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
