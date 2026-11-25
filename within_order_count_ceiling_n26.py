"""Compute sharp oracle bounds for permuting the queued TabNet outcomes within each booking."""
import json
from math import ceil

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from distinct_challenger_error_reach_n26 import population
from export_tabnet_attention_n26 import NAME
from six_way_against_scored_nine_n26 import test_decisions
from tabnet_attention_transfer_n26 import ROOT, collect

REPORT=ROOT/'reports/nov23_matched_learning_curve/within_order_count_ceiling.json'


def count_by(group, mask, size):
    return np.bincount(group,weights=mask.astype(int),minlength=size).astype(int)


def known_bounds(pos,neg,fn,fp,u,inc_fn,inc_fp):
    # x unknown positives can pay for x known negative->positive moves. The other
    # unknown members can instead absorb known positive->negative moves.
    floor=np.minimum(inc_fn,fp)+np.minimum(inc_fp,fn)
    low=np.full(len(u),999,int)
    high=np.full(len(u),-1,int)
    gross_low=low.copy()
    gross_high=high.copy()
    down=np.zeros_like(u)
    up=np.zeros_like(u)
    for x in range(u.max()+1):
        possible=x<=u
        fix_fn=np.minimum(inc_fn,fp+x)
        fix_fp=np.minimum(inc_fp,fn+u-x)
        both=fix_fn+fix_fp
        low[possible]=np.minimum(low[possible],both[possible])
        high[possible]=np.maximum(high[possible],both[possible])
        # Maximum number of target errors corrected if other known passengers may
        # be sacrificed; positives and negatives only constrain the final assignments.
        gross=np.minimum(inc_fn,pos+x)+np.minimum(inc_fp,neg+u-x)
        gross_low[possible]=np.minimum(gross_low[possible],gross[possible])
        gross_high[possible]=np.maximum(gross_high[possible],gross[possible])
        down[possible]=np.maximum(down[possible],fix_fp[possible])
        up[possible]=np.maximum(up[possible],fix_fn[possible])
    np.testing.assert_array_equal(low,floor)
    return dict(max_target_fixes_without_known_harms_unknowns_frozen=int(floor.sum()),
                maximum_target_fixes_without_known_harms_over_unknown_counts=[int(low.sum()),int(high.sum())],
                maximum_target_fixes_allowing_known_harms_over_unknown_counts=[int(gross_low.sum()),int(gross_high.sum())],
                maximum_target_FN_fixes_without_known_harms=int(up.sum()),maximum_target_FP_fixes_without_known_harms=int(down.sum()))


def run():
    raw,y,g,f,old,same,against,_,_,_,solo,paid=population()
    result,truth,groups,fold,versions=collect()
    for a,b in [(y,truth),(g,groups),(f,fold)]:
        np.testing.assert_array_equal(a,b)
    assert np.array_equal(old[same],versions[same,-1])
    pred=np.zeros(len(y),bool)
    known=pred.copy()
    for ids,common,parent,action in result['full']:
        assert not known[ids].any()
        np.testing.assert_array_equal(common,same[ids])
        np.testing.assert_array_equal(parent,versions[ids,-1])
        known[ids]=True
        pred[ids]=parent^(common&action['actual_and_scores'])
    assert known.sum()==8532
    seen_common=known&same&(pred==old)
    blind=known&same&(against==0)
    assert seen_common[blind].all() and (blind&(pred!=y)).sum()==172
    _,idx,roster=np.unique(g,return_inverse=True,return_counts=True)
    nb=len(roster)
    c=lambda m:count_by(idx,m,nb)
    pos=c(known&pred); neg=c(known&~pred)
    un=roster-pos-neg
    fn=c(known & ~pred & y); fp=c(known & pred & ~y)
    truth_sum=c(y)
    error=known&(pred!=y)
    n=int(known.sum())
    correct=int((known&~error).sum())
    required=ceil(.85*n)
    assert (correct,required-correct)==(7091,162)
    partial=(un>0)[idx]
    hom=(pos*neg==0)[idx]
    masks={'known':known,'complete_orders':known & ~partial,'partial_orders':known & partial,
           'solo':known & solo,'multiple':known&~solo,
           'known_members_same_prediction':known&hom,'known_members_mixed_prediction':known&~hom,
           'complete_multiple_same_prediction':known&~solo&~partial&hom,
           'complete_multiple_mixed_prediction':known&~solo&~partial&~hom,
           'all_eleven_agree':seen_common,'any_eleven_disagree':known&~seen_common,
           'all_ten_agreed_before_tabnet':known&same,
           'all_127_challengers_silent':blind,
           'awake_paid_negative_all_127_silent':blind & raw.CryoSleep.eq(False).to_numpy() & paid & ~pred}
    report=dict(candidate=NAME,known_rows=n,known_correct=correct,target_correct=required,known_needed=required-correct,
                raw_rows=len(y),unknown_rows=int((~known).sum()),orders=nb,complete_orders=int((un==0).sum()),
                partial_orders=int(((un>0)&(pos+neg>0)).sum()),all_unknown_orders=int(((pos+neg)==0).sum()),
                known_in_partial=int((known&partial).sum()),scopes={})
    for name,m in masks.items():
        inc_fn=c(error&m&~pred);inc_fp=c(error&m&pred)
        report['scopes'][name]=dict(rows=int(m.sum()),errors=int((m&error).sum()),false_negative=int(inc_fn.sum()),
                false_positive=int(inc_fp.sum()),**known_bounds(pos,neg,fn,fp,un,inc_fn,inc_fp))
    lo,hi=report['scopes']['known']['maximum_target_fixes_without_known_harms_over_unknown_counts']
    np.testing.assert_array_equal(2*np.minimum(fn,fp),fn+fp-np.abs(fn-fp))
    assert lo==2*np.minimum(fn,fp).sum() and hi==lo+np.minimum(abs(fn-fp),un).sum()
    overall=report['known_oracle']=dict(correct_interval=[correct+lo,correct+hi],gain_interval=[lo,hi],
               accuracy_interval=[(correct+lo)/n,(correct+hi)/n],remaining_to_85_interval=[required-correct-hi,required-correct-lo],
               gain_from_unknown_exchanges_at_most=hi-lo)
    # On the full labeled roster, optimal sorting makes only the errors forced by the
    # difference between each booking's true positive count and its fixed baseline count.
    min_gap=np.maximum.reduce([np.zeros(nb,int),pos-truth_sum,truth_sum-pos-un])
    max_gap=np.maximum(abs(pos-truth_sum),abs(pos+un-truth_sum))
    report['whole_roster_oracle']=dict(known_labels_all_rows=len(y),correct_interval=[int(len(y)-max_gap.sum()),int(len(y)-min_gap.sum())],
          accuracy_interval=[float(1-max_gap.sum()/len(y)),float(1-min_gap.sum()/len(y))],target_correct=ceil(.85*len(y)))
    report['interpretation']='Every bound uses only the count conserved within each booking. With all members known, one false negative and one false positive can be repaired together. For unknown predictions we take the minimum and maximum over all possible original positive counts, allowing them to absorb swaps. Subscope no-harm counts permit swaps with other known mistakes, and are independent maxima, not additive. Gross fixes allowing harms may displace a correct passenger; they are not accuracy gains.'
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    vote,sha=test_decisions(NAME,test.PassengerId)
    _,inv,sz=np.unique(test.PassengerId.str.split('_').str[0],return_inverse=True,return_counts=True)
    p=count_by(inv,vote,len(sz))
    report['test_unlabelled']=dict(sha256=sha,orders=len(sz),rows=len(test),solo=int((sz==1).sum()),
             mixed_prediction_orders=int(((p>0)&(p<sz)).sum()),rows_in_mixed_orders=int(sz[(p>0)&(p<sz)].sum()),
             max_changed_rows_from_fixed_counts=int(2*np.minimum(p,sz-p).sum()))
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(overall,report['whole_roster_oracle'],report['test_unlabelled'])
    for n,s in report['scopes'].items():print(n,s['errors'],s['maximum_target_fixes_without_known_harms_over_unknown_counts'])


if __name__=='__main__':
    with threadpool_limits(1):
        run()
