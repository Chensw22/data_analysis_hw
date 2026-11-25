"""Estimate edit coverage required under a stated transfer from local paired outcomes."""
import json
from math import floor
import numpy as np
import pandas as pd
from arbitration_against_scored_nine_n26 import champions
from outside_panel_coverage_n26 import ROOT,projection
from six_way_against_scored_nine_n26 import read,test_decisions,transfer_edits


def run():
    summary=read('reports/nov23_matched_learning_curve/six_way_against_scored_nine.json')
    best=read('reports/current_best_submission.json')
    receipt=read(best['online_record'])
    assert receipt['evaluation']['Status']=='COMPLETE'
    assert best['submission_ref']==summary['official_ref']==receipt['submission_ref']
    start=float(receipt['evaluation']['Public Score'])
    assert start==summary['public_score']==best['public_score']
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    before,bh=test_decisions(best['proposal'],test.PassengerId)
    after,ah=test_decisions('proposal_n26_six_way_dispute',test.PassengerId)
    assert bh==receipt['file_sha256']==summary['test']['champion_sha256'] and ah==summary['test']['candidate_sha256']
    up,down=int((after&~before).sum()),int((before&~after).sum())
    edits,n=up+down,len(test)
    target=.85
    gap=target-start
    net_needed=floor(n*gap)+1
    required=floor((n*gap+edits)/2)+1
    assert start+(2*(required-1)-edits)/n<=target<start+(2*required-edits)/n
    info=dict(official_ref=best['submission_ref'],starting_public_score=start,target=target,
              assumption='Conditional planning scenario: the all-test edit share, direction mix and held-out direction-specific edit precision transfer to the public evaluation population. The actual public subset and its labels are unknown.',
              test_rows=n,up_edits=up,down_edits=down,edits=edits,net_needed_all_test_equivalent=net_needed,
              minimum_correct_edits=required,required_edit_precision=required/edits,maximum_harms=edits-required,
              conditional_maximum_if_all_edits_correct=start+edits/n,local={})
    # Precision is the fraction of changes which correct the starting champion.
    coverages={}
    for precision in (.55,.60,.65,.70,.75,.80,.90,1.):
        count=floor(n*gap/(2*precision-1))+1
        coverages[str(precision)]=dict(minimum_edits_in_expectation=count,share_of_test=count/n)
    info['coverage_at_assumed_precision']=coverages
    _,y,g,_,views,_=projection(include_full=True)
    nine,known,_,_=champions()
    for tag in ('full','bookingmix','historical_to_new'):
        valid,candidate=views[tag]
        assert not (valid&~known).any()
        ids=np.flatnonzero(valid)
        c,w,group=candidate[ids],nine[ids],g[ids]
        truth=y[ids]
        metrics=transfer_edits(w,c,truth,group,up,down,n)
        np.testing.assert_allclose(metrics['point_gain_pp'],summary['local'][tag]['conditional_on_test_edit_counts']['point_gain_pp'])
        gain=metrics['point_gain_pp']/100
        net=gain*n
        ci=np.array(metrics['cluster_95_pp'])/100
        precision=(1+net/edits)/2
        is_up=~w&c
        is_down=w&~c
        upward=float(np.mean(c[is_up]==truth[is_up]))
        downward=float(np.mean(c[is_down]==truth[is_down]))
        correct=int((c==truth).sum())
        need_local=floor(target*len(ids))+1-correct
        info['local'][tag]=dict(rows=len(ids),up_precision=upward,down_precision=downward,
              observed_candidate_correct=correct,additional_net_to_local_target=max(need_local,0),
              test_mix_precision=precision,precision_cluster_95=((1+ci*n/edits)/2).tolist(),
              scenario_public_score=start+gain,conditional_cluster_95=(start+ci).tolist(),
              gain_pp=100*gain,remaining_gap_pp=100*(gap-gain),equivalent_net=net,
              minimum_edits_at_same_precision=floor(n*gap/(2*precision-1))+1 if precision>.5 else None,
              minimum_down_precision_at_observed_up=(required-up*upward)/down,
              minimum_up_precision_at_observed_down=(required-down*downward)/up,
              bootstrap=metrics['valid_bootstrap_draws'])
    p=ROOT/'reports/nov23_matched_learning_curve/public_target_edit_scale.json'
    p.write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__':
    run()
