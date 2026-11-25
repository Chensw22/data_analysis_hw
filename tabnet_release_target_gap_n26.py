"""Pair the queued TabNet release with the official best and quantify the size of the remaining target."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_flip_capacity_n23 import correct_count, required
from export_tabnet_attention_n26 import NAME,PARENT
from six_way_against_scored_nine_n26 import read,test_decisions,transfer_edits
from tabnet_attention_transfer_n26 import collect,RESULT
from target_mix_expert_routing_n26 import ROOT


def run():
    best=read('reports/current_best_submission.json')
    receipt=read(best['online_record'])
    assert receipt['evaluation']['Status']=='COMPLETE'
    ids=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    champion,bh=test_decisions(best['proposal'],ids)
    candidate,ch=test_decisions(NAME,ids)
    parent,_=test_decisions(PARENT,ids)
    assert bh==best['csv_sha256']==receipt['file_sha256']
    n=len(ids)
    up,down=int((candidate&~champion).sum()),int((~candidate&champion).sum())
    total=up+down
    assert total==306 and ((candidate!=parent)&(champion!=parent)).sum()==0
    denom=read('reports/nov23_matched_learning_curve/queued_target_capacity.json')['compatible_scoring_denominator']
    assert denom==n
    start=correct_count(receipt['evaluation']['Public Score'],n)
    goal=required(n)
    needed=(goal-start+total+1)//2
    evidence={}
    data,y,g,fold,versions=collect()
    checks=json.loads(RESULT.read_text())
    for tag,items in data.items():
        q,common,now=(np.concatenate([part[i] for part in items]) for i in range(3))
        act=common&np.concatenate([part[-1]['actual_and_scores'] for part in items])
        child=now^act
        nine=versions[q,0]
        delta=(child==y[q]).astype(int)-(nine==y[q])
        fixed=int((delta>0).sum())
        harmed=int((delta<0).sum())
        assert float((child==y[q]).mean())==checks[tag]['actual_and_scores']['accuracy']
        tr=transfer_edits(nine,child,y[q],g[q],up,down,n)
        center=.5*(1+tr['point_gain_pp']/100*n/total)
        bounds=(.5*(1+np.asarray(tr['cluster_95_pp'])/100*n/total)).tolist()
        evidence[tag]=dict(rows=len(q),fixed=fixed,harmed=harmed,net=fixed-harmed,edit_precision=fixed/(fixed+harmed),
                           nine_accuracy=float((nine==y[q]).mean()),candidate_accuracy=float((child==y[q]).mean()),
                           net_by_fold=[int(delta[fold[q]==k].sum()) for k in range(8)],
                           required_additional_local_correct_for_above_85=required(len(q))-int((child==y[q]).sum()),
                           test_direction_weighted_edit_precision=center,booking_bootstrap_precision_95=bounds)
        print(tag,evidence[tag],flush=True)
    reach=read('reports/nov23_matched_learning_curve/distinct_challenger_error_reach.json')['scopes']
    blind=reach['all']['none']
    awake=reach['awake_paid_negative']['none']
    # A double-confirmed change needs the TabNet mean to vote against the parent. All three
    # contributing scores are on the parent side for everyone in the completely blind group.
    remaining=evidence['full']['required_additional_local_correct_for_above_85']
    blind_target=dict(errors=blind['wrong'],correct=blind['correct'],local_net_needed=remaining,
                     minimum_share_of_blind_errors_even_if_no_harms=remaining/blind['wrong'],
                     maximum_harms_if_every_blind_error_corrected=blind['wrong']-remaining,
                     additional_needed_even_if_entire_awake_blind_group_fixed=remaining-awake['wrong'])
    report=dict(proposal=NAME,sha256=ch,champion=best['proposal'],official_receipt=best['online_record'],target=.85,
                compatible_denominator=n,champion_correct=start,required_correct=goal,required_net=goal-start,
                predictions_changed=total,changes_to_true=up,changes_to_false=down,minimum_fixes=needed,blind_only_local_scenario=blind_target,
                maximum_harms=total-needed,required_edit_precision=needed/total,local=evidence,
                interpretation='Compatible denominator inferred from published aggregate scores. Local rows and evaluation scopes overlap. Booking resampling covers only local within-sample variation; different test error rates can change online precision.')
    (RESULT.parent/'tabnet_release_target_gap.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
