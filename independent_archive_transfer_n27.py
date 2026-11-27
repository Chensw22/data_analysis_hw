"""Transfer independent frozen policies to untouched people in the two clean fare releases."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from full_roster_booking_validation_n23 import ROOT,data
from repeated_archive_misses_n26 import ART,cohort,count
from frozen_queue_shared_errors_n26 import RELEASES
from joint_candidate_coverage_n26 import branch_decisions
from family_identity_transfer_n26 import panels_and_reference
from six_way_against_scored_nine_n26 import test_decisions
from audit_clean_fare_anchor_bounds_n26 import read_sources
from export_joint_fare_foundations_n26 import NAME as NEW,BASE
from price_implied_deck_n26 import FEES

OUT=ART/'nov27_independent_archive_transfer'
REPORT=ROOT/'reports/nov23_matched_learning_curve/independent_archive_transfer_n27.json'
SOURCES=('fields','awake','band')


def groups(raw):
    paid=raw[FEES].gt(0).any(axis=1).to_numpy()
    multi=raw.PassengerId.str.split('_').str[0].duplicated(keep=False).to_numpy()
    asleep=raw.CryoSleep.eq(True).to_numpy()
    return dict(unpaid=~paid,sleep_unpaid=asleep&~paid,multi=multi,multi_unpaid=multi&~paid)


def policies(nine,pred,domains):
    flips=np.column_stack([pred[k]!=nine for k in SOURCES])
    modes={k:v for k,v in zip(SOURCES,flips.T)}
    modes['at_least_two']=flips.sum(1)>=2
    modes['fields_or_band']=flips[:,0]|flips[:,2]
    return {f'{s}_{k}':mask&v for s,mask in domains.items() for k,v in modes.items()}


def run():
    raw,y,g,f=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)
    a161,blind,known,fresh=cohort(raw,y,f)
    certain,versions=read_sources(raw,y,g,f)
    np.testing.assert_array_equal(known,certain)
    np.testing.assert_array_equal(a161[known],versions['161'][0,known])
    _,_,_,hi,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:prior=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hist,old,v131=z['fold'],z['nine'],z['v131']
        np.testing.assert_array_equal(z['rows'],hi)
    awake=raw.CryoSleep.eq(False).to_numpy()
    full=branch_decisions(np.arange(len(y)),f,p,names,native,prior,ref[1],ref[2],awake,'full',891)
    past=branch_decisions(hi,hist,h,names,old_native,v131,old,old,awake,'historical',891)
    test_pred={k:test_decisions(RELEASES[k],test.PassengerId)[0] for k in ('nine',*SOURCES)}
    test_parent={k:test_decisions(tag,test.PassengerId)[0] for k,tag in [('111',BASE),('161',NEW)]}
    domains=groups(raw)
    cur=[policies(full['nine'][b],{k:full[k][b] for k in SOURCES},domains) for b in range(2)]
    pre=policies(old,{k:past[k][0] for k in SOURCES},{k:v[hi] for k,v in domains.items()})
    target=policies(test_pred['nine'],test_pred,groups(test))
    report={}
    preds={}
    for tag,actions in target.items():
        result={}
        for parent in ('111','161'):
            base=versions[parent]
            # The historical proposals are defined as edits to nine. Retain every winner-parent change.
            edits=np.stack([cur[b][tag]&(base[b]==versions['nine'][b]) for b in range(2)])
            delta=edits*np.where(base!=y,1,-1)
            np.testing.assert_array_equal(delta[0,known],delta[1,known])
            edit=edits[0]
            pred=test_parent[parent]^(actions&(test_pred['nine']==test_parent[parent]))
            saved=pred!=test_parent[parent]
            preds[parent+'_'+tag]=pred
            historical_gate=pre[tag]
            repeat=edit[hi]&historical_gate&(old==base[0,hi])
            result[parent]=dict(test_changes=int(saved.sum()),test_changed_orders=int(test.PassengerId[saved].str[:4].nunique()),
                all_bounds=[int(delta.min(0).sum()),int(delta.max(0).sum())],
                all=count(edit,known,y!=base[0]),fresh=count(edit,known&fresh,y!=base[0]),
                blind=count(edit,blind,y!=base[0]),repeat=count(repeat,known[hi],y[hi]!=base[0,hi]),
                historical_nine=count(historical_gate,np.ones(len(hi),bool),y[hi]!=old),
                folds=[int(delta[0,known&(f==i)].sum()) for i in range(8)])
        report[tag]=result
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'test.npz',passenger=test.PassengerId.to_numpy(str),**preds)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for k,v in report.items():
        row=v['161']
        print(k,row['all_bounds'],row['fresh'],row['repeat'],row['historical_nine'],row['test_changes'])


if __name__=='__main__':
    with threadpool_limits(1):
        run()
