"""Count newly covered recurring errors and collateral errors across the latest model families."""
import numpy as np
from audit_service_distance_rescue_n23 import PRIOR_MODELS,cached
from broad_bill_completion_states_n23 import LABEL,REPORT
from broad_bill_neighbor_decision_n23 import opinions
from broad_geographic_transfer_n23 import BILLS,load
from broad_partition_evidence_n23 import SEEDS,write
from broad_tree_field_dividends_n23 import hard_cohorts

PAST={**PRIOR_MODELS,'service_distribution_distance':['budget','unordered_fees','sliced_identity','aligned_services']}
LATEST={LABEL:['masked','conditioned','missing_marginal','missing_zero','missing_paid'],
        'conditional_bill_completion':['factual','pooled_without_bills','stationary_without_bills','shuffled_without_bills',
                                       'factual_with_shift','shuffled_shift','pooled_completion','stationary_completion',
                                       'shuffled_completion'],
        'partner_specific_fields':['shared','partner_specific']}


def main():
    row,y,frame,_,_=load()
    _,_,hard,right=hard_cohorts(row,y)
    missing=frame[BILLS].isna().any(axis=1).to_numpy()
    history={}
    for seed in map(str,SEEDS):
        fold,_,ref,bill,near=opinions(seed,row,y,frame.BookingSize.to_numpy()>1)
        check=dict(row=row,y=y,fold=fold,ref=ref)
        old=[bill,near,*[cached(seed,k,v,check) for k,v in PAST.items()]]
        history.setdefault('previous',[]).append((np.column_stack(old)>=.5)==y[:,None])
        for k,columns in LATEST.items():
            p=cached(seed,k,columns,check)
            history.setdefault(k,[]).append((p>=.5)==y[:,None])
    previous=np.stack(history.pop('previous'))
    seen=previous.all(0).any(1)
    earlier_harms=(~previous.any(0)).any(1)
    assert (seen&hard).sum()==78
    fixes,damages={},{}
    for name,h in history.items():
        fixes[name]=np.stack(h).all(0).any(1)
        damages[name]=(~np.stack(h).any(0)).any(1)
    fixes['union']=np.stack(list(fixes.values())).any(0)
    damages['union']=np.stack(list(damages.values())).any(0)
    metrics={}
    for k,h in fixes.items():
        novel=h&hard&~seen
        metrics[k]=dict(stable_hard=int((h&hard).sum()),new_hard=int(novel.sum()),
                        new_hard_complete_bills=int((novel&~missing).sum()),
                        stable_harms=int((damages[k]&right).sum()),new_stable_harms=int((damages[k]&right&~earlier_harms).sum()))
    cases=[]
    for i in np.flatnonzero(fixes['union']&hard&~seen):
        experts=[key+'/'+name for key,h in history.items() for name,success in zip(LATEST[key],np.stack(h).all(0)[i]) if success]
        cases.append(dict(row=int(row[i]),transported=bool(y[i]),planet=str(frame.HomePlanet.iloc[i]),
                          age=None if np.isnan(frame.Age.iloc[i]) else float(frame.Age.iloc[i]),
                          bills=[None if np.isnan(v) else float(v) for v in frame.iloc[i][BILLS]],
                          experts=experts))
    write(REPORT/'novel_errors.json',dict(previous=78,metrics=metrics,bill_order=BILLS,cases=cases))
    print(metrics)


if __name__=='__main__':
    main()
