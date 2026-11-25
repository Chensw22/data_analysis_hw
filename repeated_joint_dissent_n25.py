"""Track the same source-level dissent and the same people under two retraining protocols."""
import json
import numpy as np

from disjoint_tree_fields_n25 import ART,DEST
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from screen_joint_disagreements_n25 import collect,OUT


def counts(active,truth,reference,cohort):
    return dict(corrected=int((active&cohort&(truth!=reference)).sum()),harmed=int((active&cohort&(truth==reference)).sum()))


def run():
    raw,y,g,f=data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,f,rows,hf)
    two=[np.column_stack(panel)>=.5 for panel in panels]
    historical,full=two[1],two[0][rows]
    historical_ref=historical[:,0]
    full_ref=full[:,0]
    cohort=(historical==historical_ref[:,None]).all(1)&(full==full_ref[:,None]).all(1)&(historical_ref==full_ref)
    old=collect(rows,hf,y,g,'historical')
    new=collect(np.arange(len(y)),f,y,g,'fold')[rows]
    with np.load(OUT/'fold0.npz') as z:
        names=['outcome_only','direct','chain_0','chain_1',*[n for n in z.files if n.startswith(('omit_','restore_'))]]
    old_conf=np.where(historical_ref[:,None],1-old,old)
    new_conf=np.where(full_ref[:,None],1-new,new)
    truth=y[rows]
    results={}
    for threshold in (.5,.65,.8,.9):
        old_dissent=old_conf>threshold
        new_dissent=new_conf>threshold
        both=old_dissent&new_dissent
        best=np.minimum(old_conf,new_conf)
        by_model={name:counts(both[:,j],truth,full_ref,cohort) for j,name in enumerate(names)}
        corrects=cohort&(truth!=full_ref)&both.any(1)
        harms=cohort&(truth==full_ref)&both.any(1)
        results[str(threshold)]=dict(historical=counts(old_dissent.any(1),truth,full_ref,cohort),
             grouped=counts(new_dissent.any(1),truth,full_ref,cohort),
             different_sources_allowed=counts(old_dissent.any(1)&new_dissent.any(1),truth,full_ref,cohort),
             same_source=counts(both.any(1),truth,full_ref,cohort),by_model=by_model,
             multiple_sources=counts(both.sum(1)>=2,truth,full_ref,cohort),
             sign_flip_to_transported=counts(both.any(1),truth,full_ref,cohort&~full_ref),
             sign_flip_to_not_transported=counts(both.any(1),truth,full_ref,cohort&full_ref))
        for mask,key in [(corrects,'corrected_profile'),(harms,'harmed_profile')]:
            group_sizes=raw.groupby(g).PassengerId.transform('size').to_numpy()[rows]
            results[str(threshold)][key]=dict(confirmed_awake=int((mask&raw.CryoSleep.eq(False).to_numpy()[rows]).sum()),
                solo=int((mask&(group_sizes==1)).sum()),max_opposition=float(best[mask].max()) if mask.any() else None)
    result=dict(paired_rows=len(rows),same_unanimous=int(cohort.sum()),same_unanimous_errors=int((cohort&(truth!=full_ref)).sum()),
                same_unanimous_correct=int((cohort&(truth==full_ref)).sum()),thresholds=results)
    DEST.with_name('repeated_joint_dissent.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['same_unanimous'],result['same_unanimous_errors'])
    for k,r in results.items():print(k,r['same_source'],r['different_sources_allowed'],r['multiple_sources'],r['sign_flip_to_transported'])


if __name__=='__main__':
    run()
