"""Use foundation support to distinguish helpful from harmful edits proposed by archived branches."""
import json
import numpy as np
import pandas as pd
from directional_foundation_transfer_n27 import calibrate
from dissent_likelihood_transfer_n27 import splits
from frozen_queue_shared_errors_n26 import paired, RELEASES
from repeated_archive_misses_n26 import ART, cohort, count
from full_roster_booking_validation_n23 import ROOT
from six_way_against_scored_nine_n26 import test_decisions

OUT=ART/'nov27_historical_dissent_screen'
REPORT=ROOT/'reports/nov23_matched_learning_curve/historical_dissent_screen.json'


def run():
    raw,y,g,f,known,old=paired()
    _,blind,known2,fresh=cohort(raw,y,f)
    np.testing.assert_array_equal(known2,known)
    n=len(y)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    with np.load(ART/'nov27_dissent_likelihood_transfer/aligned.npz') as z:
        ids=z['passenger'];first=z['first'];protected=[z['random_median'],z['contiguous_median']]
        np.testing.assert_array_equal(ids[n:],test)
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        scores=z['scores']
    bank=np.vstack([old,np.column_stack([test_decisions(name,test)[0] for name in RELEASES.values()])])
    col={name:bank[:,i] for i,name in enumerate(RELEASES)}
    current=first==col['nine']
    fields=(col['fields']!=col['nine'])&current
    band=(col['band']!=col['nine'])&current
    awake=(col['awake']!=col['nine'])&current
    later=(col['review']!=col['all_three'])&current
    proposals={'fields':fields,'band':band,'awake':awake,'review':later,'any_distinct':fields|band|awake|later,
               'two_sources':(np.column_stack([fields,band,awake,later]).sum(1)>=2)}
    median=np.median(scores[:,-3:],axis=1)
    # Conditional screening within the archived disagreement permits weaker foundation support.
    # Remap [0, 1] into (.5, 1] because the existing function expects an opposition ranking above .5.
    risk=.50001+.49998*np.where(first,1-median,median)
    report={}; saved={}
    for scheme,lock in zip(('random','contiguous'),protected):
        part=splits(g,f,scheme)
        for name,gate in proposals.items():
            a=np.zeros(len(ids),bool)
            testvotes=[]
            for k in range(8):
                for j in (0,1,2,3,None):
                    t=np.flatnonzero((f==k)&known&gate[:n]&((part!=j) if j is not None else True))
                    q=np.flatnonzero((f==k)&(part==j)) if j is not None else np.arange(n,len(ids))
                    cut=calibrate(risk,first,y,t,1.)
                    action=gate[q]&~lock[q]&(risk[q]>=cut[first[q].astype(int)])
                    if j is None:testvotes.append(action)
                    else:a[q]=action
            a[n:]=np.mean(testvotes,axis=0)>.5
            err=first[:n]!=y
            delta=a[:n]*np.where(err,1,-1)
            saved[scheme+'_'+name]=a
            report.setdefault(name,{})[scheme]=dict(all=count(a[:n],known,err),fresh=count(a[:n],fresh&known,err),
                original_unanimous=count(a[:n],blind,err),by_fold=[int(delta[known&(f==k)].sum()) for k in range(8)],
                test=int(a[n:].sum()))
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'aligned.npz',passenger=ids,y=y,fold=f,**saved)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    run()
