"""Decide a fixed number of changes per roster using only out-of-booking error rankings."""
import json
import numpy as np

from continuous_archive_margins_n26 import OUT as MODEL
from full_roster_booking_validation_n23 import ROOT, data
from repeated_archive_misses_n26 import ART, cohort, count

REPORT = ROOT/'reports/nov23_matched_learning_curve/selective_archive_margins.json'


def edits(score, rows, budget):
    ranked = rows[np.argsort(-score[rows],kind='stable')]
    keep = min(len(rows),round(len(rows)*budget))
    if not keep:
        return ranked[:0]
    # Ties at the quota boundary are all excluded, so passenger index never settles a tied score.
    cut = score[ranked[keep]] if keep < len(rows) else -np.inf
    return ranked[(score[ranked]>.5)&(score[ranked]>cut)]


def run():
    raw,y,g,f=data()
    base,blind,known,fresh=cohort(raw,y,f)
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        anchors=z['anchor']
        native_score=z['scores'][:,-3:].mean(1)
        passenger=z['passenger']
    n=len(y)
    report={}
    test_flags={}
    with np.load(MODEL/'scores.npz') as z:
        for key,v in [('passenger',passenger),('fold',f),('y',y)]:
            np.testing.assert_array_equal(z[key],v)
        opinions={k:z[k] for k in z if k not in ('passenger','fold','y')}
    opinions['foundation_mean']=native_score
    for name,p in opinions.items():
        risk=np.where(anchors[:len(p)],1-p,p)
        out={}
        for frac in [.01,.025,.05,.10]:
            flag=np.zeros(len(p),bool)
            for k in np.unique(f):
                flag[edits(risk,np.flatnonzero(f==k),frac)]=True
            test=np.arange(n,len(p))
            if len(test):
                flag[edits(risk,test,frac)]=True
                test_flags[name+'_'+str(frac)]=flag[n:]
            delta=flag[:n]*np.where(base!=y,1,-1)
            out[str(frac)]=dict(test_changes=int(flag[n:].sum()) if len(test) else None,
                by_fold=[int(delta[known&(f==k)].sum()) for k in np.unique(f)],
                scopes={a:count(flag[:n],m,base!=y) for a,m in [('known',known),('blind',blind),('fresh',known&fresh)]})
        report[name]=out
    np.savez_compressed(MODEL/'selective_test.npz',passenger=passenger[n:],reference=anchors[n:],**test_flags)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for k,v in report.items():
        print(k, [(round(float(a)*100,2),r['scopes']['known']['fixed']-r['scopes']['known']['harmed']) for a,r in v.items()])


if __name__=='__main__':
    run()
