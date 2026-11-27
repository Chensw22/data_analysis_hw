"""Check whether deviations from broad historical opinions carry information about foundation-model corrections."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from directional_foundation_transfer_n27 import calibrate
from full_roster_booking_validation_n23 import ROOT,data
from repeated_archive_misses_n26 import ART,cohort,count
from six_way_against_scored_nine_n26 import test_decisions

OUT=ART/'nov27_dissent_likelihood_transfer'
REPORT=ROOT/'reports/nov23_matched_learning_curve/dissent_likelihood_transfer_n27.json'
BASE='proposal_n26_champion_fare_six'
CURRENT='proposal_n27_directional_foundation'


def splits(g,f,name):
    if name=='random':
        return partition(g,'directional_foundation',4)
    out=np.zeros(len(f),int)
    for k in range(8):
        for j,orders in enumerate(np.array_split(np.unique(g[f==k]),4)):
            out[(f==k)&np.isin(g,orders)]=j
    return out


def run():
    raw,y,g,f=data()
    _,blind,known,fresh=cohort(raw,y,f)
    n=len(y)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    ids=np.r_[raw.PassengerId.to_numpy(str),test.to_numpy(str)]
    with np.load(ART/'nov26_nori_fare_party/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids[:n])
        first=np.r_[z['champion111'],test_decisions(BASE,test)[0]].astype(bool)
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        assert list(z['names'][-3:])==['Mitra','TabICL fare','Nori fare']
        p=z['scores']
    risk=np.where(first[:,None],1-p,p)
    median=np.median(risk[:,-3:],axis=1)
    past=np.median(risk[:,:18],axis=1)
    odds=lambda v:logit(v.clip(.005,.995))
    # A score can express additional strength over earlier families while still requiring actual foundation opposition.
    def opposition(z):
        return np.where(median>.5,np.maximum(expit(z),.500001),0.)
    gain=odds(median)-odds(past)
    scores=dict(median=median,absolute_delta=opposition(gain),
                amplified=opposition(odds(median)+gain), historical_agreement=opposition(odds(median)+odds(past)))
    # Treat all algorithms equally in the control and withhold targeted tests until policy fitting is finished.
    support=np.sum(risk[:,-3:]>.5,axis=1)
    candidates={k:np.zeros(len(ids),bool) for k in scores}
    report={}
    all_flags={}
    for protocol in ('random','contiguous'):
        part=splits(g,f,protocol)
        cuts={k:[] for k in scores}
        for k,measure in scores.items():
            vote=candidates[k].copy()
            for fold in range(8):
                all_train=np.flatnonzero((f==fold)&known)
                cuts[k].append(calibrate(measure,first,y,all_train,1.))
                for j in range(4):
                    t=np.flatnonzero((f==fold)&known&(part!=j))
                    q=np.flatnonzero((f==fold)&(part==j))
                    assert not np.isin(g[t],g[q]).any()
                    cut=calibrate(measure,first,y,t,1.)
                    vote[q]=measure[q]>=cut[first[q].astype(int)]
            vote[n:]=np.mean(measure[n:,None]>=np.asarray(cuts[k])[:,first[n:].astype(int)].T,axis=1)>.5
            all_flags[protocol+'_'+k]=vote
        old=all_flags[protocol+'_median']
        np.testing.assert_array_equal(old[n:]^first[n:],test_decisions(CURRENT,test)[0])
        current=first^old
        plans={}
        for name in ('absolute_delta','amplified','historical_agreement'):
            action=all_flags[protocol+'_'+name]
            plans[name+'_extra']=~old & action
            plans[name+'_alternate']=old^action
        plans['delta_consensus_extra']=~old&all_flags[protocol+'_amplified']&all_flags[protocol+'_absolute_delta']
        for k,edits in plans.items():
            all_flags[protocol+'_'+k]=edits
            delta=edits[:n]*np.where(current[:n]!=y,1,-1)
            report.setdefault(k,{})[protocol]=dict(test_edits=int(edits[n:].sum()),by_fold=[int(delta[known&(f==k)].sum()) for k in range(8)],
                scopes={s:count(edits[:n],m,current[:n]!=y) for s,m in [('all',known),('fresh',known&fresh),('common',blind),
                      ('unanimous_foundations',known&(support[:n]==3)),('divided_foundations',known&(support[:n]<3))]})
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'aligned.npz',passenger=ids,y=y,fold=f,first=first,**all_flags)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for k,v in report.items():print(k,[r['scopes']['all'] for r in v.values()],v['random']['test_edits'])


if __name__=='__main__':
    with threadpool_limits(1):
        run()
