"""Estimate extra corrections from arithmetic and odds pooling of dissimilar foundation models."""
import json
import numpy as np
from scipy.special import expit, logit
from booking_mix_confidence_transfer_n26 import partition
from directional_foundation_transfer_n27 import calibrate
from dissent_likelihood_transfer_n27 import splits
from full_roster_booking_validation_n23 import ROOT
from repeated_archive_misses_n26 import ART,cohort,count
from unanimous_error_reach_n27 import inputs

OUT=ART/'nov27_independent_pool_correction'
REPORT=ROOT/'reports/nov23_matched_learning_curve/independent_pool_correction.json'


def run():
    raw,frame,y,g,f,first,random,ordered,wide,_=inputs()
    _,blind,known,fresh=cohort(raw,y,f)
    n=len(y)
    cols=wide[:,1:4].astype(np.float32).astype(float).clip(.003,.997)
    lg=logit(cols)
    opinions={'arithmetic':cols.mean(1),'odds':expit(lg.mean(1)),
              'tempered_odds':expit(np.tanh(lg/2).mean(1)*2),
              'two_non_extreme':expit(np.take_along_axis(lg,np.argsort(abs(lg),axis=1)[:,:2],axis=1).mean(1)),
              'two_confident':expit(np.take_along_axis(lg,np.argsort(abs(lg),axis=1)[:,1:],axis=1).mean(1)),
              'equal_architectures':expit(logit(wide[:,[1,2,3,5,6]].clip(.003,.997)).mean(1)),
              'median_of_five':np.median(wide[:,[1,2,3,5,6]],axis=1)}
    core=np.median(wide[:,1:4],axis=1)
    splits_all=[('random',splits(g,f,'random')),('contiguous',splits(g,f,'contiguous'))]
    splits_all += [(str(i),partition(g,'independent_pool_'+str(i),4)) for i in range(6)]
    summary={}
    actions={}
    test_cuts={}
    for method,p in opinions.items():
        trials=[];parents=[]
        risk=np.where(first,1-p,p)
        core_risk=np.where(first,1-core,core)
        # Each policy is independently trained on the same disjoint bookings as its parent.
        cuts=np.stack([calibrate(risk,first,y,np.flatnonzero(known&(f==k)),1.) for k in range(8)])
        tc=(risk[n:,None]>=cuts[:,first[n:].astype(int)].T).sum(1)>4
        test_cuts[method]=cuts
        for tag,partition_ids in splits_all:
            a=np.zeros(n,bool);par=np.zeros(n,bool)
            for k in range(8):
                for j in range(4):
                    t=np.flatnonzero(known&(f==k)&(partition_ids!=j))
                    q=np.flatnonzero((f==k)&(partition_ids==j))
                    for s,dest in [(risk,a),(core_risk,par)]:
                        cut=calibrate(s,first,y,t,1.)
                        dest[q]=s[q]>=cut[first[q].astype(int)]
            if tag=='random':np.testing.assert_array_equal(par,random[:n])
            if tag=='contiguous':np.testing.assert_array_equal(par,ordered[:n])
            extra=a&~par
            trials.append(a);parents.append(par)
            actions[tag+'_'+method]=np.r_[extra,tc&~random[n:]]
            err=first[:n]!=y
            d=extra*np.where(err,1,-1)
            summary.setdefault(method,{})[tag]=dict(all=count(extra,known,err),fresh=count(extra,known&fresh,err),
                common=count(extra,blind,err),folds=[int(d[known&(f==j)].sum()) for j in range(8)],test=int((tc&~random[n:]).sum()))
        consensus=(np.mean(trials[2:],axis=0)>.5)&~(np.mean(parents[2:],axis=0)>.5)
        summary[method]['repeat']=dict(all=count(consensus,known,err),fresh=count(consensus,known&fresh,err),
             common=count(consensus,blind,err))
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=f,**actions)
    REPORT.write_text(json.dumps(summary,indent=2)+'\n')
    for method,row in summary.items():print(method,[row[k]['all'] for k in ('random','contiguous','repeat')],row['random']['test'])


if __name__=='__main__':
    run()
