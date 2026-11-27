"""Calibrate correction policies for the test manifest's observed distribution of model scores."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.special import logit
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from dissent_likelihood_transfer_n27 import splits
from full_roster_booking_validation_n23 import ROOT
from repeated_archive_misses_n26 import ART, cohort, count
from unanimous_error_reach_n27 import inputs

REPORT = ROOT/'reports/nov23_matched_learning_curve/manifest_weighted_dissent.json'
OUT = ART/'nov27_manifest_weighted_dissent'


def threshold(score, base, y, t, weights):
    cuts = np.full(2,np.inf)
    for side in (0,1):
        ix=t[(base[t]==side)&(score[t]>.5)]
        if not len(ix):
            continue
        ix=ix[np.argsort(-score[ix],kind='stable')]
        w=weights[ix]
        gain=np.cumsum(np.where(base[ix]!=y[ix],w,-w))-np.sqrt(np.cumsum(w*w))
        gain[~np.r_[score[ix[:-1]]!=score[ix[1:]],True]]=-np.inf
        if gain.max()>0:
            cuts[side]=score[ix[gain.argmax()]]
    return cuts


def run():
    raw,frame,y,g,f,first,random,ordered,wide,raw_models=inputs()
    _,blind,known,fresh=cohort(raw,y,f)
    n=len(y)
    groups=frame.PassengerId.str.split('_').str[0].to_numpy(str)
    size=pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
    wide=wide.astype(np.float32).astype(float)  # Match native float32 across decimal CSV and NumPy storage.
    x=np.column_stack([wide, np.std(wide,axis=1), np.mean(wide>=.5,axis=1),first,size,frame.CryoSleep.eq(True)])
    domain=np.arange(len(frame))>=n
    w=np.zeros(len(frame))
    split=partition(groups,'score_density_estimation',4)
    for j in range(4):
        t=split!=j
        q=~t
        model=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=8,l2_regularization=10,early_stopping=False,random_state=9)
        p=model.fit(x[t],domain[t]).predict_proba(x[q])[:,1]
        w[q]=np.exp(np.clip(logit(p)-logit(domain[t].mean()),-np.log(4.),np.log(4.)))
    rng=np.random.default_rng(218)
    shuffled=w.copy()
    for k in range(8):
        q=np.flatnonzero(f==k)
        shuffled[q]=rng.permutation(shuffled[q])
    p={'three_foundations':wide[:,4],'earlier_median':wide[:,0],'Mitra':wide[:,1],'TabICL':wide[:,2],'Nori':wide[:,3],
       'all_seven':np.median(wide,axis=1), 'boosted_receipts':raw_models['augmented_tree']}
    report={'domain_auc':float(roc_auc_score(domain,w)),'weight_quantiles':np.quantile(w[:n],[0,.1,.5,.9,1]).tolist(),'rules':{}}
    saved={}
    for scheme,protected in [('random',random),('contiguous',ordered)]:
        part=splits(g,f,scheme)
        for source,v in p.items():
            risk=np.where(first,1-v,v)
            results={}
            for name,weights in [('observed',w),('unweighted',np.ones_like(w)),('shuffled',shuffled)]:
                flags=np.zeros(len(frame),bool)
                votes=[]
                for k in range(8):
                    for j in (0,1,2,3,None):
                        t=np.flatnonzero(known&(f==k)&((part!=j) if j is not None else True))
                        q=np.flatnonzero((f==k)&(part==j)) if j is not None else np.arange(n,len(frame))
                        cut=threshold(risk,first,y,t,weights)
                        act=(risk[q]>=cut[first[q].astype(int)]) & ~protected[q]
                        if j is None:votes.append(act)
                        else:flags[q]=act
                flags[n:]=np.mean(votes,axis=0)>.5
                results[name]=flags
            results['unique']=results['observed']&~results['unweighted']&~results['shuffled']
            for name,flag in results.items():
                key=source+'_'+name
                saved[scheme+'_'+key]=flag
                err=first[:n]!=y
                delta=np.where(err,1,-1)*flag[:n]
                report['rules'].setdefault(key,{})[scheme]=dict(all=count(flag[:n],known,err),fresh=count(flag[:n],fresh&known,err),
                     old=count(flag[:n],blind,err),test=int(flag[n:].sum()),by_fold=[int(delta[known&(f==k)].sum()) for k in range(8)])
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),weights=w,y=y,fold=f,**saved)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print('Separation',report['domain_auc'],report['weight_quantiles'])
    for name,r in report['rules'].items():
        if name.endswith(('observed','unique')):
            print(name,[(s['all'],s['fresh'],s['test']) for s in r.values()])


if __name__=='__main__':
    with threadpool_limits(1):
        run()
