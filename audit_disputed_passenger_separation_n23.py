"""Measure how well held-out bookings distinguish 135 recurring rescues from 733 collateral reversals."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from audit_block_alignment_placebo_n23 import alignment
from audit_stable_rescue_placebo_n23 import panel
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import coordinates
from broad_partition_evidence_n23 import ROOT,write

REPORT=ROOT/'reports/nov23_disputed_passenger_separation'


def main():
    row,y,frame,groups,_=load()
    eligible,direction,p,ref,stable,freq,size=panel(row,y,frame)
    align=alignment(row,y,p,ref,direction)['authentic']
    old=np.column_stack(list(stable.values()))
    ids=np.flatnonzero(eligible & (old.any(1)|(align>0)))
    target=direction!=y
    assert (len(ids),int(target[ids].sum()))==(868,135)
    risk=np.where(direction,1-p.mean(0),p.mean(0))
    votes=np.column_stack([risk,direction,np.std(p,axis=0),freq/size,align,old])
    # Exact conflicts count only identical available raw profiles, including direction.
    profile=frame.iloc[ids].copy()
    profile['reference_direction']=direction[ids]
    codes=profile.groupby(list(profile.columns),dropna=False,sort=False).ngroup().to_numpy()
    yes=np.bincount(codes,weights=target[ids]).astype(int)
    no=np.bincount(codes)-yes
    mixed=(yes>0)&(no>0)
    duplicates=dict(conflicting_profiles=int(mixed.sum()),rescues_in_conflict=int(yes[mixed].sum()),
                    correct_in_conflict=int(no[mixed].sum()),unavoidable_raw_errors=int(np.minimum(yes,no).sum()))
    results={}
    for seed in (192,563,784):
        scores={k:np.zeros(len(row)) for k in ('confidence','raw','votes','raw_plus_votes')}
        masks={k:{scope:np.zeros(len(row),bool) for scope in ('positive_net','top_tenth')} for k in scores}
        for a,b in GroupKFold(5,shuffle=True,random_state=seed).split(ids,groups=groups[ids]):
            t,q=ids[a],ids[b]
            assert not np.intersect1d(groups[t],groups[q]).size
            raw,_=coordinates(frame,t)
            raw=np.column_stack([raw,direction])
            for name,x in [('confidence',votes[:,:2]),('raw',raw),('votes',votes),('raw_plus_votes',np.c_[raw,votes])]:
                if name in ('confidence','votes'):
                    scaler=StandardScaler().fit(x[t])
                    model=LogisticRegression(C=.12,max_iter=300).fit(scaler.transform(x[t]),target[t])
                    score=model.predict_proba(scaler.transform(x[q]))[:,1]
                else:
                    model=HistGradientBoostingClassifier(max_leaf_nodes=5,max_iter=55,min_samples_leaf=25,
                                                         l2_regularization=7.,early_stopping=False,random_state=192)
                    model.fit(x[t],target[t])
                    score=model.predict_proba(x[q])[:,1]
                scores[name][q]=score
                masks[name]['positive_net'][q]=score>=.5
                ranked=np.lexsort((row[q],-score))
                masks[name]['top_tenth'][q[ranked[:round(len(q)/10)]]]=True
        metrics={}
        for name,score in scores.items():
            metrics[name]=dict(auc=float(roc_auc_score(target[ids],score[ids])),
                               average_precision=float(average_precision_score(target[ids],score[ids])))
            for key,mask in masks[name].items():
                metrics[name][key]=dict(fixed=int((mask&target).sum()),harmed=int((mask&~target).sum()))
        print(seed,metrics,flush=True)
        results[str(seed)]=metrics
    write(REPORT/'audit.json',dict(passengers=len(ids),errors=int(target[ids].sum()),bookings=int(len(np.unique(groups[ids]))),
                                   exact_raw_conflicts=duplicates,plans=results))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
