"""Fit contiguous cabin activity segments from independent bookings' unlabeled service and sleep records."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT
from outside_panel_coverage_n26 import projection
from arbitration_against_scored_nine_n26 import champions
from price_implied_deck_n26 import FEES
from solo_selection_likelihood_n26 import fit_model
from solo_archive_recovery_n26 import ticker

OUT=ROOT/'artifacts/nov26_unlabeled_cabin_activity_segments'


def intervals(matrix):
    n,d=matrix.shape
    cumulative=np.vstack([np.zeros(d),matrix.cumsum(0)])
    # Minimize within-segment squared error plus a fixed per-change penalty.
    cost=np.full(n+1,np.inf)
    cost[0]=0
    parent=np.full(n+1,-1,int)
    for end in range(1,n+1):
        start=np.arange(max(1,end-15))
        if end<16 and end<n:
            continue
        segment_sum=cumulative[end]-cumulative[start]
        reduction=np.einsum('ij,ij->i',segment_sum,segment_sum)/(end-start)
        candidate=cost[start]-reduction+(start>0)*(2.5*np.log1p(n))
        parent[end]=int(start[candidate.argmin()])
        cost[end]=candidate.min()
    ends=[n]
    while ends[-1]>0:
        ends.append(parent[ends[-1]])
        assert ends[-1]>=0
    return sorted(ends)


def features(people,group):
    fields=people.Cabin.str.split('/',expand=True)
    lane=(fields[0]+fields[2]).fillna('?').to_numpy()
    pos=pd.to_numeric(fields[1]).to_numpy()
    fees=people[FEES]
    obs=pd.DataFrame(np.log1p(fees),columns=FEES)
    obs['asleep']=people.CryoSleep.astype(float)
    obs['no_charge']=fees.gt(0).sum(1).eq(0).where(fees.notna().all(1)).astype(float)
    obs['several']=fees.gt(0).sum(1).ge(2).where(fees.notna().all(1)).astype(float)
    columns=list(obs)
    obs=obs.assign(lane=lane,pos=pos,booking=group)
    hold=np.full((len(people),2,len(columns)),np.nan)
    nseg=[]
    for k,(t,q) in enumerate(GroupKFold(3,shuffle=True,random_state=761).split(group,groups=group)):
        assert not np.isin(group[t],group[q]).any()
        # Each booking contributes one profile per lane, even if several members used nearby berths.
        source=obs.iloc[t].dropna(subset=['pos']).groupby(['lane','booking'])[columns+['pos']].mean().reset_index()
        for track,s in source.groupby('lane'):
            bank=s.groupby('pos')[columns].mean()
            if len(bank)<2:continue
            idx=q[(lane[q]==track)&np.isfinite(pos[q])]
            if not len(idx):continue
            avg=bank.mean().fillna(0)
            scale=bank.std().fillna(1).clip(.1)
            x=bank.fillna(avg).to_numpy()
            edges=intervals((x-avg.to_numpy())/scale.to_numpy())
            means=np.stack([(x[a:b].sum(0)+6*avg.to_numpy())/(b-a+6) for a,b in zip(edges[:-1],edges[1:])])
            positions=bank.index.to_numpy()
            cut=np.array([.5*(positions[c-1]+positions[c]) for c in edges[1:-1]])
            rank=np.searchsorted(positions,pos[idx]).clip(max=len(positions)-1)
            rotated=positions[(rank+max(1,len(positions)//3))%len(positions)]
            for j,z in enumerate([pos[idx],rotated]):
                hold[idx,j]=means[np.searchsorted(cut,z)]
            nseg.append(dict(fold=k,track=track,positions=len(bank),segments=len(means)))
    return hold,obs[columns].to_numpy(),nseg


def run():
    raw,y,g,fold,views,_=projection(include_full=True)
    nine,known,_,_=champions()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    people=pd.concat([raw,test],ignore_index=True)
    group=people.PassengerId.str[:4].to_numpy()
    segment,obs,details=features(people,group)
    x=compact_features(raw)
    size=raw.PassengerId.groupby(g).transform('size').to_numpy()
    x['party_size']=size
    pred=np.zeros((len(y),3))
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.isin(g[t],g[q]).any()
        for j in range(3):
            view=x
            if j:
                features_frame=pd.DataFrame(segment[:len(raw),j-1],columns=[f'local_{i}' for i in range(segment.shape[2])])
                view=pd.concat([x,features_frame],axis=1)
            pred[q,j]=fit_model(view.iloc[t],y[t]).predict_proba(view.iloc[q])[:,1]
        print(k,flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'scores.npz',fold=fold,ids=raw.PassengerId.to_numpy(str),y=y,p=pred)
    report={'segments':details,'observed_field_rmse':np.sqrt(np.nanmean((segment-obs[:,None,:])**2,axis=0)).tolist(),
            'field_order':[*FEES,'asleep','no_charge','several']}
    for tag in ('full','bookingmix','historical_to_new'):
        mask,parent=views[tag]
        assert known[mask].all()
        agree=parent==nine
        wrong=parent!=y
        flip=(pred>=.5)!=parent[:,None]
        moves={**dict(zip(['own','actual','rotated'],flip.T,strict=True)),
               'actual_extra':flip[:,1]&~flip[:,0],'rotated_extra':flip[:,2]&~flip[:,0],
               'actual_only':flip[:,1]&~flip[:,0]&~flip[:,2]}
        masks={'all':mask,'solo':mask&(size==1),'agreement':mask&agree,'solo_agreement':mask&agree&(size==1)}
        report[tag]={n:{j:ticker(m,wrong,act) for j,m in masks.items()} for n,act in moves.items()}
        print(tag,{n:(d['solo_agreement']['fixed'],d['solo_agreement']['harmed']) for n,d in report[tag].items()},flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/unlabeled_cabin_activity_segments.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
