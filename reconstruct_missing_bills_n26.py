"""Infer a missing integer charge by combining matched donors with the observed price spectrum."""
import json

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold

from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT,data
from latent_bill_rounding_n26 import BILLS,OFF,OUT,REAL,learn

QUANTILES=np.array([.1,.3,.5,.7,.9])


def grid(maximum):
    limits=np.unique(np.r_[0,1,2,3,4,5,np.ceil(np.geomspace(6,maximum+1,56)).astype(int)])
    return limits[:-1],limits[1:]-1


def reweight(lo,hi,p,s,num_pos,table,shift,truth):
    rows=np.arange(len(s))
    start=s[:,None]+lo+shift
    stop=s[:,None]+hi+1+shift
    width=hi-lo+1
    c=num_pos[:,None]+(lo>0)
    mass=table[c,stop]-table[c,start]
    prob=p*mass/width
    norm=prob.sum(1)
    prob/=norm[:,None]
    cumulative=np.cumsum(prob,axis=1)
    which=np.searchsorted(lo,truth,side='right').clip(1,len(lo))-1
    exact=s+truth+shift
    pc=num_pos+(truth>0)
    actual_density=(p[rows,which]/width[which])*(table[pc,exact+1]-table[pc,exact])/norm
    samples=[]
    for quantile in QUANTILES:
        j=(cumulative<quantile).sum(1).clip(max=len(lo)-1)
        below=cumulative[rows,j]-prob[rows,j]
        frac=((quantile-below)/prob[rows,j].clip(1e-12)).clip(0,1)
        target=table[c[rows,j],start[rows,j]]+mass[rows,j]*frac
        val=np.empty(len(s),int)
        for cnt in np.unique(c[rows,j]):
            mask=c[rows,j]==cnt
            # Invert the within-bin CDF on the integer support.
            z=np.searchsorted(table[cnt],target[mask],side='right')-1
            val[mask]=np.clip(z-s[mask]-shift,lo[j[mask]],hi[j[mask]])
        samples.append(val)
    return np.column_stack(samples),prob[:,0],np.log(actual_density.clip(1e-300))


def run():
    raw,y,_,_=data()
    roster=pd.concat([raw,pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())],ignore_index=True)
    g=roster.PassengerId.str[:4].to_numpy()
    fields=compact_features(roster)
    x=fields.drop(columns=BILLS).copy()
    x['Age']=x.Age.fillna(-10)/30
    x['CabinNumber']=x.CabinNumber.fillna(-100)/600
    x['party']=roster.PassengerId.str[:4].map(roster.PassengerId.str[:4].value_counts())
    base=pd.get_dummies(x,dtype=float).to_numpy()
    values=roster[BILLS].to_numpy()
    missing=np.isnan(values)
    bills=np.nan_to_num(values).astype(int)
    complete=~missing.any(1)
    total=bills.sum(1)
    count=(bills>0).sum(1)
    planet=fields.HomePlanet.to_numpy()
    lo,hi=grid(int(bills.max()))
    max_total=int(total.max())
    length=max_total+5
    steps=int(total.max()+hi[-1]+32)
    modes=['donor','spectrum','shifted']
    predictions={m:np.zeros((len(roster),len(BILLS),len(QUANTILES)),np.float32) for m in modes}
    zero={m:np.zeros_like(bills,dtype=float) for m in modes}
    likelihood={m:np.zeros_like(bills,dtype=float) for m in modes}
    scores={}
    for fold,(t,q) in enumerate(GroupKFold(5,shuffle=True,random_state=30419).split(g,groups=g)):
        assert not np.isin(g[t],g[q]).any()
        estimates={}
        for pl in np.unique(planet):
            tr=t[(planet[t]==pl)&complete[t]&(total[t]>0)]
            spectrum=learn(total,count,tr,REAL,length)
            numbers=np.arange(max_total+1)
            hist=(spectrum[numbers[:,None]+OFF]@REAL.T).T
            bg=gaussian_filter1d(hist,14,axis=1,mode='nearest')
            ratio=np.ones((6,steps))
            ratio[:,:len(numbers)]=(hist/bg.clip(1e-14)).clip(.1,15)
            ratio[:,0]=1
            estimates[pl]=np.pad(np.cumsum(ratio,axis=1),((0,0),(1,0)))
        uniform=np.broadcast_to(np.arange(steps+1),(6,steps+1))
        for j in range(5):
            others=np.arange(5)!=j
            held=q[~missing[q][:,others].any(1)]
            src=t[complete[t]]
            xx=np.column_stack([base,(bills[:,others]>0)*.6,np.log1p(bills[:,others])/4])
            d,ix=cKDTree(xx[src]).query(xx[held],k=80)
            w=np.exp(-3*(d/d[:,-1,None].clip(.001))**2)
            w/=w.sum(axis=1,keepdims=True)
            yd=bills[src[ix],j]
            buckets=np.searchsorted(lo,yd,side='right')-1
            prob=np.zeros((len(held),len(lo)))
            np.add.at(prob,(np.arange(len(held))[:,None],buckets),w)
            # One unit of broad prior mass for every 40 nearest-neighbour units.
            bg=np.bincount(np.searchsorted(lo,bills[src,j],side='right')-1,minlength=len(lo)).astype(float)+.1
            prob=(prob+.025*bg[None,:]/bg.sum())/1.025
            for pl in np.unique(planet[held]):
                mask=planet[held]==pl
                h=held[mask]
                s=bills[h][:,others].sum(1)
                k=(bills[h][:,others]>0).sum(1)
                for mode,mat,shift in [('donor',uniform,0),('spectrum',estimates[pl],0),('shifted',estimates[pl],17)]:
                    predictions[mode][h,j],zero[mode][h,j],likelihood[mode][h,j]=reweight(
                        lo,hi,prob[mask],s,k,mat,shift,bills[h,j])
        print('reconstruction',fold,flush=True)
    for mode in modes:
        s={}
        for name,mask in [('all',complete),('positive_basket',complete&(total>0))]:
            prediction=predictions[mode][mask]
            actual=bills[mask]
            s[name]=dict(passengers=int(mask.sum()),charges=int(actual.size),
                mean_log_probability=float(likelihood[mode][mask].mean()),
                mean_abs_log1p_error_of_median=float(np.abs(np.log1p(prediction[:,:,2])-np.log1p(actual)).mean()),
                observed_zero_accuracy=float(((zero[mode][mask]>=.5)==(actual==0)).mean()),
                exact_amount_accuracy=float((prediction[:,:,2]==actual).mean()))
        scores[mode]=s
    np.savez_compressed(OUT/'missing_charge_estimates.npz',passenger=roster.PassengerId.to_numpy(str),missing=missing,
                        **predictions,**{name+'_zero':val for name,val in zero.items()},
                        **{name+'_logprob':val for name,val in likelihood.items()})
    path=ROOT/'reports/nov23_matched_learning_curve/reconstructed_charges.json'
    path.write_text(json.dumps(scores,indent=2)+'\n')
    print(scores,flush=True)


if __name__=='__main__':
    run()
