"""Test integer budget splitting against a count-independent density and opposite rounding direction."""
import json

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold

from full_roster_booking_validation_n23 import ROOT,data

OUT=ROOT/'artifacts/nov26_latent_bill_rounding'
DEST=ROOT/'reports/nov23_matched_learning_curve/latent_bill_rounding.json'
BILLS=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
OFF=np.arange(5)
# If k nonzero service shares sum to an integer, the integer loss after flooring
# has the Eulerian distribution under uniform fractional parts conditional on their sum.
KERNEL=np.array([[1,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,.5,.5,0,0],
                 [0,1/6,4/6,1/6,0],[0,1/24,11/24,11/24,1/24]],float)
# Include a small full-support component for unobserved sub-unit charges.
REAL=.85*KERNEL+.15*np.array([.02,.15,.45,.30,.08])


def learn(total,count,train,kernel,length):
    amounts=total[train]
    emissions=kernel[count[train]]
    bins=amounts[:,None]+OFF
    spread=np.bincount(amounts,minlength=length).astype(float)
    background=gaussian_filter1d(spread,16,mode='nearest')+.01
    background/=background.sum()
    pseudo=max(30.,.25*len(train))*background
    mass=background.copy()
    for _ in range(45):
        joint=mass[bins]*emissions
        resp=joint/joint.sum(1,keepdims=True)
        mass=np.bincount(bins.ravel(),weights=resp.ravel(),minlength=length)+pseudo
        mass/=mass.sum()
    return mass


def score(mass,kernel,amount,count):
    idx=amount[:,None]+OFF
    joint=mass[idx]*kernel[count]
    density=joint.sum(1)
    post=joint/density[:,None]
    latent=amount+post.argmax(1)
    values=np.arange(len(mass)-4)
    conditional=mass[values[:,None]+OFF]@kernel.T
    # Score only the exact integer inside its observed 64-unit band. This removes
    # the very different broad distributions of amounts across service counts.
    bins=np.arange(len(values))//64
    bucket=np.stack([np.bincount(bins,weights=v) for v in conditional.T],axis=1)
    local_log=np.log(density/bucket[amount//64,count])
    smooth=gaussian_filter1d(mass,12,mode='nearest').clip(1e-12)
    standout=np.log(mass[latent]/smooth[latent])
    return np.column_stack([post@OFF,post.max(1),local_log,standout,latent])


def run():
    raw,y,g,_=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    allrows=pd.concat([raw,test],ignore_index=True)
    group=allrows.PassengerId.str[:4].to_numpy()
    origin=allrows.HomePlanet.fillna('missing').to_numpy()
    bills=allrows[BILLS].to_numpy(float)
    total=np.nansum(bills,axis=1).astype(int)
    count=(bills>0).sum(1)
    eligible=np.isfinite(bills).all(1)&(total>0)
    kernels={'floor':REAL,'reverse':REAL[:,::-1],'neutral':np.tile(REAL[3],(6,1))}
    outputs={k:np.zeros((len(total),5)) for k in kernels}
    length=int(total.max())+5
    folds=np.empty(len(total),int)
    for k,(train,q) in enumerate(GroupKFold(5,shuffle=True,random_state=8897).split(total,groups=group)):
        assert not np.isin(group[train],group[q]).any()
        folds[q]=k
        for planet in np.unique(origin):
            support=train[eligible[train]&(origin[train]==planet)]
            held=q[eligible[q]&(origin[q]==planet)]
            for mode,kernel in kernels.items():
                fitted=learn(total,count,support,kernel,length)
                outputs[mode][held]=score(fitted,kernel,total[held],count[held])
        print('billing round',k,flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'features.npz',passenger=allrows.PassengerId.to_numpy(str),eligible=eligible,
                        total=total,count=count,fold=folds,**outputs)
    report={}
    for planet,ix in [('all',eligible),*[(v,eligible&(origin==v)) for v in np.unique(origin)]]:
        d=outputs['floor'][:,2]-outputs['neutral'][:,2]
        rev=outputs['floor'][:,2]-outputs['reverse'][:,2]
        report[planet]=dict(rows=int(ix.sum()),floor_advantage_nats=float(d[ix].mean()),
                            over_reverse_nats=float(rev[ix].mean()),folds=[float(d[ix&(folds==k)].mean()) for k in range(5)],
                            mean_nats={n:float(arr[ix,2].mean()) for n,arr in outputs.items()})
    DEST.write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__':
    run()
