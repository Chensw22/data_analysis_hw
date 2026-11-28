"""Measure local fluctuations of other bookings' visible attributes at multiple spatial scales."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from archive_source_harm_veto_n28 import inputs
from full_roster_booking_validation_n23 import ROOT
from manifest_activity_transitions_n27 import design
from repeated_archive_misses_n26 import ART,cohort,count
from price_implied_deck_n26 import FEES
from relational_error_corrector import write

OUT=ART/'nov28_cross_order_texture'


def context(frame):
    g=frame.PassengerId.str[:4].astype(int).to_numpy()
    cab=frame.Cabin.str.split('/',expand=True)
    region=(cab[0]+'-'+cab[2]).fillna('unknown').to_numpy()
    bills=frame[FEES]
    markers=np.column_stack([frame.HomePlanet.eq('Earth'),frame.HomePlanet.eq('Europa'),
               frame.CryoSleep.eq(True),frame.Age.lt(13),np.log1p(bills.fillna(0)),bills.isna().sum(axis=1)]).astype(float)
    def blocks(pools):
        result=np.zeros((len(frame),markers.shape[1]*12))
        for subset in pools:
            group,inv=np.unique(g[subset],return_inverse=True)
            size=np.bincount(inv)
            profile=np.zeros((len(group),markers.shape[1]))
            np.add.at(profile,inv,markers[subset])
            profile/=size[:,None]
            # At each scale take two successive disjoint windows on each side.
            # Their differences describe local shifts; absolute differences describe local variability.
            cumulative=np.vstack([np.zeros(profile.shape[1]),profile.cumsum(0)])
            aligned=[]
            for width in (3,9,27):
                mid=np.arange(len(group))
                def mean(lo,hi):
                    start=np.clip(lo,0,len(group));stop=np.clip(hi,0,len(group))
                    return (cumulative[stop]-cumulative[start])/np.maximum(stop-start,1)[:,None]
                a=mean(mid-2*width,mid-width);b=mean(mid-width,mid)
                c=mean(mid+1,mid+1+width);d=mean(mid+1+width,mid+1+2*width)
                aligned.extend([(b+c)/2,c-b,(abs(a-b)+abs(c-d))/2,(b+c-a-d)/2])
            result[subset]=np.column_stack(aligned)[inv]
        return result
    return blocks([np.arange(len(g))]),blocks([np.flatnonzero(region==r) for r in np.unique(region)])


def run():
    frame,y,g,f,ids,_,_,seven,versions=inputs();n=len(y)
    _,_,known,fresh=cohort(frame.iloc[:n],y,f)
    first=versions['random'][:,0]
    wrong=first[:n]!=y
    a,b=design(frame)
    level=np.column_stack([a,b])
    glob,corridor=context(frame)
    # Compare local means to adding signed and unsigned neighborhood changes.
    ix=np.tile(np.r_[np.ones(glob.shape[1]//12,bool),np.zeros(3*glob.shape[1]//12,bool)],3)
    means=np.column_stack([level,glob[:,ix],corridor[:,ix]])
    gs=frame.PassengerId.str[:4].to_numpy()
    ords=np.empty(n,int)
    for k,part in enumerate(np.array_split(np.unique(g),8)):ords[np.isin(g,part)]=k
    signed=np.where(first,1.,-1.)*abs(logit(np.median(seven[:,1:4],axis=1).clip(.003,.997)))
    views=[level,means,np.c_[means,glob[:,~ix]],np.c_[means,corridor[:,~ix]],np.c_[means,glob[:,~ix],corridor[:,~ix]]]
    def fit(x,t,q):
        clf=HistGradientBoostingClassifier(max_iter=145,max_leaf_nodes=15,l2_regularization=6,
               learning_rate=.07,min_samples_leaf=32,early_stopping=False,random_state=28127)
        return clf.fit(x[t],y[t]).decision_function(x[q])
    saved={};report={}
    OUT.mkdir(exist_ok=True)
    for mode,part in [('random',f),('ordered',ords)]:
        z=np.empty((len(frame),len(views)))
        for k in range(9):
            t=np.flatnonzero(part!=k)
            q=np.flatnonzero(part==k) if k<8 else np.arange(n,len(frame))
            assert not np.intersect1d(gs[t],gs[q]).size
            for j,x in enumerate(views):z[q,j]=fit(x,t,q)
        oppose=(z>=0)!=first[:,None]
        changes=((signed[:,None]+z[:,2:]-z[:,[1]]>=0)!=first[:,None])&oppose[:,2:]
        primary=changes.all(1)
        four=(versions[mode]==first[:,None]).all(1)
        seven_agree=((seven>=.5)==first[:,None]).all(1)
        scopes={'all':known,'fresh':known&fresh,'four':known&four[:n],'four_seven':known&four[:n]&seven_agree[:n]}
        rules={'combined':changes[:,2],'direction_confirmed':primary,
               'native_combined':oppose[:,-1],'mean_only':((signed+z[:,1]-z[:,0]>=0)!=first)&oppose[:,1]}
        runs={}
        for key,flag in rules.items():
            runs[key]=dict(scores={s:count(flag[:n],m,wrong) for s,m in scopes.items()},
                           accuracy=float(np.mean((first[:n]^flag[:n])[known]==y[known])),test=int(flag[n:].sum()),
                           test_four=int((flag[n:]&four[n:]).sum()))
        saved[mode+'_flag']=primary;saved[mode+'_z']=z
        report[mode]=dict(native=((z[:n]>=0)==y[:,None])[known].mean(0).tolist(),rules=runs)
        print(mode,runs['direction_confirmed'],flush=True)
    np.testing.assert_allclose(saved['ordered_z'][n:],saved['random_z'][n:])
    common=saved['ordered_flag']&saved['random_flag']
    np.savez_compressed(OUT/'aligned.npz',ids=ids,y=y,fold=f,first=first,keep=common,**saved)
    report['replicated']=dict(all=count(common[:n],known,wrong),fresh=count(common[:n],known&fresh,wrong),
            test=int(common[n:].sum()))
    write(OUT/'result.json',report)
    write(ROOT/'reports/nov23_matched_learning_curve/cross_order_texture.json',report)


if __name__=='__main__':
    with threadpool_limits(2):run()
