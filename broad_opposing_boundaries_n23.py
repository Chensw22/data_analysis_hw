"""Predict from the side of local class boundaries defined by nearby opposite-label passengers."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_local_profile_fit_n23 import coordinates
from broad_opposing_passengers_n23 import evaluate,match,tree


def boundary_features(x,query,pos,neg,rng):
    diff=x[pos]-x[neg]
    center=(x[pos]+x[neg])/2
    gap=np.sqrt((diff*diff).sum(1))
    directions=diff/np.maximum(gap[:,None],.01)
    dist,ids=cKDTree(center).query(query,k=48)
    delta=query[:,None,:]-center[ids]
    margin=np.einsum('nkd,nkd->nk',delta,directions[ids])
    # Treat small gaps as uncertain. Scales depend only on reference records.
    margin/=np.maximum(gap[ids],np.median(gap)*.35+.01)
    radius=np.maximum(dist[:,-1:],.05)
    weight=np.exp(-3*(dist/radius)**2)
    random_sign=rng.choice([-1.,1.],len(pos))
    fields={}
    for mode,signed in [('actual',margin),('random',margin*random_sign[ids])]:
        columns=[dist[:,0],dist[:,-1],gap[ids].mean(1)]
        for n in (8,48):
            w=weight[:,:n]
            w=w/w.sum(1)[:,None]
            m=signed[:,:n]
            columns.extend([(w*np.tanh(m)).sum(1),(w*(m>0)).sum(1),(w*np.abs(np.tanh(m))).sum(1)])
        fields[mode]=np.column_stack(columns)
    return fields


def run_fold(frame,g,y,t,q):
    result={}
    pairs=[]
    for a,b in GroupKFold(2,shuffle=True,random_state=914).split(t,groups=g[t]):
        support,learn=t[a],t[b]
        assert not np.intersect1d(g[support],g[learn]).size
        x,_=coordinates(frame,support)
        pos,neg=match(x[support],g[support],y[support])
        pairs.append(len(pos))
        ids=np.r_[learn,q]
        geom=boundary_features(x[support],x[ids],pos,neg,np.random.default_rng(623))
        pred={'direct':tree(frame.iloc[learn],y[learn].astype(int),frame.iloc[q])[:,1]}
        for k,features in geom.items():
            data=frame.iloc[ids].copy()
            for j,values in enumerate(features.T):
                data[f'boundary_{j}']=values
            pred[k]=tree(data.iloc[:len(learn)],y[learn].astype(int),data.iloc[len(learn):])[:,1]
        # Pure local boundary direction with the original class prevalence.
        prior=np.log((y[support].sum()+1)/((~y[support]).sum()+1))
        pred['boundary_only']=expit(prior+4*geom['actual'][len(learn):,6])
        for k,v in pred.items():
            result[k]=result.get(k,0.)+.5*v
    return result,dict(reference_pairs=pairs)


if __name__=='__main__':
    with threadpool_limits(2):
        evaluate('opposing_boundaries',run_fold)
