"""Learn shared nonlinear registration and spending coordinates from the unlabeled manifest."""
import json

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem
from threadpoolctl import threadpool_limits

from full_roster_two_view_reliability_n24 import design
from repeated_archive_misses_n26 import ART, cohort, count
from unanimous_error_reach_n27 import ROOT, inputs


def embedding(x):
    filled = np.nan_to_num(x, nan=0.)
    x = np.column_stack([filled,np.isnan(x)])
    x = (x-x.mean(0))/x.std(0).clip(.15)
    z = Nystroem(gamma=1/x.shape[1], n_components=80, random_state=511).fit_transform(x)
    z -= z.mean(0)
    eigen, rotation = eigh(z.T@z/len(z))
    return (z@rotation)/np.sqrt(eigen.clip(0)+.001)


def shared(z, h, perm):
    left, strength, right = np.linalg.svd(z.T@h[perm]/len(z),full_matrices=False)
    a, b = z@left[:,:10], h@right[:10].T
    # Differences quantify disagreement between registration and realized activity.
    return np.column_stack([a,b,a*b,abs(a-b)]),strength[:10].tolist()


def run():
    raw, frame, y, g, f, first, change, _, opinions, _ = inputs()
    n=len(y)
    parent=first^change
    _,_,known,fresh=cohort(raw,y,f)
    wrong=parent[:n]!=y
    group=frame.PassengerId.str[:4].to_numpy()
    reg,spend=design(frame,group)
    raw_x=np.column_stack([reg,spend])
    z,h=embedding(reg),embedding(spend)
    order=np.arange(len(frame))
    real, strengths=shared(z,h,order)
    rng=np.random.default_rng(9851)
    perm=order.copy()
    for ix in frame.groupby([frame.HomePlanet.fillna('?'),frame.CryoSleep.astype('string').fillna('?')]).indices.values():
        perm[ix]=rng.permutation(ix)
    fake, control=shared(z,h,perm)
    variants=[raw_x,np.column_stack([raw_x,real]),np.column_stack([raw_x,fake])]
    common=((opinions[:n]>=.5)==parent[:n,None]).all(1)
    masks=dict(all=known,new_orders=known&fresh,seven=known&common,divided=known&~common)
    odds=lambda p:logit(np.clip(p,.002,.998))
    margin=np.where(parent,1.,-1.)*abs(odds(np.median(opinions[:,1:4],axis=1)))
    ordered=np.empty(n,int)
    for k,orders in enumerate(np.array_split(np.unique(g),8)):
        ordered[np.isin(g,orders)]=k
    report=dict(real_associations=strengths,permuted_associations=control)
    saved={}
    for tag,partition in [('random',f),('ordered',ordered)]:
        scores=np.empty((len(frame),3))
        for k in range(9):
            t=np.flatnonzero(partition!=k)
            q=np.flatnonzero(partition==k) if k<8 else np.arange(n,len(frame))
            assert not np.isin(group[t],group[q]).any()
            for j,x in enumerate(variants):
                model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,min_samples_leaf=30,
                    learning_rate=.065,l2_regularization=5,early_stopping=False,random_state=58)
                model.fit(x[t],y[t])
                scores[q,j]=model.predict_proba(x[q])[:,1]
        saved[tag]=scores
        agree=(scores>=.5)==parent[:,None]
        inc=((margin+odds(scores[:,1])-odds(scores[:,0])>=0)!=parent)&~agree[:,1]
        rules=dict(real=~agree[:,1],unique=(~agree[:,1])&agree[:,0]&agree[:,2],increment=inc,
                   confirmed=inc & ((margin+odds(scores[:,1])-odds(scores[:,2])>=0)!=parent))
        report[tag]=dict(standalone=[float(np.mean((p[:n]>=.5)==y)) for p in scores.T],rules={})
        for name,flag in rules.items():
            saved[tag+'_'+name]=flag
            report[tag]['rules'][name]=dict(scopes={s:count(flag[:n],m,wrong) for s,m in masks.items()},
                accuracy=float(np.mean((parent[:n]^flag[:n])[known]==y[known])),test=int(flag[n:].sum()))
        print(tag,report[tag],flush=True)
    np.testing.assert_allclose(saved['random'][n:],saved['ordered'][n:])
    dest=ART/'nov28_shared_manifest_spending_axes'
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=f,**saved)
    (ROOT/'reports/nov23_matched_learning_curve/shared_manifest_spending_axes.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
