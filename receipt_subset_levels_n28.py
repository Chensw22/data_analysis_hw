"""Test overlap between service-specific spending levels beyond the five separate charges."""
import itertools
import json
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from manifest_activity_transitions_n27 import design
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART,cohort,count
from unanimous_error_reach_n27 import ROOT,inputs


def bills(frame):
    amounts=frame[FEES].to_numpy(float)
    levels=[]
    for col in amounts.T:
        nonzero=np.sort(col[col>0])
        levels.append(np.where(col>0,np.searchsorted(nonzero,col,side='right')/len(nonzero),0.))
    return np.column_stack(levels)


def matrices(frame):
    reg,act=design(frame)
    scores=bills(frame)
    base=np.column_stack([reg,act,scores])
    subsets=[group for size in range(2,6) for group in itertools.combinations(range(5),size)]
    together=np.column_stack([scores[:,ix].min(axis=1) for ix in subsets])
    # Compare to a product basis that has identical support and the same number of columns.
    product=np.column_stack([scores[:,ix].prod(axis=1) for ix in subsets])
    richer=[]
    for additional in [together,product]:
        parts=[base,additional]
        for planet in ['Earth','Europa','Mars']:
            member=frame.HomePlanet.eq(planet).to_numpy(float)[:,None]
            parts.extend([member*scores,member*additional])
        richer.append(np.column_stack(parts))
    separate=np.column_stack([base,*[frame.HomePlanet.eq(p).to_numpy(float)[:,None]*scores for p in ['Earth','Europa','Mars']]])
    return [separate,*richer]


def run():
    raw,frame,y,g,f,first,change,_,opinions,_=inputs()
    n=len(y); parent=first^change;wrong=parent[:n]!=y
    _,_,known,fresh=cohort(raw,y,f)
    common=((opinions[:n]>=.5)==parent[:n,None]).all(1)
    multi=(frame[FEES].gt(0).sum(axis=1)>=2).to_numpy()
    masks=dict(all=known,new_orders=known&fresh,seven=known&common)
    views=matrices(frame)
    groups=frame.PassengerId.str[:4].to_numpy()
    ords=np.empty(n,int)
    for k,part in enumerate(np.array_split(np.unique(g),8)):ords[np.isin(g,part)]=k
    odds=lambda x:logit(np.clip(x,.002,.998))
    anchor=np.where(parent,1.,-1.)*abs(odds(np.median(opinions[:,1:4],axis=1)))
    saved,report={},{}
    for tag,partition in [('random',f),('ordered',ords)]:
        scores=np.empty((len(frame),3))
        for k in range(9):
            t=np.flatnonzero(partition!=k)
            q=np.flatnonzero(partition==k) if k<8 else np.arange(n,len(frame))
            assert not np.isin(groups[t],groups[q]).any()
            for j,x in enumerate(views):
                model=LogisticRegression(C=2,solver='newton-cholesky',max_iter=80,tol=1e-7).fit(x[t],y[t])
                scores[q,j]=model.decision_function(x[q])
        oppose=(scores>=0)!=parent[:,None]
        incremental=((anchor+scores[:,1]-scores[:,0]>=0)!=parent)&multi&oppose[:,1]
        rules=dict(joint_minimum=oppose[:,1]&multi,increment=incremental,
                   corroborated=incremental&oppose[:,2],
                   distinct=incremental&((anchor+scores[:,1]-scores[:,2]>=0)!=parent))
        report[tag]=dict(native_accuracy=np.mean((scores[:n]>=0)==y[:,None],axis=0).tolist(),rules={})
        saved[tag]=scores
        for key,flag in rules.items():
            saved[tag+'_'+key]=flag
            report[tag]['rules'][key]=dict(scopes={s:count(flag[:n],m,wrong) for s,m in masks.items()},
                accuracy=float(np.mean((parent[:n]^flag[:n])[known]==y[known])),test=int(flag[n:].sum()))
        print(tag,report[tag],flush=True)
    np.testing.assert_allclose(saved['random'][n:],saved['ordered'][n:])
    out=ART/'nov28_receipt_subset_levels'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=f,**saved)
    (ROOT/'reports/nov23_matched_learning_curve/receipt_subset_levels.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):run()
