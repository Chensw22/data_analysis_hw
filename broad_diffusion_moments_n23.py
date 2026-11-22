"""Integrate conditional denoising loss exactly for Gaussian class moments."""
import json

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits

from broad_passenger_diffusion_n23 import ART,ROOT,SEEDS,compact_features,encode,panel,predictions,NEW,cohorts


def expected_error(x, mean, covariance, a):
    eye=np.eye(len(mean))
    predict=np.sqrt(1-a)*np.linalg.inv(a*covariance+(1-a)*eye)
    shift=(x-mean)@predict
    residual=eye-np.sqrt(1-a)*predict
    return a*shift**2+np.sum(residual**2,axis=0)


def run():
    row,y,base,old,_,_,_=panel()
    archived=[predictions(row,y,s,dict(**NEW,native_carte=("carte","p"))) for s in SEEDS]
    historical=(np.concatenate([old,np.stack([a[1] for a in archived])],2)>=.5)==y[None,:,None]
    hard=((base>=.5)!=y).all(0)
    never=hard&~historical.any(axis=(0,2))
    fresh=hard&~historical.all(0).any(1)
    previous=cohorts(row,y)["new50"]
    all_hits={}
    def cached(s,fold,path,key):
        with np.load(ART/f"nov23_{path}/{s}.npz") as z:
            for k,v in [("row",row),("y",y),("fold",fold)]:
                np.testing.assert_array_equal(z[k],v)
            return (z[key]>=.5)==y
    for path,key in [("broad_rotation_forest","rotation"),("broad_mondrian_partition","mondrian")]:
        previous|=fresh & np.logical_and.reduce([cached(s,a[2],path,key) for s,a in zip(SEEDS,archived)])
    assert previous.sum()==58
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Name","Transported")).iloc[row]
    groups=raw.PassengerId.str.split("_").str[0].to_numpy()
    frame=compact_features(raw)
    records={}
    for i,(s,(_,_,fold)) in enumerate(zip(SEEDS,archived)):
        ok0=(base[i]>=.5)==y
        decisions={k:np.full(len(y),np.nan) for k in ("mean","mean_cov")}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(groups[t],groups[q]).size
            x,fields=encode(frame,t)
            x=x.numpy().astype(float)
            weights=1/np.bincount(fields)[fields.numpy()]
            covs,means=[],[]
            for cls in (False,True):
                data=x[t[y[t]==cls]]
                covs.append(LedoitWolf().fit(data).covariance_)
                means.append(data.mean(0))
            pooled=(1-y[t].mean())*covs[0]+y[t].mean()*covs[1]
            for name,mat in (("mean",[pooled,pooled]),("mean_cov",covs)):
                score=np.zeros(len(q))
                for a in (.12,.35,.65,.88):
                    err=[expected_error(x[q],m,c,a)@weights for m,c in zip(means,mat)]
                    score+=err[0]-err[1]
                decisions[name][q]=score/4
        records[s]={}
        for k,v in decisions.items():
            assert np.isfinite(v).all()
            ok=(v>=0)==y
            all_hits.setdefault(k,[]).append(ok)
            records[s][k]=dict(accuracy=ok.mean(),fixed=int((ok&~ok0).sum()),harmed=int((ok0&~ok).sum()))
        for k in ("loss_vote","loss_fields","direct","base_direct","base_direct_loss"):
            all_hits.setdefault(k,[]).append(cached(s,fold,"broad_passenger_diffusion",k))
        c=all_hits["base_direct"][-1]
        d=all_hits["base_direct_loss"][-1]
        records[s]["add_loss"]=dict(fixed=int((d&~c).sum()),harmed=int((~d&c).sum()))
    coverage={}
    union=np.zeros(len(y),bool)
    for k,h in all_hits.items():
        st=np.all(h,0)
        novel=st&fresh&~previous
        union|=novel
        coverage[k]=dict(repeat=int((st&hard).sum()),never_any=int((np.any(h,0)&never).sum()),
                         never_stable=int((st&never).sum()),fresh=int(novel.sum()),
                         new_to46=int((st&fresh).sum()),overlap58=int((st&previous).sum()),
                         new_rows=row[novel].tolist())
    print(coverage,flush=True)
    out=ROOT/"reports/nov23_broad_passenger_diffusion"
    (out/"moments.json").write_text(json.dumps(dict(metrics=records,coverage=coverage,
                                               union_new=int(union.sum())),indent=2)+"\n")


if __name__=="__main__":
    with threadpool_limits(2):
        run()
