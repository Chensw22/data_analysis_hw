"""Measure what size of visible error pattern can be recovered at fixed baseline accuracy."""
import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_error_tail_ranking_n23 import (ART,REPORT,SEEDS,panel,load,historical,strata,
                                          percentile,train,metrics,write)


def patterns(frame):
    a=frame.Age.fillna(28).to_numpy()
    # These coordinates are already available to the blind model. Only the
    # informed control knows their actual combination and signs.
    bill=frame[["RoomService","FoodCourt","Spa","VRDeck","ShoppingMall"]].fillna(0)
    v=np.log1p(bill).to_numpy()
    side=(frame.Side=="S").to_numpy()
    linear=.035*a+v[:,0]*.44-v[:,1]*.36+side*.8
    interaction=np.where(side,1.,-1.)*np.where(a>=32,1.,-1.)+np.sin(.7*(v[:,2]-v[:,4]))
    return linear,interaction


def rearrange(ids,truth,ref,base,pattern,amount,seed):
    # Each block is always wholly in fitting, calibration or validation. Outcome
    # totals are conserved separately by decision direction and risk within it.
    rng=np.random.default_rng(seed)
    err=truth[ids].copy()
    risk=np.abs(base[ids]-.5)
    cell=ref[ids].astype(int)*5+np.digitize(risk,np.quantile(risk,[.2,.4,.6,.8]))
    for key in np.unique(cell):
        ix=np.flatnonzero(cell==key)
        sel=ix[rng.permutation(len(ix))[:round(amount*len(ix))]]
        n=int(err[sel].sum())
        if len(sel):
            order=np.lexsort((rng.random(len(sel)),pattern[ids[sel]]))
            err[sel]=False
            if n:
                err[sel[order[-n:]]]=True
        assert int(err[ix].sum())==int(truth[ids[ix]].sum())
    return err


def flexible(train_x,train_y,test_x):
    clf=HistGradientBoostingClassifier(max_iter=110,max_leaf_nodes=9,min_samples_leaf=28,
              learning_rate=.065,l2_regularization=8,early_stopping=False,random_state=719)
    return clf.fit(train_x,train_y).predict_proba(test_x)[:,1]


def make_modes(linear,interaction):
    return [("real",linear,0.),("shuffled",np.random.default_rng(49).random(len(linear)),1.),
            ("linear_25",linear,.25),("linear_50",linear,.5),("linear_100",linear,1.),
            ("interact_50",interaction,.5),("interact_100",interaction,1.)]


def run():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    modes=make_modes(*patterns(frame))
    records,stable={},{}
    for name in [*map(str,SEEDS),"v131"]:
        base,ref,fold,x=panel(name,row,y,frame)
        eligible=ref==(base>=.5)
        true_errors=y!=ref
        outputs={}
        for mode,pattern,amount in modes:
            errors=true_errors.copy()
            results={k:np.zeros(len(row)) for k in ("base","tail","flexible","informed")}
            for f in np.unique(fold):
                ids=np.flatnonzero((fold==f)&eligible)
                blocks=[ids[v] for _,v in GroupKFold(5,shuffle=True,random_state=851).split(ids,groups=g[ids])]
                for j,part in enumerate(blocks):
                    errors[part]=rearrange(part,true_errors,ref,base,pattern,amount,478+j)
                for j,q in enumerate(blocks):
                    c=blocks[(j+1)%5]
                    t=np.concatenate([b for l,b in enumerate(blocks) if l not in (j,(j+1)%5)])
                    assert not np.intersect1d(g[t],g[c]).size and not np.intersect1d(g[q],g[np.r_[t,c]]).size
                    ids_cq=np.r_[c,q]
                    informed_x=np.c_[x[:,0],pattern]
                    # No rule key is revealed in the random-control or original task.
                    if mode in ("real","shuffled"):
                        informed_x=x[:,[0]]
                    sc=StandardScaler().fit(x[t])
                    xt,xcq=[sc.transform(x[idx]).clip(-5,5) for idx in (t,ids_cq)]
                    scores=dict(base=x[ids_cq,0],tail=xcq@train(xt,errors[t],.1),
                                flexible=flexible(x[t],errors[t],x[ids_cq]),
                                informed=flexible(informed_x[t],errors[t],informed_x[ids_cq]))
                    for k,vec in scores.items():
                        fit_rank,predict_rank=percentile(vec[:len(c)]),percentile(vec[len(c):])
                        m=LogisticRegression(C=.8).fit(logit(fit_rank)[:,None],errors[c])
                        results[k][q]=m.predict_proba(logit(predict_rank)[:,None])[:,1]
            assert np.array_equal(np.bincount(fold,weights=errors),np.bincount(fold,weights=true_errors))
            target=ref^errors
            outputs[mode]=dict(baseline=float((~errors).mean()),changed_status=int((errors!=true_errors).sum()),methods={})
            for k,risk in results.items():
                action=(risk>.5)&eligible
                r=metrics(action,ref,target,masks if mode=="real" else {})
                r["auc"]=float(roc_auc_score(errors[eligible],risk[eligible]))
                outputs[mode]["methods"][k]=r
                if mode=="real" and name!="v131":
                    stable.setdefault(k,[]).append((ref^action)==y)
            print(name,mode,outputs[mode]["methods"]["flexible"]["accuracy"],flush=True)
        records[name]=outputs
    write(REPORT/"power.json",dict(plans=records,real_stable={k:{s:int((np.logical_and.reduce(v)&m).sum())
          for s,m in masks.items()} for k,v in stable.items()}))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
