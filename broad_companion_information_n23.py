"""Separate the value of another passenger's outcome from knowing pair concordance."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import ART,ROOT,SEEDS,historical,load,strata,evaluate,write
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import BILLS
from broad_refit_dissent_n23 import features

OUT=ART/"nov23_companion_information"
REPORT=ROOT/"reports/nov23_companion_information"


def edges(groups,ids):
    buckets=pd.Series(ids).groupby(groups[ids]).apply(list)
    return np.asarray([p for bucket in buckets for p in combinations(bucket,2)],int).reshape(-1,2)


def auxiliary(frame,x,e,p):
    i,j=e.T
    bill=frame[BILLS].fillna(0).to_numpy()
    pay=(bill>0).any(1)
    sleep=frame.CryoSleep.to_numpy()
    same_berth=(frame.Deck.to_numpy()[i]==frame.Deck.to_numpy()[j]) & (frame.Side.to_numpy()[i]==frame.Side.to_numpy()[j]) & (frame.CabinNumber.to_numpy()[i]==frame.CabinNumber.to_numpy()[j])
    basis=np.column_stack([np.ones(len(i)),p[i]+p[j]-1.,abs(p[i]-p[j]),pay[i]==pay[j],
                           same_berth,(sleep[i]==sleep[j])&(sleep[i]!="__MISSING__")])
    pair=np.column_stack([basis,np.abs(x[i]-x[j]),(x[i]+x[j])/2])
    return basis,pair


def bounds(p,q):
    product=p*q
    return np.minimum(p,q)-product, product-np.maximum(p+q-1,0.)


def covariance(p,q,x,w):
    pos,neg=bounds(p,q)
    r=np.tanh(x@w)
    scale=np.where(r>=0,pos,neg)
    return r*scale,(1-r*r)*scale


def train_coupling(p,y,e,x,groups):
    i,j=e.T
    n=np.bincount(groups[np.unique(e)])[groups[i]]
    weight=1/(n-1)
    a,b=p[i],p[j]
    observed=np.where(y[i],a,1-a)*np.where(y[j],b,1-b)
    sign=np.where(y[i]==y[j],1,-1)
    def objective(w):
        cov,deriv=covariance(a,b,x,w)
        prob=(observed+sign*cov).clip(1e-15)
        return -weight@np.log(prob)+3*(w@w), -(weight*sign*deriv/prob)@x+6*w
    fit=minimize(objective,np.full(x.shape[1],.01),jac=True,method="L-BFGS-B",options=dict(maxiter=400))
    if not fit.success:
        fit=minimize(lambda w:objective(w)[0],fit.x,method="Powell",options=dict(maxiter=400))
    assert fit.success
    return fit.x


def select(e,score):
    # Choose one peer using the expected variance reduction from a reliable observation.
    # The rule only sees the two marginal probabilities and the training estimate of covariance.
    focal=np.r_[e[:,0],e[:,1]]
    partner=np.r_[e[:,1],e[:,0]]
    order=np.lexsort((partner,-score,focal))
    _,first=np.unique(focal[order],return_index=True)
    indices=order[first]
    assert np.all(focal[indices]!=partner[indices])
    return focal[indices],partner[indices],indices%len(e)


def conditional(p,q,c):
    # The four cells of the joint distribution, conditional on neither unknown outcome.
    yes=p*q+c
    no=(1-p)*(1-q)+c
    agree=yes+no
    return np.stack([(p-yes)/(1-q),yes/q],1),np.stack([(p-yes)/(1-agree),yes/agree],1),agree


def posterior(parts,prob):
    return parts[:,0]*(1-prob)+parts[:,1]*prob


def diagnostic(ch,prior,truth,quality):
    # A binary report has symmetric, conditionally independent error. Integrate
    # over both possible reports exactly; no target labels enter a deployable policy.
    p0=(1-quality)*prior/((1-quality)*prior+quality*(1-prior))
    p1=quality*prior/(quality*prior+(1-quality)*(1-prior))
    w=np.where(truth,quality,1-quality)
    return posterior(ch,p0),posterior(ch,p1),w


def run():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    scope=dict(**masks,repeat_paid=masks["repeat"]&(strata(frame)==2)&y,
               repeat_zero=masks["repeat"]&(strata(frame)==0)&~y)
    features_x=features(frame)[0]
    OUT.mkdir(exist_ok=True)
    plans,stable={},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{name}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            fold,p,ref=z["fold"],z["base"].clip(.0001,.9999),z["ref"]
        signals=[p]
        for src,k in (("broad_neural_path","epoch_80"),("broad_boosting_path","trees_800"),
                      ("broad_training_membership","ordinary"),("service_assignment","factor_real")):
            with np.load(ART/f"nov23_{src}/{name}.npz") as z:
                for f,v in (("row",row),("fold",fold),("y",y)):
                    np.testing.assert_array_equal(z[f],v)
                signals.append(z[k])
        opinions=np.stack([logit(v.clip(.001,.999)) for v in signals],1)
        scores={k:p.copy() for k in ("base","peer_proxy","peer_shuffle","agreement_proxy","agreement_shuffle",
                                      "revealed_peer","revealed_agreement")}
        ref_hit=ref==y
        expected={k:{v:np.zeros(len(y)) for v in (.5,.65,.8,.95,1.)} for k in ("peer","agreement")}
        selected=np.full(len(y),-1)
        for f in np.unique(fold):
            ids=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=721).split(ids,groups=g[ids]):
                t,q=ids[a],ids[b]
                assert not np.intersect1d(g[t],g[q]).size
                et,eq=edges(g,t),edges(g,q)
                xt,dt=auxiliary(frame,features_x,et,p)
                xq,dq=auxiliary(frame,features_x,eq,p)
                w=train_coupling(p,y,et,xt,g)
                left,right=eq.T
                c,_=covariance(p[left],p[right],xq,w)
                benefit=np.r_[c*c/(p[right]*(1-p[right])),c*c/(p[left]*(1-p[left]))]
                i,j,link=select(eq,benefit)
                selected[i]=j
                peer,agr,prior=conditional(p[i],p[j],c[link])
                np.testing.assert_allclose(posterior(peer,p[j]),p[i],atol=1e-10)
                np.testing.assert_allclose(posterior(agr,prior),p[i],atol=1e-10)
                agree_train=y[et[:,0]]==y[et[:,1]]
                clf=HistGradientBoostingClassifier(max_iter=95,max_leaf_nodes=9,l2_regularization=8,
                        min_samples_leaf=30,early_stopping=False,random_state=817)
                estimate=clf.fit(dt,agree_train).predict_proba(dq[link])[:,1]
                est_peer=calibrated(opinions[t],opinions[j],y,t)
                # Match placebo replacements on the model's prior for the same observation.
                rng=np.random.default_rng(539)
                for key,ch,initial,new in (("peer",peer,p[j],est_peer),("agreement",agr,prior,estimate)):
                    scores[key+"_proxy"][i]=posterior(ch,new)
                    ix=np.arange(len(i))
                    for block in np.array_split(np.argsort(initial,kind="stable"),10):
                        ix[block]=rng.permutation(block)
                    scores[key+"_shuffle"][i]=posterior(ch,new[ix])
                for key,ch,initial,outcome in (("peer",peer,p[j],y[j]),("agreement",agr,prior,y[i]==y[j])):
                    scores["revealed_"+key][i]=posterior(ch,outcome)
                    for accuracy,values in expected[key].items():
                        lo,hi,w1=diagnostic(ch,initial,outcome,accuracy)
                        locked=ref[i]!=(p[i]>=.5)
                        oklo=np.where(locked,ref[i],lo>=.5)==y[i]
                        okhi=np.where(locked,ref[i],hi>=.5)==y[i]
                        values[i]=w1*okhi+(1-w1)*oklo-ref_hit[i]
        scores={"update_"+k:np.clip(v,1e-9,1-1e-9) for k,v in scores.items()}
        results,hits=evaluate(y,p,ref,scores,fold,name=="v131",scope)
        noise={k:{str(rate):dict(accuracy=float(ref_hit.mean()+gain.mean()),fixes=float(gain.clip(0).sum()),
                        harms=float(-gain.clip(max=0).sum())) for rate,gain in by_rate.items()} for k,by_rate in expected.items()}
        for k in expected:
            assert np.isclose(noise[k]["1.0"]["accuracy"],results["update_revealed_"+k]["accuracy"])
            assert np.isclose(noise[k]["0.5"]["accuracy"],ref_hit.mean())
        eligible=selected>=0
        matched=y[eligible]==y[selected[eligible]]
        plans[name]=dict(models=results,noise=noise,eligible=int(eligible.sum()),agreement=float(matched.mean()),
                         eligible_scopes={k:int((eligible&m).sum()) for k,m in scope.items()})
        if name!="v131":
            for k,(hit,_) in hits.items():
                stable.setdefault(k,[]).append(hit)
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,base=p,ref=ref,selected=selected,**scores)
        print(name,[(k,d["fixed"],d["harmed"]) for k,d in results.items()],flush=True)
    counts={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in scope.items()} for k,v in stable.items()}
    write(REPORT/"information.json",dict(plans=plans,stable=counts))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
