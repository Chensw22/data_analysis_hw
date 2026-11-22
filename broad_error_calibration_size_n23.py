"""On identical unseen bookings, vary the number of labels available to an error detector."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.metrics import log_loss,roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import ART, ROOT, SEEDS, load, historical, write

OUT=ART/"nov23_error_calibration_size"
REPORT=ROOT/"reports/nov23_error_calibration_size"
SOURCES=(("broad_basket_balance","weighted"),("service_assignment","factor_real"),
         ("broad_neural_path","epoch_80"),("broad_boosting_path","trees_800"),
         ("broad_training_membership","ordinary"))
LEVELS=(128,512,1024,"full")


def cache(name,row,y):
    views=[]
    for family,key in SOURCES:
        with np.load(ART/f"nov23_{family}/{name}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            if not views:
                fold,base,ref=z["fold"],z["base"],z["ref"]
            else:
                for k,v in (("fold",fold),("ref",ref),("base",base)):
                    np.testing.assert_array_equal(z[k],v)
            views.append(logit(z[key].clip(.002,.998)))
    return np.stack(views,1),logit(base.clip(.002,.998)),fold,ref


def problem(anchor,x,y):
    sign=np.where(anchor>=0,-1.,1.)
    different=sign[:,None]*(x-anchor[:,None])
    features=np.c_[np.ones(len(y)),abs(anchor)/2,sign,different/2,sign[:,None]*different/2,
                   np.mean((x>=0)!=(anchor>=0)[:,None],1)]
    return features,-abs(anchor),y!=(anchor>=0)


def fit(rows,problems,x,offset):
    matrix,prior,target=[np.concatenate([p[k][rows] for p in problems]) for k in range(3)]
    def fn(beta):
        v=matrix@beta+prior
        objective=(np.logaddexp(0,v)-v*target).sum()/len(problems)+6*(beta@beta)
        grad=matrix.T@(expit(v)-target)/len(problems)+12*beta
        return objective,grad
    opt=minimize(fn,np.zeros(matrix.shape[1]),jac=True,method="L-BFGS-B")
    assert opt.success,opt.message
    return expit(offset+x@opt.x)


def conditional_auc(miss,score,risk,pred,use):
    # Compare wrong and right passengers within the same 2-percentage-point
    # prior-error bin and baseline direction, weighting each eligible pair equally.
    band=(risk*50).astype(int)*2+pred
    area=pairs=0.
    for k in np.unique(band[use]):
        sel=use&(band==k)
        n=miss[sel].sum()
        count=n*(sel.sum()-n)
        if count:
            area+=count*roc_auc_score(miss[sel],score[sel])
            pairs+=count
    return area/pairs


def metrics(p,risk,truth,pred,locked):
    miss=truth!=pred
    all_rows=np.ones(len(p),bool)
    high=risk<=.2
    flip=(p>.5)&~locked
    hit=truth==(pred^flip)
    scores=dict(accuracy=float(hit.mean()),fix=int((flip&miss).sum()),harm=int((flip&~miss).sum()),
                log_loss=float(log_loss(miss,p)))
    for key,use in (("all",all_rows),("high_confidence",high)):
        ids=np.flatnonzero(use)
        ordered=ids[np.argsort(-p[ids],kind="stable")[:int(np.ceil(len(ids)*.1))]]
        scores[key]=dict(n=int(use.sum()),misses=int(miss[use].sum()),auc=float(roc_auc_score(miss[use],p[use])),
                         matched_auc=conditional_auc(miss,p,risk,pred,use),top_decile_misses=int(miss[ordered].sum()),
                         inspected=len(ordered))
    return scores


def run():
    row,y,_,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    OUT.mkdir(exist_ok=True)
    report,history={},{}
    for name in [*map(str,SEEDS),"v131"]:
        x,anchor,fold,ref=cache(name,row,y)
        native=problem(anchor,x,y)
        # Other anchors are derived only from the frozen out-of-fold scores.
        # The same passenger contributes total weight one, also with six anchors.
        augmented=[native,*[problem((anchor+v)/2,x,y) for v in x.T]]
        probs={}
        realized={str(level):[] for level in LEVELS}
        for f in np.unique(fold):
            ids=np.flatnonzero(fold==f)
            for j,(a,b) in enumerate(GroupKFold(8,shuffle=True,random_state=631).split(ids,groups=g[ids])):
                t,q=ids[a],ids[b]
                assert not np.intersect1d(g[t],g[q]).size
                unique,cnt=np.unique(g[t],return_counts=True)
                for draw in range(3):
                    order=np.random.default_rng(71+int(f)*53+j*7+draw).permutation(len(cnt))
                    mass=np.cumsum(cnt[order])
                    for level in LEVELS:
                        end=len(cnt) if level=="full" else np.argmin(abs(mass-level))+1
                        src=t[np.isin(g[t],unique[order[:end]])]
                        realized[str(level)].append(len(src))
                        for mode,ps in (("native",[native]),("replicas",augmented)):
                            tag=mode+"_"+str(level)
                            dest=probs.setdefault(tag,np.zeros((3,len(y))))
                            if level=="full" and draw:
                                dest[draw,q]=dest[0,q]
                            else:
                                dest[draw,q]=fit(src,ps,native[0][q],native[1][q])
        locked=ref!=(anchor>=0)
        risk=np.where(ref,expit(-anchor),expit(anchor))
        scores,averages={},{}
        for key,variants in probs.items():
            assert np.all((variants>0)&(variants<1))
            mean=variants.mean(0)
            # A locked historical revision remains in force; its error probability
            # is the complement of that computed for the underlying anchor vote.
            assessed=np.where(locked,1-mean,mean)
            scores[key]=metrics(assessed,risk,y,ref,locked)
            draw_net=[]
            individual=[]
            for v in variants:
                mv=metrics(np.where(locked,1-v,v),risk,y,ref,locked)
                individual.append(mv)
                flip=(v>.5)&~locked
                draw_net.append(int(((y==(ref^flip)).astype(int)-(y==ref)).sum()))
            scores[key]["single_draw_net"]=draw_net
            scores[key]["single_draw"]=individual
            averages[key]=assessed
            if name!="v131":history.setdefault(key,[]).append((ref^((mean>.5)&~locked))==y)
        scores["prior"]=metrics(risk,risk,y,ref,locked)
        report[name]=dict(methods=scores,passengers={k:[min(v),max(v),float(np.mean(v))] for k,v in realized.items()})
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,ref=ref,risk=risk,**averages)
        print(name,[(k,v["fix"]-v["harm"]) for k,v in scores.items()],flush=True)
    stable={k:{s:int((np.logical_and.reduce(h)&m).sum()) for s,m in masks.items()} for k,h in history.items()}
    write(REPORT/"results.json",dict(plans=report,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
