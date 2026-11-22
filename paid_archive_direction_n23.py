"""Audit repeated paid false negatives and train one-sided changes on separate bookings."""
import numpy as np
from scipy.special import expit,logit
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel,predictions
from broad_bill_exception_n23 import strata
from broad_booking_policy_n23 import EXTRA
from broad_diverse_rescue_n23 import ACTIONS,MORE
from broad_error_placement_control_n23 import SOURCES
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,historical,interval,load,write

REPORT=ROOT/"reports/nov23_paid_archive_direction"
COMMON={
    "service":("broad_service_transport","learned"),
    "partition":("broad_partition_evidence","native"),
    "latent":("broad_latent_scale","native_real"),
    "counts":("broad_count_supervision","native_similar"),
    "passenger_diff":("broad_passenger_differences/balanced","native_pair"),
    "billing_neutral":("broad_bill_exception","neutral"),
}


def census(rows,y,paid):
    orig,labels,bases,old,_,names,_=panel()
    np.testing.assert_array_equal(orig,rows)
    np.testing.assert_array_equal(labels,y)
    extended={**EXTRA,**dict(list(SOURCES.items())[:4])}
    pieces=[]
    for j,seed in enumerate(SEEDS):
        b,p,_=predictions(rows,y,seed,extended)
        np.testing.assert_array_equal(b,bases[j])
        pieces.append(np.column_stack([old[j],p,predictions(rows,y,seed,MORE)[1]]))
    bank=np.stack(pieces)
    names=list(names)+list(extended)+list(MORE)
    assert bank.shape==(3,len(y),65) and len(set(names))==65
    base_hit=(bases>=.5)==y
    accuracy=((bank>=.5)==y[None,:,None]).mean(1)
    near=(accuracy>=base_hit.mean(1)[:,None]-.01).all(0)
    can=(bank>=.5).all(0)
    consistent_neg=paid&(bases<.5).all(0)
    positive=consistent_neg&y
    negative=consistent_neg&~y
    assert positive.sum()==428
    groups={"comparable":near,"other":~near}
    coverage={}
    for group,idx in groups.items():
        reach=can[:,idx].any(1)
        other=can[:,~idx].any(1)
        coverage[group]=dict(members=int(idx.sum()),stable_paid_fixes=int((positive&reach).sum()),
                  exclusive_fixes=int((positive&reach&~other).sum()),
                  exposed_correct_negatives=int((negative&reach).sum()))
    reduced=(~near)&(np.array(names)!="hypothetical_outcome")
    coverage["other_without_hypothetical"]=dict(members=int(reduced.sum()),
        stable_paid_fixes=int((positive&can[:,reduced].any(1)).sum()),
        exposed_correct_negatives=int((negative&can[:,reduced].any(1)).sum()))
    individual={}
    for j,n in enumerate(names):
        individual[n]=dict(comparable=bool(near[j]),stable_fixes=int((can[:,j]&positive).sum()),
               stable_harms=int((can[:,j]&negative).sum()),accuracy=accuracy[:,j].tolist())
    return dict(positives=int(positive.sum()),negatives=int(negative.sum()),groups=coverage,
                neither=int((positive&~can.any(1)).sum()),models=individual)


def score(x,y,t,q,mode):
    sc=StandardScaler().fit(x[t])
    a,b=sc.transform(x[t]),sc.transform(x[q])
    if mode=="quadratic":
        terms=[]
        for value in (False,True):
            xs=a[y[t]==value]
            covariance=LedoitWolf().fit(xs)
            # The tiny diagonal floor also covers score-only and duplicated columns.
            cov=covariance.covariance_+.001*np.eye(a.shape[1])
            residual=b-covariance.location_
            distance=np.sum(residual*np.linalg.solve(cov,residual.T).T,axis=1)
            terms.append(np.log(len(xs)/len(t))-.5*(np.linalg.slogdet(cov)[1]+distance))
        return expit(terms[1]-terms[0])
    if mode=="lda":
        clf=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto")
    else:
        clf=LogisticRegression(C=.10,max_iter=400)
    return clf.fit(a,y[t]).predict_proba(b)[:,1]


def run():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(orig,row)
    np.testing.assert_array_equal(labels,y)
    paid=strata(frame)==2
    audit=census(row,y,paid)
    hist,report={},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_bill_exception/{name}.npz") as b:
            for k,v in (("row",row),("y",y)):
                np.testing.assert_array_equal(b[k],v)
            base,fold,ref=b["base"],b["fold"],b["ref"]
        cols=[]
        for folder,key in COMMON.values():
            with np.load(ART/f"nov23_{folder}/{name}.npz") as z:
                for k,v in (("row",row),("y",y),("fold",fold),("base",base)):
                    np.testing.assert_array_equal(z[k],v)
                cols.append(z[key])
        banks={"base":base[:,None],"common6":np.column_stack([base,*cols])}
        if name!="v131":
            b,d,f=predictions(row,y,int(name),{**ACTIONS,**MORE})
            np.testing.assert_array_equal(b,base)
            np.testing.assert_array_equal(f,fold)
            banks["direct29"]=np.column_stack([base,d,*cols[2:]])
        eligible=paid&(base<.5)&~ref
        summary={}
        for key,prob in banks.items():
            x=logit(prob.clip(.002,.998))
            for method in ("logistic","lda","quadratic"):
                out=ref.copy()
                matched=ref.copy()
                for k in np.unique(fold):
                    ids=np.flatnonzero((fold==k)&eligible)
                    for a,b in GroupKFold(3,shuffle=True,random_state=179).split(ids,groups=g[ids]):
                        t,q=ids[a],ids[b]
                        assert not np.intersect1d(g[t],g[q]).size
                        out[q]=score(x,y,t,q,method)>=.5
                        # Same number of flips in each held-out booking batch.
                        # The control chooses the baseline negatives closest to
                        # .5, without consulting the query labels.
                        n=int(out[q].sum())
                        if n:
                            matched[q[np.argsort(-base[q],kind="stable")[:n]]]=True
                hit=out==y
                delta=hit.astype(int)-(ref==y)
                label=key+"_"+method
                summary[label]=dict(accuracy=float(hit.mean()),fixed=int((delta>0).sum()),
                       harmed=int((delta<0).sum()),matched_count_fixes=int((matched&~ref&y).sum()),by_fold=[int(delta[fold==f].sum()) for f in np.unique(fold)])
                if name!="v131":
                    hist.setdefault(label,[]).append((hit,delta))
        report[name]=summary
    stable={}
    for k,values in hist.items():
        h=np.logical_and.reduce([p for p,d in values])
        stable[k]=dict(paid_428=int((h&paid&y&masks["repeat"]).sum()),shared509=int((h&masks["unresolved"]).sum()),
                       net_pp_ci95=interval(np.mean([d for p,d in values],axis=0),g))
    write(REPORT/"results.json",dict(census=audit,plans=report,stable=stable,common=COMMON))


if __name__=="__main__":
    with threadpool_limits(1):
        run()
